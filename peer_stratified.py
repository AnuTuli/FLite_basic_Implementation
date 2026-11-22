import os
import time
import tracemalloc
import pickle

from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, mean_absolute_error, mean_squared_error
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, Flatten, Reshape
)


DATA_FILE = "Cropdata.csv"
LOCAL_EPOCHS = 50
BATCH_SIZE = 32
TEST_SIZE = 0.20
RANDOM_STATE = 42

INPUT_SHAPE = None  

AES_KEY_ENV = "PEER_AES_KEY_HEX"


def get_aes_key():
    """Read a shared 16-byte AES key from the environment (hex encoded)."""
    key_hex = os.environ.get(AES_KEY_ENV)
    if not key_hex:
        raise RuntimeError(
            f"Set {AES_KEY_ENV} to a shared 32-character hex string (16 bytes). "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(16))\""
        )
    try:
        key = bytes.fromhex(key_hex.strip())
    except ValueError as exc:
        raise ValueError(f"{AES_KEY_ENV} must be valid hexadecimal.") from exc
    if len(key) != 16:
        raise ValueError(f"{AES_KEY_ENV} must decode to exactly 16 bytes (AES-128).")
    return key



def encrypt_weights(weights, aes_key, sender_id, round_id):
    """Serialize model weights and encrypt/authenticate them with AES-128-GCM."""
    payload = pickle.dumps(weights, protocol=pickle.HIGHEST_PROTOCOL)
    aesgcm = AESGCM(aes_key)
    nonce = os.urandom(12)  
    aad = f"peer-weights|round={round_id}|sender={sender_id}".encode("utf-8")
    ciphertext = aesgcm.encrypt(nonce, payload, aad)
    return {
        "sender_id": sender_id,
        "round_id": round_id,
        "nonce": nonce,
        "aad": aad,
        "ciphertext": ciphertext,
    }



def decrypt_weights(encrypted_payload, aes_key, expected_round):
    """Authenticate and decrypt peer weights; reject stale or tampered payloads."""
    if encrypted_payload["round_id"] != expected_round:
        raise ValueError("Received model payload from a different training round.")
    aesgcm = AESGCM(aes_key)
    plaintext = aesgcm.decrypt(
        encrypted_payload["nonce"],
        encrypted_payload["ciphertext"],
        encrypted_payload["aad"],
    )
    weights = pickle.loads(plaintext)
    return weights


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim



    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(1, input_shape[1], self.feature_dim),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)



    def call(self, inputs):
        return inputs + self.positional_embedding



    def get_config(self):
        config = super().get_config()
        config.update({"feature_dim": self.feature_dim})
        return config



def transformer_encoder(inputs, num_heads=4, key_dim=64, ff_dim=128, dropout_rate=0.3):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = Add()([inputs, attention_output])
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)


    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Add()([attention_output, ff_output])
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output)
    return ff_output



def slm_transformer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    feature_embedding = Dense(64, activation="relu")(inputs)
    feature_embedding = Reshape((1, 64))(feature_embedding)
    feature_embedding = PositionalEncoding(feature_dim=64)(feature_embedding)


    transformer_output = transformer_encoder(feature_embedding)
    transformer_output = transformer_encoder(transformer_output)
    flattened_output = Flatten()(transformer_output)
    dense_output = Dense(64, activation="relu")(flattened_output)
    dense_output = Dropout(0.3)(dense_output)
    outputs = Dense(num_classes, activation="softmax")(dense_output)


    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model



def init_model(input_shape, num_classes):
    return slm_transformer_model(input_shape, num_classes)

def weighted_average_weights(weight_sets, sample_counts):
    """Sample-count-weighted average of model weight arrays."""
    if not weight_sets:
        raise ValueError("No model weights supplied for aggregation.")
    if len(weight_sets) != len(sample_counts):
        raise ValueError("weight_sets and sample_counts lengths do not match.")
    total_samples = int(sum(sample_counts))
    if total_samples <= 0:
        raise ValueError("Total sample count must be positive.")

    averaged = []
    for layer_idx in range(len(weight_sets[0])):
        averaged_layer = np.zeros_like(weight_sets[0][layer_idx], dtype=np.float64)
        for weights, count in zip(weight_sets, sample_counts):
            averaged_layer += weights[layer_idx] * (count / total_samples)
        averaged.append(averaged_layer.astype(weight_sets[0][layer_idx].dtype))
    return averaged



def get_peer_ids(client_id, num_clients):
    if num_clients <= 1:
        return []
    peers = [(client_id - 1) % num_clients, (client_id + 1) % num_clients]
    return sorted(set(j for j in peers if j != client_id))



def exchange_encrypted_weights(encrypted_by_client, peer_ids):
    """In-process exchange simulation: return encrypted payloads from listed peers."""
    return [encrypted_by_client[peer_id] for peer_id in peer_ids]



def stratified_client_partition(X, y, num_clients, random_state=42):
    """Distribute samples among clients while approximately preserving class proportions.

    Each class is shuffled and split across clients separately. Remainder samples
    for each class are assigned to randomly selected clients, reducing systematic
    client-size bias. Supports classes with fewer samples than the client count.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")
    if num_clients > len(y):
        raise ValueError("Number of clients exceeds the number of training samples.")

    rng = np.random.default_rng(random_state)
    client_indices = [[] for _ in range(num_clients)]

    for class_label in np.unique(y):
        class_indices = np.flatnonzero(y == class_label)
        rng.shuffle(class_indices)
        base_count, remainder = divmod(len(class_indices), num_clients)
        class_client_order = rng.permutation(num_clients)
        class_counts = np.full(num_clients, base_count, dtype=int)
        if remainder:
            class_counts[class_client_order[:remainder]] += 1

        cursor = 0
        for client_id in range(num_clients):
            count = class_counts[client_id]
            if count:
                client_indices[client_id].extend(class_indices[cursor:cursor + count])
                cursor += count

    client_data, client_labels = [], []
    for indices in client_indices:
        indices = np.asarray(indices, dtype=int)
        rng.shuffle(indices)
        client_data.append(X[indices])
        client_labels.append(y[indices])

    return client_data, client_labels

start_time = time.time()
tracemalloc.start()


num_clients = int(input("Enter number of clients: "))
num_rounds = int(input("Enter number of rounds: "))
if num_clients < 1:
    raise ValueError("Number of clients must be at least 1.")
if num_rounds < 1:
    raise ValueError("Number of rounds must be at least 1.")

AES_KEY = get_aes_key()

data = pd.read_csv(DATA_FILE)
if data.shape[1] < 2:
    raise ValueError("Dataset must contain at least one feature column and a target column.")


X_all = data.iloc[:, :-1].values
labels_all = data.iloc[:, -1].values
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_all, labels_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels_all
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_test = scaler.transform(X_test_raw).astype(np.float32)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw).astype(np.int64)
y_test = label_encoder.transform(y_test_raw).astype(np.int64)
input_shape = (X_train.shape[1],)
num_classes = len(label_encoder.classes_)

client_data, client_labels = stratified_client_partition(
    X_train, y_train, num_clients, random_state=RANDOM_STATE
)
client_sample_counts = [len(x) for x in client_data]
print("Client class distributions (counts):")
for client_id, labels in enumerate(client_labels):
    class_counts = np.bincount(labels, minlength=num_classes)
    print(f"Client {client_id}: {dict(enumerate(class_counts.tolist()))}")
if any(count == 0 for count in client_sample_counts):
    raise ValueError("Number of clients exceeds the number of training samples.")

client_models = [init_model(input_shape, num_classes) for _ in range(num_clients)]

for round_id in range(num_rounds):
    print(f"\n*****Round {round_id + 1}/{num_rounds}*****")

    local_weights = []
    for client_id in range(num_clients):
        client_model = init_model(input_shape, num_classes)
        client_model.set_weights(client_models[client_id].get_weights())
        client_model.fit(
            client_data[client_id],
            client_labels[client_id],
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        local_weights.append(client_model.get_weights())

    encrypted_by_client = {
        client_id: encrypt_weights(local_weights[client_id], AES_KEY, client_id, round_id)
        for client_id in range(num_clients)
    }


    client_aggregated_weights = []
    for client_id in range(num_clients):
        peer_ids = get_peer_ids(client_id, num_clients)
        peer_payloads = exchange_encrypted_weights(encrypted_by_client, peer_ids)

        received_weights = [
            decrypt_weights(payload, AES_KEY, expected_round=round_id)
            for payload in peer_payloads
        ]

        weights_to_aggregate = [local_weights[client_id]] + received_weights
        counts_to_aggregate = [client_sample_counts[client_id]] + [
            client_sample_counts[peer_id] for peer_id in peer_ids
        ]
        aggregated_weights = weighted_average_weights(
            weights_to_aggregate, counts_to_aggregate
        )
        client_aggregated_weights.append(aggregated_weights)

    for client_id, agg_weights in enumerate(client_aggregated_weights):
        client_models[client_id].set_weights(agg_weights)

    accuracies = []
    maes = []
    rmses = []
    for client_id, agg_weights in enumerate(client_aggregated_weights):
        eval_model = init_model(input_shape, num_classes)
        eval_model.set_weights(agg_weights)
        y_pred = np.argmax(eval_model.predict(X_test, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        accuracies.append(acc)
        maes.append(mae)
        rmses.append(rmse)
        print(f"Client {client_id}: accuracy={acc:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
        print(classification_report(y_test, y_pred, labels=np.arange(num_classes),
                                target_names=[str(c) for c in label_encoder.classes_],
                                zero_division=0))

    print(f"Round {round_id + 1} mean client accuracy: {np.mean(accuracies):.4f}")
    print(f"Round {round_id + 1} mean MAE: {np.mean(maes):.4f}")
    print(f"Round {round_id + 1} mean RMSE: {np.mean(rmses):.4f}")
    print("Personalized client models updated; no global model aggregation performed.")

end_time = time.time()
print("\nTotal time:", end_time - start_time, "seconds")
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1e6:.2f} MB")
print(f"Peak memory usage: {peak / 1e6:.2f} MB")
tracemalloc.stop()
