import os
import time
import pickle
import secrets
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, MultiHeadAttention, LayerNormalization,
    Add, Flatten, Reshape
)

t1 = time.time()

AES_KEY_ENV = "PEER_AES_KEY_HEX"

def get_aes_key():
    key_hex = os.getenv(AES_KEY_ENV, "").strip()
    if not key_hex:
        raise RuntimeError(
            f"Set {AES_KEY_ENV} to a 32-character hex string (16 bytes). "
            "Generate with: python -c \"import secrets; print(secrets.token_hex(16))\""
        )
    try:
        key = bytes.fromhex(key_hex)
    except ValueError as exc:
        raise ValueError(f"{AES_KEY_ENV} must contain hexadecimal characters only.") from exc
    if len(key) != 16:
        raise ValueError(f"{AES_KEY_ENV} must decode to exactly 16 bytes for AES-128.")
    return key

AES_KEY = get_aes_key()

def encrypt_model_payload(payload, key):
    """Serialize and encrypt model payload using AES-128-GCM with a fresh 12-byte nonce."""
    nonce = secrets.token_bytes(12)
    plaintext = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    return nonce + ciphertext

def decrypt_model_payload(encrypted_payload, key):
    """Decrypt and authenticate an AES-GCM model payload."""
    if len(encrypted_payload) < 12 + 16:
        raise ValueError("Encrypted model payload is too short.")
    nonce, ciphertext = encrypted_payload[:12], encrypted_payload[12:]
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    return pickle.loads(plaintext)

num_clients = int(input("Enter number of clients: "))
num_round = int(input("Enter number of rounds: "))

peer_files = [
    "converted_Sangi.csv",
    "converted_Satara.csv",
    "converted_Solahpur.csv",
    "converted_Pune.csv",
    "converted_Kolahpur.csv",
]

if num_clients < 2:
    raise ValueError("At least two clients are required for peer exchange.")
if num_clients > len(peer_files):
    raise ValueError(f"num_clients cannot exceed available peer datasets ({len(peer_files)}).")

data_all = [pd.read_csv(file) for file in peer_files[:num_clients]]
if any(data.shape[1] < 2 for data in data_all):
    raise ValueError("Each peer CSV must contain features and a final target column.")


all_labels = np.concatenate([data.iloc[:, -1].values for data in data_all])
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
num_classes = len(label_encoder.classes_)

feature_counts = [data.iloc[:, :-1].shape[1] for data in data_all]
if len(set(feature_counts)) != 1:
    raise ValueError("All peer datasets must have the same number of feature columns.")
num_features = feature_counts[0]
input_shape = (1, num_features)

X_train_peer, y_train_peer, X_test_peer, y_test_peer = [], [], [], []
for data in data_all:
    X_raw = data.iloc[:, :-1].values.astype(np.float32)
    y = label_encoder.transform(data.iloc[:, -1].values).astype(np.int32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    class_counts = Counter(y)
    can_stratify = len(class_counts) > 1 and min(class_counts.values()) >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if can_stratify else None
    )
    X_train_peer.append(X_train)
    y_train_peer.append(y_train)
    X_test_peer.append(X_test)
    y_test_peer.append(y_test)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim

    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            shape=(1, input_shape[1], self.feature_dim),
            initializer="random_normal",
            trainable=True,
            name="positional_embedding",
        )

    def call(self, inputs):
        return inputs + self.positional_embedding

    def get_config(self):
        config = super().get_config()
        config.update({"feature_dim": self.feature_dim})
        return config

def transformer_encoder(inputs, num_heads=4, key_dim=64, ff_dim=128, dropout_rate=0.1):
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
    output = Dense(num_classes, activation="softmax")(dense_output)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def init_model():
    return slm_transformer_model(input_shape, num_classes)

def train_local_model(model, X, y):
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    return model

def get_peer_ids(client_id, n_clients):
    peers = sorted({(client_id - 1) % n_clients, (client_id + 1) % n_clients})
    return [j for j in peers if j != client_id]

def weighted_federated_average(weight_sets, sample_counts):
    total_samples = float(sum(sample_counts))
    if total_samples <= 0:
        raise ValueError("Total aggregation sample count must be positive.")
    averaged = []
    for layer_idx in range(len(weight_sets[0])):
        layer_avg = np.zeros_like(weight_sets[0][layer_idx], dtype=np.float64)
        for weights, count in zip(weight_sets, sample_counts):
            layer_avg += weights[layer_idx].astype(np.float64) * (count / total_samples)
        averaged.append(layer_avg.astype(weight_sets[0][layer_idx].dtype))
    return averaged

client_models = [init_model() for _ in range(num_clients)]

for r in range(num_round):
    print(f"\n*****Round {r + 1}/{num_round}*****")

    for client_id in range(num_clients):
        client_models[client_id] = train_local_model(
            client_models[client_id],
            X_train_peer[client_id],
            y_train_peer[client_id],
        )

    encrypted_messages = {}
    for sender_id in range(num_clients):
        sender_payload = {
            "round": r,
            "sender_id": sender_id,
            "sample_count": len(X_train_peer[sender_id]),
            "weights": client_models[sender_id].get_weights(),
        }
        for receiver_id in get_peer_ids(sender_id, num_clients):
            encrypted_messages[(sender_id, receiver_id)] = encrypt_model_payload(
                sender_payload, AES_KEY
            )

    round_accuracies = []
    for receiver_id in range(num_clients):
        own_weights = client_models[receiver_id].get_weights()
        aggregation_weights = [own_weights]
        aggregation_counts = [len(X_train_peer[receiver_id])]

        for sender_id in get_peer_ids(receiver_id, num_clients):
            encrypted = encrypted_messages[(sender_id, receiver_id)]
            received = decrypt_model_payload(encrypted, AES_KEY)

            if received["round"] != r or received["sender_id"] != sender_id:
                raise ValueError("Received peer payload metadata is inconsistent.")
            peer_weights = received["weights"]
            if len(peer_weights) != len(own_weights) or any(
                a.shape != b.shape for a, b in zip(peer_weights, own_weights)
            ):
                raise ValueError(f"Client {sender_id} sent incompatible model weights.")

            aggregation_weights.append(peer_weights)
            aggregation_counts.append(int(received["sample_count"]))

        aggregated_weights = weighted_federated_average(
            aggregation_weights, aggregation_counts
        )
        client_models[receiver_id].set_weights(aggregated_weights)

        y_pred = np.argmax(
            client_models[receiver_id].predict(X_test_peer[receiver_id], verbose=0),
            axis=1,
        )
        y_test = y_test_peer[receiver_id]
        accuracy = accuracy_score(y_test, y_pred)
        round_accuracies.append(accuracy)

        print(f"\nClient {receiver_id} accuracy: {accuracy:.4f}")
        print(classification_report(
            y_test, y_pred, labels=np.arange(num_classes),
            target_names=[str(c) for c in label_encoder.classes_],
            zero_division=0
        ))
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    print(f"\nMean personalized client accuracy: {np.mean(round_accuracies):.4f}")

t2 = time.time()
print("\nElapsed time (seconds):", t2 - t1)
