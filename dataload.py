import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from peer_config import DATA_FILES, TEST_SIZE, RANDOM_STATE


def load_all_labels():

    labels = set()

    for peer_id, file_path in DATA_FILES.items():

        df = pd.read_csv(file_path)

        target = df.iloc[:, -1]

        labels.update(target.astype(str).unique())

    labels = sorted(labels)

    encoder = LabelEncoder()
    encoder.fit(labels)

    return encoder


def load_peer_data(peer_id, global_encoder=None):

    file_path = DATA_FILES[peer_id]

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"Dataset for Peer {peer_id} is empty.")

    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].astype(str).copy()

    for column in X.columns:

        if X[column].dtype == "object":

            X[column] = LabelEncoder().fit_transform(
                X[column].astype(str)
            )

    X = X.apply(pd.to_numeric, errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    X = X.fillna(X.median())

    X = X.to_numpy(dtype=np.float32)

    if global_encoder is None:
        global_encoder = load_all_labels()

    y = global_encoder.transform(y)

    unique_classes, class_counts = np.unique(y, return_counts=True)

    stratify_labels = (
        y if np.min(class_counts) >= 2 else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(
        X_train.shape[0], 1, X_train.shape[1]
    )

    X_test = X_test.reshape(
        X_test.shape[0], 1, X_test.shape[1]
    )

    return {
        "X_train": X_train.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_train": np.asarray(y_train, dtype=np.int32),
        "y_test": np.asarray(y_test, dtype=np.int32),
        "scaler": scaler,
        "encoder": global_encoder,
        "feature_names": list(df.columns[:-1]),
        "input_features": X_train.shape[-1],
        "num_classes": len(global_encoder.classes_)
    }
