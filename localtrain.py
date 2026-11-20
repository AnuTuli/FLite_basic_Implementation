import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from transformer import (
    build_slm_transformer,
    compile_model
)

from peer_config import (
    LOCAL_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    D_MODEL,
    NUM_HEADS,
    FF_DIM,
    DROPOUT
)


def train_local_model(data, epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE):

    model = build_slm_transformer(
        input_features=data["input_features"],
        num_classes=data["num_classes"],
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        dropout=DROPOUT
    )

    model = compile_model(
        model,
        learning_rate=LEARNING_RATE
    )

    history = model.fit(
        data["X_train"],
        data["y_train"],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.10,
        verbose=1,
        shuffle=True
    )

    return model, history


def evaluate_local_model(model, data):

    probabilities = model.predict(
        data["X_test"],
        verbose=0
    )

    predictions = np.argmax(probabilities, axis=1)

    y_true = data["y_test"]

    results = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision_weighted": precision_score(
            y_true, predictions,
            average="weighted",
            zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, predictions,
            average="weighted",
            zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, predictions,
            average="weighted",
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist()
    }

    return results
