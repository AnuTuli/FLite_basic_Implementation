import numpy as np
import tensorflow as tf

from peer_config import (
    DATA_FILES,
    MODEL_DIR,
    WEIGHTS_DIR,
    METRICS_DIR,
    NUM_ROUNDS,
    LOCAL_EPOCHS,
    BATCH_SIZE,
    NEIGHBORS
)

from dataload import load_all_labels, load_peer_data

from localtrain import (
    train_local_model,
    evaluate_local_model
)

from aggregation import aggregate_local_and_neighbors

from metricslog import log_metrics


PEER_ID = 1


def save_weights(model, peer_id, round_number):

    file_path = WEIGHTS_DIR / (
        f"peer_{peer_id}_round_{round_number}.npz"
    )

    weights = model.get_weights()

    np.savez(
        file_path,
        **{
            f"weight_{i}": w
            for i, w in enumerate(weights)
        }
    )

    return file_path


def load_weights(file_path):

    data = np.load(file_path)

    keys = sorted(
        data.files,
        key=lambda k: int(k.split("_")[-1])
    )

    return [data[k] for k in keys]


def run_peer1():

    print("=" * 60)
    print("PEER 1 - LOCAL FEDERATED LEARNING")
    print("=" * 60)

    print("Peer ID:", PEER_ID)
    print("Neighbors:", NEIGHBORS[PEER_ID])
    print("Dataset:", DATA_FILES[PEER_ID])

    global_encoder = load_all_labels()

    data = load_peer_data(
        PEER_ID,
        global_encoder=global_encoder
    )

    print("Training samples:", len(data["X_train"]))
    print("Testing samples:", len(data["X_test"]))
    print("Input features:", data["input_features"])
    print("Classes:", data["num_classes"])

    model = None

    for round_number in range(1, NUM_ROUNDS + 1):

        print(f"\n--- Peer 1 | Round {round_number} ---")

        model, history = train_local_model(
            data,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE
        )

        metrics = evaluate_local_model(model, data)

        print("Accuracy:", metrics["accuracy"])
        print("Precision:", metrics["precision_weighted"])
        print("Recall:", metrics["recall_weighted"])
        print("F1:", metrics["f1_weighted"])

        log_metrics(
            str(METRICS_DIR / "peer1_metrics.csv"),
            PEER_ID,
            round_number,
            metrics
        )

        model_path = MODEL_DIR / (
            f"peer1_round_{round_number}.keras"
        )

        model.save(model_path)

        save_weights(
            model,
            PEER_ID,
            round_number
        )

    print("\nPeer 1 local training completed.")


if __name__ == "__main__":
    run_peer1()
