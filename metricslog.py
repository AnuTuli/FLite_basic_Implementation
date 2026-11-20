import csv
import os


def log_metrics(csv_file, peer_id, round_number, metrics):

    os.makedirs(
        os.path.dirname(csv_file) or ".",
        exist_ok=True
    )

    file_exists = os.path.isfile(csv_file)

    row = {
        "peer_id": peer_id,
        "round": round_number,
        "accuracy": metrics.get("accuracy"),
        "precision_weighted": metrics.get("precision_weighted"),
        "recall_weighted": metrics.get("recall_weighted"),
        "f1_weighted": metrics.get("f1_weighted")
    }

    with open(csv_file, "a", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys())
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
