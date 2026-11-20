from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PEER_IDS = [1, 2, 3, 4, 5]

NEIGHBORS = {
    1: [2, 5],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 1],
}

DATA_FILES = {
    1: BASE_DIR / "Peer_1" / "data1.csv",
    2: BASE_DIR / "Peer_2" / "data2.csv",
    3: BASE_DIR / "Peer_3" / "data3.csv",
    4: BASE_DIR / "Peer_4" / "data4.csv",
    5: BASE_DIR / "Peer_5" / "data5.csv",
}

NUM_ROUNDS = 5
LOCAL_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

TEST_SIZE = 0.20
RANDOM_STATE = 42

D_MODEL = 64
NUM_HEADS = 4
FF_DIM = 128
DROPOUT = 0.3

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
METRICS_DIR = OUTPUT_DIR / "metrics"

for directory in [OUTPUT_DIR, MODEL_DIR, WEIGHTS_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
