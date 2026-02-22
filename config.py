#------------------------------------
# Central config for ColabSwinMamba.
#-----------------------------------
#Edit values below. This file exposes a single `cfg` object imported
#by the training script (`from config import cfg`).

from types import SimpleNamespace

# Primary config values (dataset, model, training)
cfg = SimpleNamespace(
    # Dataset / IO
    DATA_ROOT="CASME2/raw",
    ANNOTATION_FILE="CASME2/CASME2.csv",
    LIMIT=None,                # set to int for quick debug runs (e.g. 2)
    NUM_FRAMES=30,

    # DataLoader / training
    BATCH_SIZE=8,
    LR=1e-3,
    EPOCHS=50,
    TRAIN_SPLIT=0.7,           # proportion of data for training
    VAL_SPLIT=0.15,            # proportion of data for validation
    TEST_SPLIT=0.15,           # proportion of data for testing

    # Model
    NUM_CLASSES=7,
    EMBED_DIM=512,
    WINDOW_SIZE=7,
    BKBONE_LR=1e-5,           # learning rate for backbone (if using pretrained)
    TEMP_LTR=1e-3,             # learning rate for temporal module

    # Experiment
    EXP_NAME="swin_mamba_experiment",
)


if __name__ == "__main__":
    # quick print for debugging
    print("Config:\n", cfg)