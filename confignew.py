class Config:
    def __init__(self):
        # Dataset / IO
        self.DATA_ROOT = "CASME2/raw"
        self.ANNOTATION_FILE = "CASME2/CASME2.csv"
        self.LIMIT = None
        self.NUM_FRAMES = 30

        # Training
        self.BATCH_SIZE = 8
        self.LR = 1e-3
        self.EPOCHS = 50
        self.TRAIN_SPLIT = 0.7
        self.VAL_SPLIT = 0.15
        self.TEST_SPLIT = 0.15

        # Model
        self.NUM_CLASSES = 7
        self.EMBED_DIM = 512
        self.WINDOW_SIZE = 7
        self.BKBONE_LR = 1e-5
        self.TEMP_LR = 1e-3

        # Experiment
        self.EXP_NAME = "swin_mamba_experiment"


cfg = Config()