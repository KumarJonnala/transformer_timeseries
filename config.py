class Config:

    # Training configuration
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 25
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Model configuration
    NUM_FEATURES = 6
    NUM_CLASSES = 2
    EMBEDDING_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 4
    DROPOUT = 0.1
    
    # Optimizer configuration
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    
    # Data configuration
    PICKLE_PATH = '/Users/kumar/Desktop/Projects/transformer_timeseries/wesad_raw.pkl'

    # Subject configuration
    SUBJECT_COUNTS = {
        'S2': 440, 'S3': 445, 'S4': 449, 'S5': 460, 'S6': 458, 'S7': 457,
        'S8': 460, 'S9': 456, 'S10': 476, 'S11': 465, 'S13': 461, 'S14': 464,
        'S15': 464, 'S16': 463, 'S17': 476
    }