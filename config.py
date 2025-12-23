class Config:

    # Read data
    # DATASET_PATH = '~/Library/Mobile Documents/com~apple~CloudDocs/Phoenix/OVGU/HiWi2/Tasks/10_WESAD/WESAD.nosync'
    # PICKLE_PATH = '/Users/kumar/Desktop/Projects/transformer_timeseries/wesad_raw.pkl'
    PICKLE_PATH = '/home/bumu60du/transformers_ovgu/wesad_raw.pkl'

    # Subject configuration, num of sliding windows per subject
    SUBJECT_COUNTS = {
        'S2': 440, 'S3': 445, 'S4': 449, 'S5': 460, 'S6': 458, 'S7': 457,
        'S8': 460, 'S9': 456, 'S10': 476, 'S11': 465, 'S13': 461, 'S14': 464,
        'S15': 464, 'S16': 463, 'S17': 476
    }
    
    # Training configuration
    # Window size = 128, How many timesteps or consecutive records each sample contains
    # window size is downsampled from 700 Hz to 32Hz, 128/32 = 4 seconds of data per window
    RANDOM_SEED = 42
    N_TRIALS = 40
    TIMEOUT = 7200 #2 hours
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    BATCH_SIZE = 16        # batch size = How many independent samples are processed in parallel
    EARLY_STOPPING_PATIENCE = 8
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Model configuration
    NUM_FEATURES = 6       # 6 features total: ['ACC','Resp','EDA','Temp','ECG','EMG'], 5 for LOMO studies
    NUM_CLASSES = 2        # 2 labels, 0 for Baseline and 1 for Stress label for 4 seconds window
    EMBEDDING_DIM = 64
    NUM_HEADS = 4          # num_heads = width of attention (how many perspectives are considered in parallel).
    NUM_LAYERS = 4         # num_layers = depth of reasoning (how many times the model refines its understanding)
    DROPOUT = 0.1
    
    # Optimizer configuration
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3

    CHECKPOINT_DIR = "./checkpoints"
    
