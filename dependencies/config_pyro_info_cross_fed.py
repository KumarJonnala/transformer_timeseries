class Config:

    # ======================== Dataset Configuration ========================
    # DATASET_PATH = '~/Library/Mobile Documents/com~apple~CloudDocs/Phoenix/OVGU/HiWi2/Tasks/10_WESAD/WESAD.nosync'
    # PICKLE_PATH = '/Users/kumar/Desktop/Projects/transformer_timeseries/wesad_raw.pkl'
    PICKLE_PATH = '/home/bumu60du/transformers_ovgu/wesad_raw.pkl'

    # Subject configuration, num of sliding windows per subject
    SUBJECT_COUNTS = {
        'S2': 440, 'S3': 445, 'S4': 449, 'S5': 460, 'S6': 458, 'S7': 457,
        'S8': 460, 'S9': 456, 'S10': 476, 'S11': 465, 'S13': 461, 'S14': 464,
        'S15': 464, 'S16': 463, 'S17': 476
    }

    # ======================== Data Preprocessing ========================
    # Window size = 128 timesteps
    # Downsampled from 700 Hz to 32Hz = 128/32 = 4 seconds of data per window
    SEQ_LEN = 128                      # Sequence length (sliding window size)
    WINDOW_SIZE = 128
    WINDOW_OVERLAP = 0.0
    
    # ======================== Training Configuration ========================
    RANDOM_SEED = 42
    N_TRIALS = 40
    TIMEOUT = 7200                    # 2 hours
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    BATCH_SIZE = 16                   # How many independent samples processed in parallel
    EARLY_STOPPING_PATIENCE = 8
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # ======================== Model Data Configuration ========================
    # WESAD has 6 physiological signals
    ENC_IN = 6                        # Encoder input: ['ACC','Resp','EDA','Temp','ECG','EMG']
    DEC_IN = 6                        # Decoder input dimensions
    C_OUT = 6                         # Channel output dimensions
    
    NUM_FEATURES = 6                  # Total features for compatibility
    NUM_CLASSES = 2                   # Binary classification: 0=Baseline, 1=Stress
    
    # ======================== Task Configuration ========================
    TASK_NAME = 'classification'      # Options: 'long_term_forecast', 'short_term_forecast', 
                                      # 'imputation', 'anomaly_detection', 'classification'
    PRED_LEN = 1                      # Prediction length (not used for classification, set to 1)
    LABEL_LEN = 0                     # Label length (not used for classification)
    
    # ======================== Transformer Model Architecture ========================
    D_MODEL = 64                      # Dimension of transformer hidden states
    N_HEADS = 4                       # Number of attention heads
    NUM_HEADS = 4                     # Alias for compatibility
    
    E_LAYERS = 2                      # Number of encoder layers
    D_LAYERS = 1                      # Number of decoder layers
    NUM_LAYERS = 4                    # Alias for compatibility with other models
    
    D_FF = 256                        # Dimension of feedforward networks (usually 4*d_model)
    DROPOUT = 0.1                     # Dropout rate
    ACTIVATION = 'gelu'               # Activation function: 'gelu' or 'relu'
    
    # FEDformer/Autoformer-specific configurations
    MOVING_AVG = 25                   # Kernel size for moving average decomposition
    
    # Informer-specific configurations
    DISTIL = True                     # Whether to use distillation (downsampling) in encoder
    FACTOR = 5                        # Sparsity factor for ProbAttention
    EMBED = 'timeF'                   # Embedding type: 'timeF', 'fixed', 'learned'
    FREQ = 'h'                        # Frequency for time embeddings
    
    # Embedding configuration
    EMBEDDING_DIM = 64                # Embedding dimension (alias for compatibility)
    
    # ======================== Optimizer Configuration ========================
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3

    # ======================== Checkpoint and Output ========================
    CHECKPOINT_DIR = "./checkpoints"
