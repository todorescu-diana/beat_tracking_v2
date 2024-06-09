import numpy as np

FPS = 100

HOP_SIZE = 441
FIXED_SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = int(0.01 * FIXED_SAMPLE_RATE)
WINDOW = 'hann'
N_MELS = 81

F_MIN = 30
F_MAX = 17000

FRAME_DURATION = 1 / FPS

PAD_FRAMES = 2

NUM_FRAMES = 5

INPUT_SHAPE = (None, None, 81, 1)

# conv layers
ACTIVATION = 'elu'
DROPOUT_RATE = 0.15
CONV_NUM_FILTERS = 20
TCN_KERNEL_SIZE = 5
TCN_DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# optimizer
LEARNRATE_1 = 0.002
CLIPNORM_1 = 0.5

NUM_FILTERS_2 = 16
DROPOUT_RATE_2 = 0.1
NUM_FILTERS_TCN_2 = 16
NUM_DILATIONS_TCN_2 = 11
DROPOUT_RATE_TCN_2 = 0.1

# k-fold-cross-validation
NUM_FOLDS = 8

# train
NUM_EPOCHS = 100

VALID_DATASET_NAMES = ['ballroom', 'smc', 'gtzan', 'dagstuhl_choir', 'beatboxset', 'guitarset_mic', 'guitarset_pickup', 'tiny_aam', 'harmonix_set']

DATASET_PATHS = {
    VALID_DATASET_NAMES[0]: {
        "audio_dir": '',
        "annot_dir": ''
    },
    VALID_DATASET_NAMES[1]: {
        "audio_dir": '',
        "annot_dir": ''
    },
    VALID_DATASET_NAMES[2]: {
        "audio_dir": '',
        "annot_dir": ''
    },
    VALID_DATASET_NAMES[3]: {
        "audio_dir": '',
        "annot_dir": ''
    },
    VALID_DATASET_NAMES[4]: {
        "audio_dir": '',
        "annot_dir": ''
    },
    VALID_DATASET_NAMES[5]: {
        "audio_dir": '',
        "annot_dir": ''
    },
    VALID_DATASET_NAMES[6]: {
        "audio_dir": '',
        "annot_dir": ''
    },
    VALID_DATASET_NAMES[7]: {
        "audio_dir": '',
        "annot_dir": ''
    },
    VALID_DATASET_NAMES[8]: {
        "audio_dir": '',
        "annot_dir": ''
    }
}

SUMMARY_SAVE_PATH = ''
MODEL_SAVE_PATH = ''
PLOT_SAVE_PATH = ''
RESULTS_DIR_PATH = ''
CSV_LOSSES_PATH = ''


plot_colors = {
    'train': 'darkslategray',
    'val': 'firebrick'
}