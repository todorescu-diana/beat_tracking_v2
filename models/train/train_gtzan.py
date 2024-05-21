import sys
sys.path.append('')

from sklearn.model_selection import train_test_split
from classes.data_sequence import DataSequence
from classes.spectrogram_processor import SpectrogramProcessor
from constants.constants import MODEL_SAVE_PATH, PLOT_SAVE_PATH, SUMMARY_SAVE_PATH, VALID_DATASET_NAMES
from utils.dataset_utils import load_dataset
from utils.model_utils import build_model, compile_model, train_model
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

with tf.device('/GPU:0'):
    dataset_name = VALID_DATASET_NAMES[2]   # gtzan
    pre_processor = SpectrogramProcessor()
    dataset_tracks = load_dataset(dataset_name, gtzan=True)
    model_name = 'gtzan_v2_fixed_final'

    if dataset_tracks is not None:
        train_files, test_files = train_test_split(list(dataset_tracks.keys()), test_size=0.2, random_state=1234)
        train = DataSequence(
                data_sequence_tracks={k: v for k, v in dataset_tracks.items() if k in train_files},
                data_sequence_pre_processor=pre_processor,
                pad_frames=2
            )
        train.widen_beat_targets()
        test = DataSequence(
                data_sequence_tracks={k: v for k, v in dataset_tracks.items() if k in test_files},
                data_sequence_pre_processor=pre_processor,
                pad_frames=2
            )
        test.widen_beat_targets()

        model = build_model()
        compile_model(model, summary=True, model_name=model_name, summary_save_path=SUMMARY_SAVE_PATH)
        train_model( model, train_data=train, test_data=test, save_model=True, model_name=model_name, model_save_path=MODEL_SAVE_PATH, plot_save=True, plot_save_path=PLOT_SAVE_PATH)