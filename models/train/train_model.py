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
    # datasets - separate
    for dataset_name in VALID_DATASET_NAMES:
        pre_processor = SpectrogramProcessor()
        model_name = dataset_name + '_v2'

        strict = False
        gtzan = False
        tiny_aam = False

        if dataset_name == 'dagstuhl_choir' or dataset_name == 'beatboxset':
            strict = True
        elif dataset_name == 'gtzan':
            gtzan = True
        elif dataset_name == 'tiny_aam':
            tiny_aam = True
        dataset_tracks = load_dataset(dataset_name, strict, gtzan, tiny_aam)

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
            train.widen_beat_targets()

            model = build_model()
            compile_model(model, summary=True, model_name=model_name, summary_save_path=SUMMARY_SAVE_PATH)
            train_model( model, train_data=train, test_data=test, save_model=True, model_name=model_name, model_save_path=MODEL_SAVE_PATH, plot_save=True, plot_save_path=PLOT_SAVE_PATH)

    # datasets - merged
    datasets_merged = {}
    for dataset_name in VALID_DATASET_NAMES:
        strict = False
        gtzan = False
        tiny_aam = False

        if dataset_name == 'dagstuhl_choir' or dataset_name == 'beatboxset':
            strict = True
        elif dataset_name == 'gtzan':
            gtzan = True
        elif dataset_name == 'tiny_aam':
            tiny_aam = True
        dataset_tracks = load_dataset(dataset_name, strict, gtzan, tiny_aam)
        datasets_merged.update(dataset_tracks)

    pre_processor = SpectrogramProcessor()
    model_name = 'all_datasets_merged_v2'

    if datasets_merged is not None:
        train = DataSequence(
            data_sequence_tracks={k: v for k, v in datasets_merged.items()},
            data_sequence_pre_processor=pre_processor
        )
        train.widen_beat_targets()

        model = build_model()
        compile_model(model, summary=True, model_name=model_name, summary_save_path=SUMMARY_SAVE_PATH)
        train_model( model, train_data=train, test_data=test, save_model=True, model_name=model_name, model_save_path=MODEL_SAVE_PATH, plot_save=True, plot_save_path=PLOT_SAVE_PATH)