from constants.constants import RESULTS_DIR_PATH, VALID_DATASET_NAMES
from classes.spectrogram_processor import SpectrogramProcessor
from utils.dataset_utils import get_load_dataset_params, load_dataset
from utils.validate_model_utils import k_fold_cross_validation
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

# datasets - separate
for dataset_name in VALID_DATASET_NAMES:
    pre_processor = SpectrogramProcessor()

    replace_dots_with_underline, tiny_aam = get_load_dataset_params(dataset_name)
    dataset_tracks = load_dataset(dataset_name, replace_dots_with_underline, tiny_aam)

    if dataset_tracks is not None:
        k_fold_cross_validation(dataset_tracks, dataset_name, results_dir_path=RESULTS_DIR_PATH)

# datasets - merged
datasets_merged = {}
for dataset_name in VALID_DATASET_NAMES:
    pre_processor = SpectrogramProcessor()

    replace_dots_with_underline, tiny_aam = get_load_dataset_params(dataset_name)
    dataset_tracks = load_dataset(dataset_name, replace_dots_with_underline, tiny_aam)
    datasets_merged.update(dataset_tracks)

if datasets_merged is not None:
    k_fold_cross_validation(datasets_merged, 'all_datasets_merged', results_dir_path=RESULTS_DIR_PATH)