import sys
sys.path.append('')
from constants.constants import RESULTS_DIR_PATH, VALID_DATASET_NAMES
from utils.dataset_utils import get_load_dataset_params, load_dataset
from utils.validate_model_utils import k_fold_cross_validation

dataset_name = VALID_DATASET_NAMES[10]

replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm = get_load_dataset_params(dataset_name)
dataset_tracks = load_dataset(dataset_name, replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm)

if dataset_tracks is not None:
    k_fold_cross_validation(dataset_tracks=dataset_tracks, dataset_name=dataset_name, results_dir_path=RESULTS_DIR_PATH)