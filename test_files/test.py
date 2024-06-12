import sys

from classes.sequences.spectrogram_sequence import SpectrogramSequence
from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory
sys.path.append('')
from classes.audio_track import AudioTrack
from utils.utils import play_audio_with_clicktrack
from utils.model_utils import predict
from keras.models import load_model
from constants.constants import VALID_DATASET_NAMES
from utils.dataset_utils import get_load_dataset_params, load_dataset
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory

model = load_model('')

dataset_name = VALID_DATASET_NAMES[2]

spectrogram_processor_factory = SpectrogramProcessorFactory()
mel_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('mel')
cqt_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('cqt')
pre_processor = mel_preprocessor

replace_dots_with_underline, tiny_aam, harmonix_set = get_load_dataset_params(dataset_name)
dataset_tracks = load_dataset(dataset_name, replace_dots_with_underline, tiny_aam, harmonix_set)

first_item = next(iter(dataset_tracks.items()))
print(first_item)  # Output: ('a', 1)
first_k, first_v = first_item

spectrogram_sequence = SpectrogramSequence(
    data_sequence_tracks={first_k: first_v},
    data_sequence_pre_processor=pre_processor,
    data_sequence_pad_frames=2
)

# # predict for metrics
# activations, detections = predict(model, spectrogram_sequence)
# print(first_k, first_v)
# # print(spectrogram_sequence[0][0].shape)
# # print(activations[first_k])
# # print(detections[first_k])
# # print(activations[first_k]['beats'].shape)
# # print(detections[first_k]['beats'].shape)
# # play_audio_with_clicktrack(first_v, detections[first_k]['beats'])

# cemgil_helper = EvaluationHelperFactory.create_evaluation_helper('cemgil')
# continuity_helper = EvaluationHelperFactory.create_evaluation_helper('continuity')
# f_measure_helper = EvaluationHelperFactory.create_evaluation_helper('f_measure')
# # predict for metrics
# _, detections = predict(model, spectrogram_sequence)
# beat_detections = detections
# beat_annotations = {k: {'beats': v.beats.times} for k, v in dataset_tracks.items() if v.beats is not None}

# # print("DETECTIONS: ", beat_detections)
# # print("ANNOTATIONS: ", beat_annotations)
# cemgil_dataset_mean = cemgil_helper.calculate_metric(beat_detections['blues.00000']['beats'], beat_annotations['blues.00000']['beats'])
# continuity_dataset_mean = continuity_helper.calculate_metric(beat_detections['blues.00000']['beats'], beat_annotations['blues.00000']['beats'])
# f_measure_dataset_mean = f_measure_helper.calculate_metric(beat_detections['blues.00000']['beats'], beat_annotations['blues.00000']['beats'])

# print(cemgil_dataset_mean)
# print(continuity_dataset_mean)
# print(f_measure_dataset_mean)

# cemgil_dataset_mean_1 = cemgil_helper.calculate_mean_metric(beat_detections, beat_annotations)
# continuity_dataset_mean_1 = continuity_helper.calculate_mean_metric(beat_detections, beat_annotations)
# f_measure_dataset_mean_1 = f_measure_helper.calculate_mean_metric(beat_detections, beat_annotations)

# print(cemgil_dataset_mean_1)
# print(continuity_dataset_mean_1)
# print(f_measure_dataset_mean_1)

# cemgil_metric_components = cemgil_helper.metric_components()
# continuity_metric_components = continuity_helper.metric_components()
# f_measure_metric_components = f_measure_helper.metric_components()

# print(cemgil_metric_components)
# print(continuity_metric_components)
# print(f_measure_metric_components)

# # # calculate remaining / extra metrics
# # cemgil_dataset_mean = cemgil_helper.calculate_mean_metric(beat_detections, beat_annotations)
# # continuity_dataset_mean = continuity_helper.calculate_mean_metric(beat_detections, beat_annotations)
# # f_measure_dataset_mean = f_measure_helper.calculate_mean_metric(beat_detections, beat_annotations)

# # print("ok pana acum.")
# # import sys
# # sys.path.append('')
# # from constants.constants import RESULTS_DIR_PATH


# # with open(RESULTS_DIR_PATH + "/" + 'empty_dir_placeholder_file' + ".txt", "w") as file:
# #     file.write("test ...")