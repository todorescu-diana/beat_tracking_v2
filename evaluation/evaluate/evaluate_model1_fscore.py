import sys
sys.path.append('')
from constants.constants import VALID_DATASET_NAMES
from utils.dataset_utils import get_load_dataset_params, load_dataset
from classes.sequences.spectrogram_sequence import SpectrogramSequence
from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory
from utils.model_utils import predict
import mir_eval
import mirdata
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory
from keras.models import load_model

dataset_name = VALID_DATASET_NAMES[9]

replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm = get_load_dataset_params(dataset_name)
dataset_tracks = load_dataset(dataset_name, replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm)

first_track = next(iter(dataset_tracks.items()))
# print("---------------- ", first_track)  # output: (k, v)
k, v = first_track

model = load_model('models/saved/trained_gtzan_rhythm_v2_mel_best.h5')

mel_preprocessor = SpectrogramProcessorFactory.create_spectrogram_processor('mel')
pre_processor = mel_preprocessor

spectrogram_sequence = SpectrogramSequence(
    tracks = {k: v},
    pre_processor=pre_processor,
    pad_frames=2
)

act, det = predict(model, spectrogram_sequence)

# f_score = mir_eval.beat.f_score(v.beats.times, det[k]['beats'])
# print("mir_eval f_score: ", f_score)  # over all metrical variations

f_score_helper = EvaluationHelperFactory.create_evaluation_helper('f_measure')
# f_score = f_score_helper.calculate_metric(det[k]['beats'], v.beats.times)
# print("my f_score: ", f_score)    # only original metrical level

# print("custom mean f_score: ", f_score_helper.calculate_mean_metric(det, {k: {'beats': v.beats.times}}))

f_score_helper.plot_beats(det[k]['beats'], v.beats.times, n=5)
