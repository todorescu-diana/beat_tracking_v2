import sys
sys.path.append('')
from classes.sequences.spectrogram_sequence import SpectrogramSequence
from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory
from classes.audio_track import AudioTrack
from utils.utils import play_audio_with_clicktrack
from utils.model_utils import predict
from keras.models import load_model
from constants.constants import VALID_DATASET_NAMES
from utils.dataset_utils import get_load_dataset_params, load_dataset
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory

dataset_name = VALID_DATASET_NAMES[9]

spectrogram_processor_factory = SpectrogramProcessorFactory()
mel_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('mel')
pre_processor = mel_preprocessor

replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm = get_load_dataset_params(dataset_name)
dataset_tracks = load_dataset(dataset_name, replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm)

first_track = next(iter(dataset_tracks.items()))
print("---------------- ", first_track)  # output: ('a', 1)
first_k, first_v = first_track

spectrogram_sequence = SpectrogramSequence(
    tracks={first_k: first_v},
    pre_processor=pre_processor,
    pad_frames=2
)
print("SPECTROGRAM SHAPE: ", spectrogram_sequence[0].shape)

model = load_model('models/saved/trained_gtzan_rhythm_v2_mel_best.h5')

activations, detections = predict(model, spectrogram_sequence)
print("DETECTIONS SHAPE: ", detections[first_k]['beats'].shape)

play_audio_with_clicktrack(first_v, detections[first_k]['beats'])