import sys

import numpy as np
sys.path.append('')
from matplotlib import pyplot as plt
from classes.sequences.spectrogram_sequence import SpectrogramSequence
from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory
from classes.audio_track import AudioTrack
from utils.utils import play_audio_with_clicktrack
from utils.model_utils import predict
from keras.models import load_model
from constants.constants import FPS, VALID_DATASET_NAMES
from utils.dataset_utils import get_load_dataset_params, load_dataset
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory

dataset_name = VALID_DATASET_NAMES[9]

mel_preprocessor = SpectrogramProcessorFactory.create_spectrogram_processor('mel')
pre_processor = mel_preprocessor

replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm = get_load_dataset_params(dataset_name)
dataset_tracks = load_dataset(dataset_name, replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm)

first_track = next(iter(dataset_tracks.items()))
# print("---------------- ", first_track)  # output: (k, v)
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
print("ACTIVATIONS SHAPE: ", activations[first_k]['beats'].shape)

time = np.arange(len(activations[first_k]['beats'])) / FPS

# # plotting
# plt.figure(figsize=(10, 4))
# plt.plot(time, activations[first_k]['beats'], label='beat activation probability', color='lightseagreen')
# plt.xlabel('time [s]')
# plt.ylabel('probability')
# plt.title('beat activation function')
# plt.legend()
# plt.grid(True)
# plt.savefig('figure_plots/baf.png')
# plt.close()

# play_audio_with_clicktrack(first_v, detections[first_k]['beats'])