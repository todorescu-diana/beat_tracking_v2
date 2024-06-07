import sys
sys.path.append('')
from constants.constants import VALID_DATASET_NAMES
from utils.dataset_utils import get_load_dataset_params, load_dataset
import logging
from classes.audio_track import AudioTrack
from classes.spectrogram_sequence import SpectrogramSequence
from utils.model_utils import predict
from utils.utils import play_audio_with_clicktrack
import mirdata
from classes.spectrogram_processor import SpectrogramProcessor
from keras.models import load_model
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

# dataset = mirdata.initialize('gtzan_genre', version='mini')
# tracks = dataset.load_tracks()
# dataset_name = VALID_DATASET_NAMES[5]
# replace_dots_with_underline, tiny_aam, harmonix_set = get_load_dataset_params(dataset_name)
    
# pre_processor = SpectrogramProcessor()
# tracks = load_dataset(dataset_name, replace_dots_with_underline, tiny_aam, harmonix_set)
# first_track = next(iter(tracks.items()))
# first_k, first_v = first_track
# # chericherilady = AudioTrack(audio_path=first_v.audio_path)
pre_processor = SpectrogramProcessor()
# spectrogram_sequence = SpectrogramSequence(
#     data_sequence_tracks = tracks,
#     data_sequence_pre_processor=pre_processor,
#     data_sequence_pad_frames=2
# )

spectrogram = pre_processor.process('')
# pre_processor.plot_spectrogram(spectrogram, duration=5)

print(spectrogram[:100])

spectrogram = pre_processor.process('')
# pre_processor.plot_spectrogram(spectrogram, duration=5)
print(spectrogram[:100])
# print(spectrogram.shape)
# pre_processor.plot_spectrogram(spectrogram)

# print("Spectrogram Shape: ", spectrogram_sequence[0][0].shape)

# model = load_model('models/saved/train_gtzan_v2_best')

# act, det = predict(model, spectrogram_sequence)

# play_audio_with_clicktrack(first_track, det[first_k]['beats'])
# play_audio_with_clicktrack(first_track, first_v.beats.times)

