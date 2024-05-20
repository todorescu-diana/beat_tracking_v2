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

dataset = mirdata.initialize('gtzan_genre', version='mini')
tracks = dataset.load_tracks()
first_track = next(iter(tracks.items()))
first_k, first_v = first_track
chericherilady = AudioTrack(audio_path='')
pre_processor = SpectrogramProcessor()
spectrogram_sequence = SpectrogramSequence(
    data_sequence_tracks = {'test_track': chericherilady},
    data_sequence_pre_processor=pre_processor,pad_frames=2
)

# pre_processor.plot_spectrogram(spectrogram, duration=5)

print("Spectrogram Shape: ", spectrogram_sequence[0][0].shape)

model = load_model('')

act, det = predict(model, spectrogram_sequence)
# print(det)
play_audio_with_clicktrack(chericherilady, det['test_track']['beats'])

