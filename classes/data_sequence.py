from keras.utils import Sequence
import sys
import numpy as np
from scipy.ndimage import maximum_filter1d
import tensorflow as tf
from constants.constants import FPS
from utils.utils import beats_to_frame_indices, one_hot_encode_beats
from utils.model_utils import cnn_pad

# wrap training/test data as a keras sequence to use with fit()
class DataSequence(Sequence):
    def __init__(self, data_sequence_tracks, data_sequence_pre_processor, pad_frames=None, fps=FPS):
        # store features and targets in dictionaries with name of the song as key
        self.spectrogram = {}
        self.beats = {}
        self.ids = []
        self.pad_frames = pad_frames
        self.fps = fps
        # iterate over all tracks
        for data_sequence_i, data_sequence_key in enumerate(data_sequence_tracks):
            # print progress
            sys.stderr.write(
                f'\rprocessing track {data_sequence_i + 1}/{len(data_sequence_tracks)}: {data_sequence_key + " " * 20}')
            sys.stderr.flush()
            t = data_sequence_tracks[data_sequence_key]
            if t.beats.times is not None:
                # use track only if it contains beats
                data_sequence_beats = t.beats.times
                # compute features first to be able to quantize beats
                data_sequence_spectrogram = data_sequence_pre_processor.process(t.audio_path)
                if len(data_sequence_spectrogram):
                  self.spectrogram[data_sequence_key] = data_sequence_spectrogram

                  beat_positions_frames = beats_to_frame_indices(data_sequence_beats, self.fps)
                  quantized_beat_frames = one_hot_encode_beats(beat_positions_frames, data_sequence_spectrogram.shape[0])
                  self.beats[data_sequence_key] = quantized_beat_frames
                else:
                  continue
            else:
                # no beats found, skip this file
                print(f'\r{data_sequence_key} has no beat information, skipping\n')
                continue

            # keep track of IDs
            self.ids.append(data_sequence_key)
        assert len(self.spectrogram) == len(self.beats) == len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # convert int idx to key
        if isinstance(idx, int):
            idx = self.ids[idx]

        # get the full spectrogram and beat information
        data_sequence_spectrogram = self.spectrogram[idx]
        if self.pad_frames:
            data_sequence_spectrogram = cnn_pad(data_sequence_spectrogram, self.pad_frames)
        beat_data = self.beats[idx]

        return tf.convert_to_tensor(data_sequence_spectrogram[np.newaxis, ..., np.newaxis]), {'beats': tf.convert_to_tensor(beat_data[np.newaxis, ..., np.newaxis])}

    def widen_beat_targets(self, size=3, value=0.5):
        for y in self.beats.values():
            np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)

    def append(self, other):
        assert not any(key in self.ids for key in other.ids), 'IDs must be unique'
        self.spectrogram.update(other.spectrogram)
        self.beats.update(other.beats)
        self.ids.extend(other.ids)