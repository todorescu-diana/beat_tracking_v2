from keras.utils import Sequence
import sys
import numpy as np
from scipy.ndimage import maximum_filter1d
import tensorflow as tf
from constants.constants import FPS
from utils.utils import beats_to_frame_indices, one_hot_encode_beats
from utils.model_utils import cnn_pad


class DataSequence(Sequence):
    def __init__(self, tracks, pre_processor, pad_frames=None, fps=FPS):
        self.ids = []
        self.spectrogram = {}
        self.beats = {}
        self.pad_frames = pad_frames
        self.fps = fps
        
        for data_sequence_i, data_sequence_key in enumerate(tracks):
            sys.stderr.write(
                f'\rprocessing track {data_sequence_i + 1}/{len(tracks)}: {data_sequence_key + " " * 20}')
            sys.stderr.flush()

            t = tracks[data_sequence_key]
            if t.beats.times is not None:
                data_sequence_beats = t.beats.times
                data_sequence_spectrogram = pre_processor.process(t.audio_path)
                if len(data_sequence_spectrogram):
                  self.spectrogram[data_sequence_key] = data_sequence_spectrogram

                  beat_positions_frames = beats_to_frame_indices(data_sequence_beats, self.fps)
                  quantized_beat_frames = one_hot_encode_beats(beat_positions_frames, data_sequence_spectrogram.shape[0])
                  self.beats[data_sequence_key] = quantized_beat_frames
                else:
                  continue
            else:
                print(f'\r{data_sequence_key} has no beat information, skipping\n')
                continue

            self.ids.append(data_sequence_key)
        assert len(self.spectrogram) == len(self.beats) == len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = self.ids[idx]

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