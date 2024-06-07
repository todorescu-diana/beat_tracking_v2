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
                # ---
                # # Define the start and end times in seconds
                # start_time = 60  # 1 minute in seconds
                # end_time = 90    # 1 minute 30 seconds in seconds

                # # Filter the beat times to retain only those between 1:00 and 1:30
                # filtered_beat_times = [bt for bt in t.beats.times if start_time <= bt <= end_time]
                # data_sequence_beats = filtered_beat_times
                # ---
                # use track only if it contains beats
                data_sequence_beats = t.beats.times
                # compute features first to be able to quantize beats
                data_sequence_spectrogram = data_sequence_pre_processor.process(t.audio_path)
                # # Min-Max Normalization to range [0, 1]
                # min_val = np.min(data_sequence_spectrogram)
                # max_val = np.max(data_sequence_spectrogram)
                # normalized_spectrogram = (data_sequence_spectrogram - min_val) / (max_val - min_val)
                # data_sequence_spectrogram = normalized_spectrogram
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

class SlicedDataSequence(Sequence):
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
        total_slices = sum(max(0, (len(self.spectrogram[key]) - 1000) // 999 + 1) for key in self.ids)
        return (total_slices + 32 - 1) // 32  # ceiling division

    def __getitem__(self, idx):
        batch_start = idx * 32
        batch_x = []
        batch_y = []

        cumulative_idx = 0

        for key in self.ids:
            num_slices = max(0, (len(self.spectrogram[key]) - 1000) // 999 + 1)
            for slice_idx in range(num_slices):
                if len(batch_x) >= 32:
                    break  # batch is full

                if cumulative_idx >= batch_start:
                    slice_start = slice_idx * 999
                    slice_end = slice_start + 1000
                    data_sequence_spectrogram = self.spectrogram[key][slice_start:slice_end]
                    beat_data = self.beats[key][slice_start:slice_end]

                    # # Normalize spectrogram
                    # max_abs = np.max(np.abs(data_sequence_spectrogram))
                    # data_sequence_spectrogram = (data_sequence_spectrogram + max_abs) / (2 * max_abs)

                    if self.pad_frames:
                        data_sequence_spectrogram = cnn_pad(data_sequence_spectrogram, self.pad_frames)

                    batch_x.append(tf.convert_to_tensor(data_sequence_spectrogram[..., np.newaxis]))
                    batch_y.append(tf.convert_to_tensor(beat_data[..., np.newaxis]))


                cumulative_idx += 1

        batch_x = np.stack(batch_x)
        batch_y = np.stack(batch_y)

        return tf.convert_to_tensor(batch_x), {'beats': tf.convert_to_tensor(batch_y)}

    def widen_beat_targets(self, size=3, value=0.5):
        for y in self.beats.values():
            np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)

    def append(self, other):
        assert not any(key in self.ids for key in other.ids), 'IDs must be unique'
        self.spectrogram.update(other.spectrogram)
        self.beats.update(other.beats)
        self.ids.extend(other.ids)