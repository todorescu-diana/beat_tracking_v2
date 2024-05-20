import tensorflow as tf
from sklearn.model_selection import train_test_split
from constants.constants import PAD_FRAMES
from classes.spectrogram_processor import SpectrogramProcessor
from classes.data_sequence import DataSequence
from utils.dataset_utils import load_dataset
from utils.model_utils import build_model, compile_model, predict, train_model
from utils.utils import play_audio_with_clicktrack

# BALLROOM
ballroom_tracks = load_dataset('ballroom')
ballroom_track = ballroom_tracks['Albums-Cafe_Paradiso-05']
print("BALLROOM TRACK: ", ballroom_track.audio_path)

# SMC
smc_tracks = load_dataset('smc')
smc_track = smc_tracks['SMC_001']
print("SMC TRACK: ", smc_track.audio_path)

# GTZAN
gtzan_tracks = load_dataset('gtzan')
gtzan_track = gtzan_tracks['blues.00001']
print("GTZAN TRACK: ", gtzan_track.audio_path)

# DAGSTUHL CHOIR
dagstuhl_choir_tracks = load_dataset('dagstuhl_choir', strict=True)
dagstuhl_choir_tracks_filtered = {name: track for name, track in dagstuhl_choir_tracks.items() if track.beats.times is not None}

dagstuhl_choir_track = dagstuhl_choir_tracks_filtered['DCS_LI_QuartetB_Take04_Stereo_STM']
print("DAGSTUHL CHOIR TRACK: ", dagstuhl_choir_track.audio_path)

# beatboxset
beatboxset_tracks = load_dataset('beatboxset', strict=True)

beatboxset_track = beatboxset_tracks['battleclip_daq']
print("BEATBOXSET TRACK: ", beatboxset_track.audio_path)

# guitarset_mic
guitarset_mic_tracks = load_dataset('guitarset_mic')

guitarset_mic_track = guitarset_mic_tracks['00_BN1-129-Eb_comp_mic']
print("GUITARSET_MIC TRACK: ", guitarset_mic_track.audio_path)

# guitarset_pickup
guitarset_pickup_tracks = load_dataset('guitarset_pickup')

guitarset_pickup_track = guitarset_pickup_tracks['00_BN1-129-Eb_comp_mix']
print("GUITARSET_PICKUP TRACK: ", guitarset_pickup_track.audio_path)

# tiny_aam
tiny_aam_tracks = load_dataset('tiny_aam', tiny_aam=True)

tiny_aam_track = tiny_aam_tracks['0001_mix']
print("TINY_AAM TRACK: ", tiny_aam_track.audio_path)

tracks = tiny_aam_tracks

train_files, test_files = train_test_split(list(tracks.keys()), test_size=0.2, random_state=1234)
pre_processor = SpectrogramProcessor()
dagstuhl_choir_spectrogram = pre_processor.process(dagstuhl_choir_track.audio_path)
pre_processor.plot_spectrogram(dagstuhl_choir_spectrogram)
play_audio_with_clicktrack(dagstuhl_choir_track, dagstuhl_choir_track.beats.times)

train = DataSequence(
        data_sequence_tracks={k: v for k, v in tracks.items() if k in train_files},
        data_sequence_pre_processor=pre_processor,
        data_sequence_pad_frames=PAD_FRAMES
    )
train.widen_beat_targets()

test = DataSequence(
    data_sequence_tracks={k: v for k, v in tracks.items() if k in test_files},
    data_sequence_pre_processor=pre_processor, data_sequence_pad_frames=PAD_FRAMES
)
test.widen_beat_targets()

model = build_model()
compile_model(model)
train_model(model, train_data=train, test_data=test, model_name='test_model_ballroom')
model = tf.keras.models.load_model('../models/saved/test_model_ballroom_best.h5')
activations, detections = predict(model, test)

first_k, first_v = next(iter(tracks))
play_audio_with_clicktrack(first_v, first_v.beats.times)

print(detections)
