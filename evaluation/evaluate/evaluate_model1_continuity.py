import sys
sys.path.append('')
from classes.spectrogram_sequence import SpectrogramSequence
from utils.model_utils import predict
import mir_eval
import mirdata
from classes.mel_spectrogram_processor import MelSpectrogramProcessor
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory
from keras.models import load_model

gtzan = mirdata.initialize('gtzan_genre', version='mini')
tracks = gtzan.load_tracks()
track = list(tracks.items())[0]
k, v = track

model = load_model('')

pre_processor = MelSpectrogramProcessor()
spectrogram_sequence = SpectrogramSequence(
    data_sequence_tracks = {k: v},
    data_sequence_pre_processor=pre_processor,
    data_sequence_pad_frames=2
)

act, det = predict(model, spectrogram_sequence)

continuity = mir_eval.beat.continuity(v.beats.times, det[k]['beats'])
print("mir_eval continuity: ", continuity)  # over all metrical variations

continuity_helper = EvaluationHelperFactory.create_evaluation_helper('continuity')
continuity = continuity_helper.calculate_metric(det[k]['beats'], v.beats.times)
print("my continuity: ", continuity)    # only original metrical level

print("custom mean continuity: ", continuity_helper.calculate_mean_metric(det, {k: {'beats': v.beats.times}}))

continuity_helper.plot_beats(det[k]['beats'], v.beats.times)
