import sys
sys.path.append('')
from classes.spectrogram_sequence import SpectrogramSequence
from utils.model_utils import predict
import mir_eval
import mirdata
from classes.spectrogram_processor import SpectrogramProcessor
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory
from keras.models import load_model

gtzan = mirdata.initialize('gtzan_genre', version='mini')
tracks = gtzan.load_tracks()
track = list(tracks.items())[0]
k, v = track

model = load_model('')

pre_processor = SpectrogramProcessor()
spectrogram_sequence = SpectrogramSequence(
    data_sequence_tracks = {k: v},
    data_sequence_pre_processor=pre_processor,pad_frames=2
)

act, det = predict(model, spectrogram_sequence)

f_measure = mir_eval.beat.f_measure(v.beats.times, det[k]['beats'])
print("mir_eval f_measure: ", f_measure)

f_measure_helper = EvaluationHelperFactory.create_evaluation_helper('f_measure')
precision, recall, f_score = f_measure_helper.calculate_metric(det[k]['beats'], v.beats.times)
print("my f_measure: ", precision, recall, f_score)

print("custom mean f_score: ", f_measure_helper.calculate_mean_metric(det, {k: {'beats': v.beats.times}}))

f_measure_helper.plot_beats(det[k]['beats'], v.beats.times)
