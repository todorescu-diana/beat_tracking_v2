import mir_eval
import mirdata
from classes.spectrogram_processor import SpectrogramProcessor
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory

gtzan = mirdata.initialize('gtzan_genre', version='mini')
tracks = gtzan.load_tracks()
track = list(tracks.items())[0]
k, v = track
print("TRACK: ", track)
pre_processor = SpectrogramProcessor()
f_measure = mir_eval.beat.f_measure(v.beats.times, v.beats.times)
print("mir_eval f_measure: ", f_measure)

f_measure_helper = EvaluationHelperFactory.create_evaluation_helper('f_measure')
precision, recall, f_score = f_measure_helper.calculate_metric(v.beats.times, v.beats.times)
print("my f_measure: ", precision, recall, f_score)

print("custom mean f_score: ", f_measure_helper.calculate_mean_metric({'beats': v.beats.times}, {'beats': v.beats.times}))

f_measure_helper.plot_beats(v.beats.times, v.beats.times)
