import mir_eval
import mirdata
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory
from classes.spectrogram_processor import SpectrogramProcessor

gtzan = mirdata.initialize('gtzan_genre', version='mini')
tracks = gtzan.load_tracks()
track = list(tracks.items())[0]
k, v = track
print("TRACK: ", track)
pre_processor = SpectrogramProcessor()

cemgil = mir_eval.beat.cemgil(v.beats.times, v.beats.times)
print("mir_eval cemgil: ", cemgil)  # over all metrical variations

cemgil_helper = EvaluationHelperFactory.create_evaluation_helper('cemgil')
cemgil = cemgil_helper.calculate_metric(v.beats.times, v.beats.times)
print("my cemgil: ", cemgil)    # only original metrical level

print("custom mean cemgil: ", cemgil_helper.calculate_mean_metric({'track123': v.beats.times}, {'track123': v.beats.times}))

cemgil_helper.plot_beats(v.beats.times, v.beats.times, n=4)
