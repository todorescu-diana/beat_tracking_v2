import mir_eval
import mirdata
from classes.spectrogram_processor import SpectrogramProcessor
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory

gtzan = mirdata.initialize('gtzan_genre', version='mini')
tracks = gtzan.load_tracks()
track = list(tracks.items())[0]
k, v = track
pre_processor = SpectrogramProcessor()
continuity = mir_eval.beat.continuity(v.beats.times, v.beats.times)
print("mir_eval continuity: ", continuity)  # over all metrical variations

continuity_helper = EvaluationHelperFactory.create_evaluation_helper('continuity')
continuity = continuity_helper.calculate_metric(v.beats.times, v.beats.times)
print("my continuity: ", continuity)    # only original metrical level

print("custom mean continuity: ", continuity_helper.calculate_mean_metric({'beats': v.beats.times}, {'beats': v.beats.times}))

continuity_helper.plot_beats(v.beats.times, v.beats.times)
