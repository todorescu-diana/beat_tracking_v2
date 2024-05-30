import sys
sys.path.append('')
from classes.spectrogram_sequence import SpectrogramSequence
from utils.model_utils import predict
import mir_eval
import mirdata
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory
from classes.spectrogram_processor import SpectrogramProcessor
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

cemgil = mir_eval.beat.cemgil(v.beats.times, det[k]['beats'])
print("mir_eval cemgil: ", cemgil)  # over all metrical variations

cemgil_helper = EvaluationHelperFactory.create_evaluation_helper('cemgil')
cemgil = cemgil_helper.calculate_metric(det[k]['beats'], v.beats.times)
print("my cemgil: ", cemgil)    # only original metrical level

print("custom mean cemgil: ", cemgil_helper.calculate_mean_metric(det, {k: {'beats': v.beats.times}}))

cemgil_helper.plot_beats(det[k]['beats'], v.beats.times)
