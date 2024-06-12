import sys
sys.path.append('')
from classes.sequences.spectrogram_sequence import SpectrogramSequence
from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory
from utils.model_utils import predict
import mir_eval
import mirdata
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory
from keras.models import load_model

gtzan = mirdata.initialize('gtzan_genre', version='mini')
tracks = gtzan.load_tracks()
track = list(tracks.items())[0]
k, v = track

model = load_model('')

spectrogram_processor_factory = SpectrogramProcessorFactory()
mel_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('mel')
cqt_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('cqt')
pre_processor = mel_preprocessor
spectrogram_sequence = SpectrogramSequence(
    data_sequence_tracks = {k: v},
    data_sequence_pre_processor=pre_processor,
    data_sequence_pad_frames=2
)

act, det = predict(model, spectrogram_sequence)

f_measure = mir_eval.beat.f_measure(v.beats.times, det[k]['beats'])
print("mir_eval f_measure: ", f_measure)

f_measure_helper = EvaluationHelperFactory.create_evaluation_helper('f_measure')
precision, recall, f_score = f_measure_helper.calculate_metric(det[k]['beats'], v.beats.times)
print("my f_measure: ", precision, recall, f_score)

print("custom mean f_score: ", f_measure_helper.calculate_mean_metric(det, {k: {'beats': v.beats.times}}))

f_measure_helper.plot_beats(det[k]['beats'], v.beats.times)
