import sys
sys.path.append('')
from sklearn.model_selection import train_test_split

from constants.constants import VALID_DATASET_NAMES
from utils.dataset_utils import get_load_dataset_params, load_dataset
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
# Transform the original dictionary
train_files, test_files = train_test_split(list(tracks.keys()), test_size=0.2, random_state=1234)

dataset_name = VALID_DATASET_NAMES[0]

replace_dots_with_underline, tiny_aam, harmonix_set = get_load_dataset_params(dataset_name)


dataset_tracks = load_dataset(dataset_name, replace_dots_with_underline, tiny_aam, harmonix_set)
new_dict = {key: {'beats': value.beats.times} for key, value in dataset_tracks.items()}
k, v = track

model = load_model('models/saved/trained_gtzan_v2_mel_best.h5')

spectrogram_processor_factory = SpectrogramProcessorFactory()
mel_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('mel')
pre_processor = mel_preprocessor

spectrogram_sequence = SpectrogramSequence(
    tracks = dataset_tracks,
    pre_processor=pre_processor,
    pad_frames=2
)

act, det = predict(model, spectrogram_sequence)

# cemgil = mir_eval.beat.cemgil(det, tracks)
# print("mir_eval cemgil: ", cemgil)  # over all metrical variations

cemgil_helper = EvaluationHelperFactory.create_evaluation_helper('cemgil')
# cemgil = cemgil_helper.calculate_metric(det, tracks)
# print("my cemgil: ", cemgil)    # only original metrical level

print("custom mean cemgil: ", cemgil_helper.calculate_mean_metric(det, new_dict))

# cemgil_helper.plot_beats(det[k]['beats'], v.beats.times)
