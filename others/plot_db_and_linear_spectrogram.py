import sys
sys.path.append('')
from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory

spectrogram_processor_factory = SpectrogramProcessorFactory()
lin_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('lin')
db_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('db')

spectrogram = lin_preprocessor.process('')
lin_preprocessor.plot_spectrogram(spectrogram, duration_s=5)
spectrogram = db_preprocessor.process('')
db_preprocessor.plot_spectrogram(spectrogram, duration_s=5)