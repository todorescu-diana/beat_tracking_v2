import sys

from classes.spectrograms.LinearSpectrogram import LinearSpectrogramProcessor
sys.path.append('')
from classes.spectrograms.DecibelSpectrogramProcessor import DecibelSpectrogramProcessor
from classes.spectrograms.MelSpectrogramProcessor import MelSpectrogramProcessor


class SpectrogramProcessorFactory:
    _registry = {}

    @classmethod
    def register_processor(cls, type, processor_class):
        cls._registry[type] = processor_class

    @classmethod
    def create_spectrogram_processor(cls, type, **kwargs):
        if type not in cls._registry:
            raise ValueError(f"Invalid type: {type}")
        processor_class = cls._registry[type]
        return processor_class(**kwargs)

SpectrogramProcessorFactory.register_processor('db', DecibelSpectrogramProcessor)
SpectrogramProcessorFactory.register_processor('lin', LinearSpectrogramProcessor)
SpectrogramProcessorFactory.register_processor('mel', MelSpectrogramProcessor)