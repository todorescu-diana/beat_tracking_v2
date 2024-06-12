from abc import ABC, abstractmethod


class SpectrogramProcessorBase(ABC):
    @abstractmethod
    def process(self, audio_path):
        pass

    @abstractmethod
    def plot_spectrogram(self, spectrogram, duration_s, cmap):
        pass

    @staticmethod
    @abstractmethod
    def spectrogram_type():
        pass
