import sys
sys.path.append('')
from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory
from utils.utils import play_audio_with_clicktrack
from constants.constants import VALID_DATASET_NAMES
from utils.dataset_utils import get_load_dataset_params, load_dataset


dataset_name = VALID_DATASET_NAMES[8]

replace_dots_with_underline, tiny_aam, harmonix_set = get_load_dataset_params(dataset_name)

dataset_tracks = load_dataset(dataset_name, replace_dots_with_underline, tiny_aam, harmonix_set)

spectrogram_processor_factory = SpectrogramProcessorFactory()
mel_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('mel')
cqt_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('cqt')

pre_processor = mel_preprocessor
spectrogram = pre_processor.process(dataset_tracks['0001_12step'].audio_path)
pre_processor.plot_spectrogram(spectrogram, duration=5)

# print(dataset_tracks['0992_wrongthinwhiteduke'])
# print(dataset_tracks['0001_12step'].beats.times)

# play_audio_with_clicktrack(dataset_tracks['0001_12step'], dataset_tracks['0001_12step'].beats.times)

# import os
# from pydub import AudioSegment
# from pydub.silence import detect_leading_silence

# def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
#     '''
#     sound is a pydub.AudioSegment
#     silence_threshold is in dB
#     chunk_size is in ms
#     iterate over chunks until you find the first one with sound
#     '''
#     trim_ms = 0  # ms
#     assert chunk_size > 0  # to avoid infinite loop
#     while trim_ms < len(sound):
#         chunk = sound[trim_ms:trim_ms + chunk_size]
#         if chunk.dBFS > silence_threshold:
#             return trim_ms
#         trim_ms += chunk_size

#     return trim_ms

# def trim_leading_silence(input_folder, output_folder, silence_threshold=-50.0, chunk_size=10):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     for filename in os.listdir(input_folder):
#         if filename.endswith('.wav'):
#             file_path = os.path.join(input_folder, filename)
#             sound = AudioSegment.from_wav(file_path)
            
#             start_trim = detect_leading_silence(sound, silence_threshold, chunk_size)
#             end_trim = detect_leading_silence(sound.reverse(), silence_threshold, chunk_size)
            
#             duration = len(sound)
#             trimmed_sound = sound[start_trim:duration - end_trim]
            
#             output_path = os.path.join(output_folder, filename)
#             trimmed_sound.export(output_path, format="wav")
#             print(f"Processed {filename}")

# # Set your input and output folders here
# input_folder = 'path_to_your_input_folder'
# output_folder = 'path_to_your_output_folder'

# trim_leading_silence('', '')
