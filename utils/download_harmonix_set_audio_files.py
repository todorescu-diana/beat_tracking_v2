# import csv
# import os
# from pytube import YouTube, exceptions
# from moviepy.editor import AudioFileClip

# def csv_to_dict(file_path):
#     result_dict = {}
#     with open(file_path, mode='r', encoding='utf-8') as file:
#         csv_reader = csv.reader(file)
#         next(csv_reader)
#         for row in csv_reader:
#             file_name, youtube_url = row
#             result_dict[file_name] = youtube_url
#     return result_dict

# def create_folder(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def download_audio(url, output_path):
#     yt = YouTube(url)
#     audio_stream = yt.streams.filter(only_audio=True).first()
#     downloaded_file = audio_stream.download(output_path=output_path)
#     return downloaded_file

# def convert_to_wav(input_file, output_file):
#     audio_clip = AudioFileClip(input_file)
#     audio_clip.write_audiofile(output_file, codec='pcm_s16le')
#     audio_clip.close()

# def main(csv_file_path):
#     # Read the CSV and create the dictionary
#     file_url_dict = csv_to_dict(csv_file_path)
    
#     # Get the directory where the CSV is located
#     csv_directory = os.path.dirname(csv_file_path)
    
#     # Create the audio_files folder
#     audio_files_folder = os.path.join(csv_directory, "audio_files")
#     create_folder(audio_files_folder)
    
#     # Process each file in the dictionary
#     for file_name, youtube_url in file_url_dict.items():
#         try:
#             print(f"Downloading and converting {file_name} from {youtube_url}")
            
#             # Download the audio from YouTube
#             downloaded_file = download_audio(youtube_url, audio_files_folder)
            
#             # Define the output WAV file path
#             wav_file_path = os.path.join(audio_files_folder, f"{file_name}.wav")
            
#             # Convert the downloaded file to WAV
#             convert_to_wav(downloaded_file, wav_file_path)
            
#             # Optionally, remove the original downloaded file to save space
#             os.remove(downloaded_file)
            
#             print(f"Saved {wav_file_path}")
#         except exceptions.VideoUnavailable:
#             print(f"Video {youtube_url} is unavailable. Skipping...")
#         except Exception as e:
#             print(f"An error occurred with video {youtube_url}: {str(e)}")

# csv_file_path = ''
# main(csv_file_path)

import csv
import os
from pytube import YouTube, exceptions
from moviepy.editor import AudioFileClip
from pydub import AudioSegment

def csv_to_dict(file_path):
    result_dict = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            file_name = row[0]
            youtube_url_instruction = row[1]
            # Split the URL and instruction
            if ' ' in youtube_url_instruction:
                youtube_url, instruction = youtube_url_instruction.rsplit(' ', 1)
            else:
                youtube_url = youtube_url_instruction
                instruction = None
            result_dict[file_name] = (youtube_url, instruction)
    return result_dict

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_audio(url, output_path):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    downloaded_file = audio_stream.download(output_path=output_path)
    return downloaded_file

def convert_to_wav(input_file, output_file, instruction=None):
    audio_clip = AudioFileClip(input_file)
    temp_output = "temp.wav"
    audio_clip.write_audiofile(temp_output, codec='pcm_s16le')
    audio_clip.close()

    audio = AudioSegment.from_file(temp_output)
    
    if instruction:
        print(f"Processing instruction: {instruction}")
        if "backwards" in instruction:
            audio = audio.reverse()
        elif "sped-up" in instruction:
            speed_change = int(instruction.split('%')[0])
            print(f"Speed change value: {speed_change}")
            if "needs to be" in instruction:
                speed_change = 1 + (speed_change / 100.0)
            else:
                speed_change = 1 - (speed_change / 100.0)
            print(f"Playback speed change: {speed_change}")
            audio = audio.speedup(playback_speed=speed_change)

    audio.export(output_file, format="wav")
    os.remove(temp_output)

def main(csv_file_path):
    # Read the CSV and create the dictionary
    file_url_dict = csv_to_dict(csv_file_path)
    
    # Get the directory where the CSV is located
    csv_directory = os.path.dirname(csv_file_path)
    
    # Create the audio_files folder
    audio_files_folder = os.path.join(csv_directory, "audio_files")
    create_folder(audio_files_folder)
    
    # Process each file in the dictionary
    for file_name, (youtube_url, instruction) in file_url_dict.items():
        try:
            print(f"Downloading and converting {file_name} from {youtube_url} with instruction: {instruction}")
            
            # Download the audio from YouTube
            downloaded_file = download_audio(youtube_url, audio_files_folder)
            
            # Define the output WAV file path
            wav_file_path = os.path.join(audio_files_folder, f"{file_name}.wav")
            
            # Convert the downloaded file to WAV with processing instruction
            convert_to_wav(downloaded_file, wav_file_path, instruction)
            
            # Optionally, remove the original downloaded file to save space
            os.remove(downloaded_file)
            
            print(f"Saved {wav_file_path}")
        except exceptions.VideoUnavailable:
            print(f"Video {youtube_url} is unavailable. Skipping...")
        except Exception as e:
            print(f"An error occurred with video {youtube_url}: {str(e)}")
            
# Example usage
csv_file_path = ''
main(csv_file_path)
