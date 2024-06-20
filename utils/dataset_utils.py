import os
import jams
import re
import numpy as np
from classes.audio_track import AudioTrack
from constants.constants import DATASET_PATHS, VALID_DATASET_NAMES

def extract_beat_annotations_jams(jams_data):
    for annotation in jams_data['annotations']:
        if annotation['sandbox']['annotation_type'] == 'beat':
            return annotation['data']
    return []

def load_audio_tracks(audio_dir, annot_dir=None, replace_dots_with_underline=False, tiny_aam=False, harmonix_set=False, gtzan_rhythm=False):
    audio_tracks = {}
    # define a regular expression pattern to match either whitespace or a comma
    separator_pattern = re.compile(r'\s+|,')

    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            try: # iterate through audio files in directory
                # check if the file has a .wav (or .mp3) extension
                if file.lower().endswith(('.wav', '.mp3', '.flac')):
                    # extract track ID from file name
                    track_id = os.path.splitext(file)[0]

                    # construct full path to audio file
                    audio_path = os.path.join(root, file)

                    # load beat times from annotations (if available)
                    beat_times = []
                    if annot_dir:
                        # construct full path to annotations file
                        annotation_file = find_annotation_file(annot_dir, track_id, replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm)

                        if annotation_file is not None and os.path.exists(annotation_file):
                            if annotation_file.lower().endswith('.jams'):
                                jam = jams.load(annotation_file)
                                annotations = jam.annotations
                                if harmonix_set is True:
                                    beat_annotation = annotations.search(namespace='beat')[0]
                                elif gtzan_rhythm is True:
                                    beat_annotation = annotations.search(namespace='beat')[0]
                                else:
                                    beat_annotation = annotations.search(namespace='beat_position')[0]
                                # for annotation in beat_annotations:
                                data = beat_annotation.data
                                for data_object in data:
                                    beat_time = float(data_object.time)
                                    beat_times.append(beat_time)

                            elif annotation_file.lower().endswith('.arff'):
                                with open(annotation_file, 'r') as f:
                                    for line in f:
                                        # strip whitespace from the line
                                        line = line.strip()
                                        # check if the line is empty or starts with '@'
                                        if line and not line.startswith('@'):
                                            # print the non-empty, non-@ line
                                            beat_times = [float(separator_pattern.split(line)[0]) for line in f]

                            elif (annotation_file.lower().endswith('.beats') or annotation_file.lower().endswith('.txt')
                                    or annotation_file.lower().endswith('.csv')):
                                with open(annotation_file, 'r') as f:
                                    # Extract beat times from each line
                                    beat_times = [float(separator_pattern.split(line)[0]) for line in f]
                    # Round each number to 3 decimal places
                    # rounded_beat_times = [round(num, 3) for num in beat_times]
                    # beat_times = np.array(rounded_beat_times)
                    beat_times = np.array(beat_times)

                    # create AudioTrack instance
                    if len(beat_times) == 0:
                        beat_times = None
                    audio_track = AudioTrack(audio_path, beat_times)

                    # store AudioTrack instance in dictionary
                    audio_tracks[track_id] = audio_track
            except Exception as e:
                print("Exception occurred: ", e)
                continue

    return audio_tracks

def get_load_dataset_params(dataset_name):
    replace_dots_with_underline = False
    tiny_aam = False
    harmonix_set = False
    gtzan_rhythm = False

    if dataset_name == 'tiny_aam' or dataset_name == 'aam':
        tiny_aam = True
    elif dataset_name == 'harmonix_set':
        harmonix_set = True
    elif dataset_name == 'gtzan_rhythm':
        gtzan_rhythm = True
    else:
        replace_dots_with_underline = True

    return replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm


def load_dataset(dataset_name, replace_dots_with_underline=False, tiny_aam=False, harmonix_set=False, gtzan_rhythm=False):
    if dataset_name not in VALID_DATASET_NAMES:
        return None
    else:
        audio_dir = DATASET_PATHS[dataset_name]["audio_dir"]
        annot_dir = DATASET_PATHS[dataset_name]["annot_dir"]
        
        return load_audio_tracks(audio_dir, annot_dir, replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm)


def find_annotation_file(annotations_dir, track_id, replace_dots_with_underline=False, tiny_aam=False, harmonix_set=False, gtzan_rhythm=False):
    # iterate through files and directories in annotations directory
    for item in os.listdir(annotations_dir):
        # construct the full path
        item_path = os.path.join(annotations_dir, item)
        # check if it's a directory
        if os.path.isdir(item_path):
            # check if the directory name contains the word 'x'
            if track_id in item:
                # recursively search within the directory
                annotation_file = find_annotation_file(item_path, track_id, replace_dots_with_underline)
                # if annotation file found, return it
                if annotation_file:
                    return annotation_file
        # if it's a file and contains the track ID in its name, return its path
        elif os.path.isfile(item_path):
            if tiny_aam is True and track_id.split('_')[0] == item.split('_')[0] and 'beatinfo' in item:
                return item_path
            elif harmonix_set is True and track_id.split('.')[0] == item.split('.')[0]:
                return item_path
            elif replace_dots_with_underline is True and ((track_id.replace('.', '_') in item.split('.')[0]) or (item.split('.')[0] in track_id.replace('.', '_'))):
                return item_path
            elif gtzan_rhythm is True and track_id == item.rsplit('.', 2)[0]:
                return item_path
    # if not found, return None
    return None