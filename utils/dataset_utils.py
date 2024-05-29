import os
import jams
import re

import numpy as np
from classes.audio_track import AudioTrack
from constants.constants import DATASET_PATHS, VALID_DATASET_NAMES

def load_audio_tracks(audio_dir, annot_dir=None, replace_dots_with_underline=False, tiny_aam=False):
    audio_tracks = {}
    # define a regular expression pattern to match either whitespace or a comma
    separator_pattern = re.compile(r'\s+|,')

    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            try: # iterate through audio files in directory
                # check if the file has a .wav (or .mp3) extension
                if file.lower().endswith(('.wav', '.mp3')):
                    # extract track ID from file name
                    track_id = os.path.splitext(file)[0]

                    # construct full path to audio file
                    audio_path = os.path.join(root, file)

                    # load beat times from annotations (if available)
                    beat_times = []
                    if annot_dir:
                        # construct full path to annotations file
                        annotation_file = find_annotation_file(annot_dir, track_id, replace_dots_with_underline, tiny_aam)

                        if annotation_file is not None and os.path.exists(annotation_file):
                            if annotation_file.lower().endswith('.jams'):
                                jam = jams.load(annotation_file)
                                annotations = jam.annotations
                                beat_annotations = annotations.search(namespace='beat_position')

                                for annotation in beat_annotations:
                                    data = annotation.data
                                    for data_object in data:
                                        beat_time = data_object.time
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
                                            beat_times = np.array(beat_times)

                            elif (annotation_file.lower().endswith('.beats') or annotation_file.lower().endswith('.txt')
                                    or annotation_file.lower().endswith('.csv')):
                                with open(annotation_file, 'r') as f:
                                    # Extract beat times from each line
                                    beat_times = [float(separator_pattern.split(line)[0]) for line in f]
                                    # Convert the list to a numpy array
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

    if dataset_name == 'tiny_aam':
        tiny_aam = True
    else:
        replace_dots_with_underline = True

    return replace_dots_with_underline, tiny_aam


def load_dataset(dataset_name, replace_dots_with_underline=False, tiny_aam=False):
    if dataset_name not in VALID_DATASET_NAMES:
        return None
    else:
        audio_dir = DATASET_PATHS[dataset_name]["audio_dir"]
        annot_dir = DATASET_PATHS[dataset_name]["annot_dir"]
        
        return load_audio_tracks(audio_dir, annot_dir, replace_dots_with_underline, tiny_aam)


def find_annotation_file(annotations_dir, track_id, replace_dots_with_underline=False, tiny_aam=False):
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
            if tiny_aam is True and track_id.split('.')[0] == item.split('_')[0] and 'beatinfo' in item:
                return item_path
            elif replace_dots_with_underline is True and ((track_id.replace('.', '_') in item.split('.')[0]) or (item.split('.')[0] in track_id.replace('.', '_'))):
                return item_path
    # if not found, return None
    return None