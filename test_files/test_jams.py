import os
import jams

file_path = ""

jam = jams.load(file_path)
annotations = jam.annotations

beat_annotation = annotations.search(namespace='beat_position')
data = beat_annotation.data

for data_object in data:
    beat_time = float(data_object.time)
    print(beat_time)