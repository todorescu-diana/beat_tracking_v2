import sys
sys.path.append('')
from classes.sequences.data_sequence import DataSequence
from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory
from sklearn.model_selection import train_test_split
from constants.constants import MODEL_SAVE_PATH, PAD_FRAMES, PLOT_SAVE_PATH, SUMMARY_SAVE_PATH, VALID_DATASET_NAMES
from utils.dataset_utils import get_load_dataset_params, load_dataset
from utils.model_utils import build_model, compile_model, train_model
import tensorflow as tf

with tf.device('/GPU:0'):
    dataset_name = VALID_DATASET_NAMES[8]

    replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm = get_load_dataset_params(dataset_name)
    
    spectrogram_processor_factory = SpectrogramProcessorFactory()
    mel_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('mel')
    pre_processor = mel_preprocessor
    pre_processor_type = pre_processor.spectrogram_type()

    dataset_tracks = load_dataset(dataset_name, replace_dots_with_underline, tiny_aam, harmonix_set, gtzan_rhythm)
    model_name = 'trained_' + dataset_name + '_spectrograms_data_v2_' + pre_processor_type

    if dataset_tracks is not None:
        train_files, test_files = train_test_split(list(dataset_tracks.keys()), test_size=0.2, random_state=1234)

        train = DataSequence(
                tracks={k: v for k, v in dataset_tracks.items() if k in train_files},
                pre_processor=pre_processor,
                pad_frames=PAD_FRAMES
            )
        train.widen_beat_targets()
        test = DataSequence(
                tracks={k: v for k, v in dataset_tracks.items() if k in test_files},
                pre_processor=pre_processor,
                pad_frames=PAD_FRAMES
            )
        test.widen_beat_targets()

        model = build_model()
        compile_model(model, lr=0.002, summary=True, model_name=model_name, summary_save_path=SUMMARY_SAVE_PATH)
        train_model(model, train_data=train, test_data=test, save_model=True, model_name=model_name, model_save_path=MODEL_SAVE_PATH, plot_save=True, plot_save_path=PLOT_SAVE_PATH)











# import sys
# sys.path.append('')
# from classes.sequences.data_sequence import DataSequence
# from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory
# from sklearn.model_selection import train_test_split
# from constants.constants import MODEL_SAVE_PATH, PAD_FRAMES, PLOT_SAVE_PATH, SUMMARY_SAVE_PATH, VALID_DATASET_NAMES
# from utils.dataset_utils import get_load_dataset_params, load_dataset
# from utils.model_utils import build_model, compile_model, train_model
# import tensorflow as tf

# with tf.device('/GPU:0'):
#     dataset_name_1 = VALID_DATASET_NAMES[1]
#     dataset_name_2 = VALID_DATASET_NAMES[3]
#     dataset_name_3 = VALID_DATASET_NAMES[4]
#     dataset_name_4 = VALID_DATASET_NAMES[7]

#     replace_dots_with_underline_1, tiny_aam_1, harmonix_set_1, gtzan_rhythm_1 = get_load_dataset_params(dataset_name_1)
#     replace_dots_with_underline_2, tiny_aam_2, harmonix_set_2, gtzan_rhythm_2 = get_load_dataset_params(dataset_name_2)
#     replace_dots_with_underline_3, tiny_aam_3, harmonix_set_3, gtzan_rhythm_3 = get_load_dataset_params(dataset_name_3)
#     replace_dots_with_underline_4, tiny_aam_4, harmonix_set_4, gtzan_rhythm_4 = get_load_dataset_params(dataset_name_4)
    
#     mel_preprocessor = SpectrogramProcessorFactory.create_spectrogram_processor('mel')

#     pre_processor = mel_preprocessor
#     pre_processor_type = pre_processor.spectrogram_type()

#     dataset_tracks_1 = load_dataset(dataset_name_1, replace_dots_with_underline_1, tiny_aam_1, harmonix_set_1, gtzan_rhythm_1)
#     dataset_tracks_2 = load_dataset(dataset_name_2, replace_dots_with_underline_2, tiny_aam_2, harmonix_set_2, gtzan_rhythm_2)
#     dataset_tracks_3 = load_dataset(dataset_name_3, replace_dots_with_underline_3, tiny_aam_3, harmonix_set_3, gtzan_rhythm_3)
#     dataset_tracks_4 = load_dataset(dataset_name_4, replace_dots_with_underline_4, tiny_aam_4, harmonix_set_4, gtzan_rhythm_4)
#     # model_name = 'trained_' + dataset_name + '_v2_' + pre_processor_type
#     model_name = 'trained_' + dataset_name_1 + '_' + dataset_name_2 + '_' + dataset_name_3 + '_' + dataset_name_4 + '_v2_' + pre_processor_type
#     # model_name = 'trained_' + dataset_name + '_rounded_to_3_decimals' + '_v2_' + pre_processor_type

#     dataset_tracks = dataset_tracks_1 | dataset_tracks_2 | dataset_tracks_3 | dataset_tracks_4

#     # print(len(list(dataset_tracks_1.keys())))
#     # print(len(list(dataset_tracks_2.keys())))
#     # print(len(list(dataset_tracks_3.keys())))
#     # print(len(list(dataset_tracks.keys())))

#     if dataset_tracks is not None:
#         train_files, test_files = train_test_split(list(dataset_tracks.keys()), test_size=0.2, random_state=1234)

#         train = DataSequence(
#                 tracks={k: v for k, v in dataset_tracks.items() if k in train_files},
#                 pre_processor=pre_processor,
#                 pad_frames=PAD_FRAMES
#             )
#         train.widen_beat_targets()
#         # print("---", len(train))
#         test = DataSequence(
#                 tracks={k: v for k, v in dataset_tracks.items() if k in test_files},
#                 pre_processor=pre_processor,
#                 pad_frames=PAD_FRAMES
#             )
#         test.widen_beat_targets()
#         # print(len(test))

#         model = build_model()
#         compile_model(model, lr=0.002, summary=True, model_name=model_name, summary_save_path=SUMMARY_SAVE_PATH)
#         train_model(model, train_data=train, test_data=test, save_model=True, model_name=model_name, model_save_path=MODEL_SAVE_PATH, plot_save=True, plot_save_path=PLOT_SAVE_PATH)