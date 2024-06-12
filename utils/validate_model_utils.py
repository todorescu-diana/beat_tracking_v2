import numpy as np
from sklearn.model_selection import KFold
from classes.sequences.data_sequence import DataSequence
from classes.sequences.spectrogram_sequence import SpectrogramSequence
from classes.spectrograms.SpectrogramProcessorFactory import SpectrogramProcessorFactory
from constants.constants import MODEL_SAVE_PATH, NUM_FOLDS, PAD_FRAMES, NUM_EPOCHS, PLOT_SAVE_PATH, SUMMARY_SAVE_PATH
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory
from utils.model_utils import build_model, compile_model, train_model, predict


def k_fold_cross_validation(dataset_tracks, n_splits=NUM_FOLDS, epochs=NUM_EPOCHS, dataset_name='', results_dir_path=''):
    print(f"Performing k-fold-cross-validation on {dataset_name.upper()} dataset")
    total_valid_dataset_tracks = 0
    if dataset_tracks is not None:
        dataset_tracks_keys = list(dataset_tracks.keys())

        if len(dataset_tracks_keys):
            # Initialize KFold cross-validator
            kf = KFold(n_splits=2, shuffle=True, random_state=1234)

            # Initialize lists to store evaluation metrics
            accuracy_scores = []
            loss_scores = []
            train_accuracy_scores = []
            train_loss_scores = []
            cemgil_scores = []
            continuity_scores = []
            f_measure_scores = []

            cemgil_helper = EvaluationHelperFactory.create_evaluation_helper('cemgil')
            continuity_helper = EvaluationHelperFactory.create_evaluation_helper('continuity')
            f_measure_helper = EvaluationHelperFactory.create_evaluation_helper('f_measure')

            for fold_idx, (train_index, test_index) in enumerate(kf.split(dataset_tracks_keys)):
                print(f"\t\tFold {fold_idx + 1}/8")
                train_files = [dataset_tracks_keys[i] for i in train_index]
                test_files = [dataset_tracks_keys[i] for i in test_index]

                spectrogram_processor_factory = SpectrogramProcessorFactory()
                mel_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('mel')
                cqt_preprocessor = spectrogram_processor_factory.create_spectrogram_processor('cqt')
                pre_processor = mel_preprocessor # TODO

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
                compile_model(model, summary=True, model_name=dataset_name + f'_fold{fold_idx}')
                history = train_model(model, epochs=epochs, train_data=train, test_data=test, model_name=dataset_name + f'_fold{fold_idx}', plot_save=True, plot_save_path=PLOT_SAVE_PATH)

                # Store training history
                train_accuracy_scores.append(history.history['binary_accuracy'])
                train_loss_scores.append(history.history['loss'])

                # Evaluate the model
                loss, accuracy = model.evaluate(test)

                spectrogram_sequence = SpectrogramSequence(
                    tracks={k: v for k, v in dataset_tracks.items() if k in test_files},
                    pre_processor=pre_processor,
                    pad_frames=2
                )

                total_valid_dataset_tracks = len(spectrogram_sequence)

                # predict for metrics
                _, detections = predict(model, spectrogram_sequence)
                beat_detections = detections
                beat_annotations = {k: {'beats': v.beats.times} for k, v in dataset_tracks.items() if v.beats is not None}

                # calculate remaining / extra metrics
                cemgil_dataset_mean = cemgil_helper.calculate_mean_metric(beat_detections, beat_annotations)
                continuity_dataset_mean = continuity_helper.calculate_mean_metric(beat_detections, beat_annotations)
                f_measure_dataset_mean = f_measure_helper.calculate_mean_metric(beat_detections, beat_annotations)

                # Store evaluation metrics
                accuracy_scores.append(accuracy)
                loss_scores.append(loss)

                cemgil_scores.append(cemgil_dataset_mean)
                continuity_scores.append(continuity_dataset_mean)
                f_measure_scores.append(f_measure_dataset_mean)

            # Calculate average evaluation metrics
            avg_accuracy = np.mean(accuracy_scores)
            avg_loss = np.mean(loss_scores)

            num_tracks = total_valid_dataset_tracks
            if len(cemgil_scores) == len(continuity_scores) == len(f_measure_scores):
                count = len(cemgil_scores)

                if num_tracks:
                    sum_cemgil = {}
                    sum_continuity = {}
                    sum_f_measure = {}
                    for fold_idx in range(2):
                        for key, value in cemgil_scores[fold_idx].items():
                            if key in sum_cemgil:
                                sum_cemgil[key] += value
                            else:
                                sum_cemgil[key] = value
                        # Calculate the mean for each key
                        mean_cemgil = {key: round(sum_value / count, 2) for key, sum_value in sum_cemgil.items()}
                        for key, value in continuity_scores[fold_idx].items():
                            if key in sum_continuity:
                                sum_continuity[key] += value
                            else:
                                sum_continuity[key] = value
                        mean_continuity = {key: round(sum_value / count, 2) for key, sum_value in sum_continuity.items()}
                        for key, value in f_measure_scores[fold_idx].items():
                            if key in sum_f_measure:
                                sum_f_measure[key] += value
                            else:
                                sum_f_measure[key] = value
                        mean_f_measure = {key: round(sum_value / count, 2) for key, sum_value in sum_f_measure.items()}

                    with open(results_dir_path + "/" + dataset_name.upper() + ".txt", "w") as file:
                        file.write(f"{dataset_name.upper()}:\n")
                        file.write(f"\tAverage Binary Accuracy: {avg_accuracy}\n")
                        file.write(f"\tAverage Loss: {avg_loss}\n")
                        file.write(f"\tAverage Cemgil Accuracy: {mean_cemgil}\n")
                        file.write(f"\tAverage Continuity Score: {mean_continuity}\n")
                        file.write(f"\tAverage F-Measure: {mean_f_measure}\n")

