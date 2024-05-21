import numpy as np
from sklearn.model_selection import KFold

from classes.data_sequence import DataSequence
from classes.spectrogram_processor import SpectrogramProcessor
from classes.spectrogram_sequence import SpectrogramSequence
from constants.constants import NUM_FOLDS, PAD_FRAMES, NUM_EPOCHS
from evaluation.classes.EvaluationHelperFactory import EvaluationHelperFactory
from utils.model_utils import build_model, compile_model, train_model, predict


def k_fold_cross_validation(dataset_tracks, n_splits=NUM_FOLDS, epochs=NUM_EPOCHS, dataset_name='', results_dir_path=''):
    print(f"Performing k-fold-cross-validation on {dataset_name.upper()} dataset")
    if dataset_tracks is not None:
        dataset_tracks_keys = list(dataset_tracks.keys())

        if len(dataset_tracks_keys):
            # Initialize KFold cross-validator
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)

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

                pre_processor = SpectrogramProcessor()

                train = DataSequence(
                    data_sequence_tracks={k: v for k, v in dataset_tracks.items() if k in train_files},
                    data_sequence_pre_processor=pre_processor,
                    data_sequence_pad_frames=PAD_FRAMES
                )
                train.widen_beat_targets()

                test = DataSequence(
                    data_sequence_tracks={k: v for k, v in dataset_tracks.items() if k in test_files},
                    data_sequence_pre_processor=pre_processor,
                    data_sequence_pad_frames=PAD_FRAMES
                )
                test.widen_beat_targets()

                model = build_model()
                compile_model(model)
                history = train_model(model, train_data=train, test_data=test, epochs=epochs)

                # Store training history
                train_accuracy_scores.append(history.history['binary_accuracy'])
                train_loss_scores.append(history.history['loss'])

                # Evaluate the model
                loss, accuracy = model.evaluate(test)

                spectrogram_sequence = SpectrogramSequence(
                    data_sequence_tracks=dataset_tracks,
                    data_sequence_pre_processor=pre_processor,
                    data_sequence_pad_frames=2
                )
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

            num_tracks = len(dataset_tracks)

            cemgil_metric_components = cemgil_helper.metric_components()
            continuity_metric_components = continuity_helper.metric_components()
            f_measure_metric_components = f_measure_helper.metric_components()

            for fold_idx in range(n_splits):
                for component in cemgil_metric_components:
                    cemgil_scores[fold_idx][component] /= num_tracks
                    cemgil_scores[fold_idx][component] = round(cemgil_scores[fold_idx][component], 2)
                for component in continuity_metric_components:
                    continuity_scores[fold_idx][component] /= num_tracks
                    continuity_scores[fold_idx][component] = round(continuity_scores[fold_idx][component], 2)
                for component in f_measure_metric_components:
                    f_measure_scores[fold_idx][component] /= num_tracks
                    f_measure_scores[fold_idx][component] = round(f_measure_scores[fold_idx][component], 2)
            print(f"{dataset_name.upper()}:")
            print(f"\t\t\t\t{dataset_name.upper()} Average Binary Accuracy:", avg_accuracy)
            print(f"\t\t\t\t{dataset_name.upper()} Average Loss:", avg_loss)
            print(f"\t\t\t\t{dataset_name.upper()} Average Cemgil Accuracy:", cemgil_scores)
            print(f"\t\t\t\t{dataset_name.upper()} Average Continuity Score:", continuity_scores)
            print(f"\t\t\t\t{dataset_name.upper()} Average F-Measure:", f_measure_scores)

            with open(results_dir_path + "/" + dataset_name.upper() + ".txt", "w") as file:
                file.write(f"{dataset_name.upper()}:\n")
                file.write(f"\t\t\t\t{dataset_name.upper()} Average Binary Accuracy: {avg_accuracy}\n")
                file.write(f"\t\t\t\t{dataset_name.upper()} Average Loss: {avg_loss}\n")
                file.write(f"\t\t\t\t{dataset_name.upper()} Average Cemgil Accuracy: {cemgil_scores}\n")
                file.write(f"\t\t\t\t{dataset_name.upper()} Average Continuity Score: {continuity_scores}\n")
                file.write(f"\t\t\t\t{dataset_name.upper()} Average F-Measure: {f_measure_scores}\n")

