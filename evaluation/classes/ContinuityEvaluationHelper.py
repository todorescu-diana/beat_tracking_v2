import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from evaluation.classes.EvaluationHelperBase import EvaluationHelperBase
from utils.utils import sum_continuous_true_segments, get_longest_continuous_segment_length, \
    get_longest_continuous_segment_bounds, get_all_except_longest_continuous_segment_bounds
from utils.beat_utils import get_metrical_level_variations, get_metrical_level_variations_names


class ContinuityEvaluationHelper(EvaluationHelperBase):
    def __init__(self, tolerance_factor=0.175):
        self.tolerance_factor = tolerance_factor

    def get_tolerance_window_list(self, annotations):
        tolerance_window_list = []

        for j in range(1, len(annotations)):
            ann_interval = annotations[j] - annotations[j - 1]  # Calculate inter-annotation interval
            tolerance_window = ann_interval * self.tolerance_factor

            tolerance_window_list.append(tolerance_window)
            if j == 1:
                tolerance_window_list.append(tolerance_window)

        return tolerance_window_list

    def get_beat_successes(self, annotations, detected_beats):
        used_annotations = [False] * len(annotations)
        beat_successes = [False] * len(detected_beats)

        for i in range(1, len(detected_beats)):
            for j in range(1, len(annotations)):
                if not used_annotations[j]:
                    ann_interval = annotations[j] - annotations[j - 1]  # Calculate inter-annotation interval
                    det_interval = detected_beats[i] - detected_beats[i - 1]  # Calculate inter-beat interval

                    tolerance_window = ann_interval * self.tolerance_factor

                    if (annotations[j - 1] - tolerance_window / 2 <= detected_beats[i - 1] <= annotations[
                        j - 1] + tolerance_window / 2 and
                            annotations[j] - tolerance_window / 2 <= detected_beats[i] <= annotations[
                                j] + tolerance_window / 2 and
                            ann_interval - tolerance_window / 2 < det_interval < ann_interval + tolerance_window / 2):
                        used_annotations[j] = True
                        beat_successes[i] = True

                        if j == 1:
                            used_annotations[0] = True
                            beat_successes[0] = True
                        break  # Move to the next detected beat
        return beat_successes

    def calculate_cmlc(self, detected_beats, annotations):
        if len(annotations) < 2 or len(detected_beats) < 2:
            return 0.0  # Cannot compute CMLc with less than 2 annotations

        longest_correct_segment = 0
        beat_successes = self.get_beat_successes(annotations, detected_beats)

        longest_correct_segment = get_longest_continuous_segment_length(beat_successes)
        if longest_correct_segment == 0. or longest_correct_segment == 1.:
            return 0.
        
        # Adjusting the calculation of longest_correct_segment
        longest_correct_segment = longest_correct_segment / len(annotations)

        return round(longest_correct_segment, 2)

    def calculate_amlc(self, detected_beats, annotations):
        """
        Calculate the Correct Metrical Level, Continuity (CMLc) score.

        Args:
        - detected_beats (list): A list of detected beat timestamps.
        - annotations (list): A list of ground truth annotation timestamps.
        - tolerance_factor (float): Tolerance factor relative to the inter-annotation interval.

        Returns:
        - cmlc_score (float): The CMLc score, representing the longest continuously correct segment.
        """
        amlc_score = 0.

        metrical_level_variations = get_metrical_level_variations(annotations)

        for reference_beats in metrical_level_variations:
            cmlc = self.calculate_cmlc(detected_beats, reference_beats)
            amlc_score = max(amlc_score, cmlc)

        return round(amlc_score, 2)

    def calculate_cmlt(self, detected_beats, annotations):
        if len(annotations) < 2 or len(detected_beats) < 2:
            return 0.0  # Cannot compute CMLc with less than 2 annotations

        used_annotations = [False] * len(annotations)
        beat_successes = [False] * len(detected_beats)

        for i in range(1, len(detected_beats)):
            for j in range(1, len(annotations)):
                if not used_annotations[j]:
                    ann_interval = annotations[j] - annotations[j - 1]  # Calculate inter-annotation interval
                    det_interval = detected_beats[i] - detected_beats[i - 1]  # Calculate inter-beat interval

                    tolerance_window = ann_interval * self.tolerance_factor

                    if (annotations[j - 1] - tolerance_window / 2 <= detected_beats[i - 1] <= annotations[
                        j - 1] + tolerance_window / 2 and
                            annotations[j] - tolerance_window <= detected_beats[i] <= annotations[
                                j] + tolerance_window / 2 and
                            ann_interval - tolerance_window / 2 < det_interval < ann_interval + tolerance_window / 2):
                        used_annotations[j] = True
                        beat_successes[i] = True

                        if j == 1:
                            used_annotations[0] = True
                            beat_successes[0] = True

                        break  # Move to the next detected beat

        correct_segments_sum = sum_continuous_true_segments(beat_successes)
        if not correct_segments_sum:
            return 0.
        # Adjusting the calculation of longest_correct_segment
        cmlt_score = correct_segments_sum / len(annotations)

        return round(cmlt_score, 2)

    def calculate_amlt(self, detected_beats, annotations):
        """
        Calculate the Correct Metrical Level, Continuity (CMLc) score.

        Args:
        - detected_beats (list): A list of detected beat timestamps.
        - annotations (list): A list of ground truth annotation timestamps.
        - tolerance_factor (float): Tolerance factor relative to the inter-annotation interval.

        Returns:
        - cmlc_score (float): The CMLc score, representing the longest continuously correct segment.
        """
        amlt_score = 0.

        metrical_level_variations = get_metrical_level_variations(annotations)

        for reference_beats in metrical_level_variations:
            cmlt = self.calculate_cmlt(detected_beats, reference_beats)
            amlt_score = max(amlt_score, cmlt)

        return round(amlt_score, 2)

    def calculate_metric(self, predicted_beats, ground_truth_beats):
        """
            Parameters
            ----------
            predicted_beats : np.ndarray
                reference beat times, in seconds
            ground_truth_beats : np.ndarray
                query beat times, in seconds

            Returns
            -------
            CMLc : float
                Correct Metrical Level in Continuity
            CMLt : float
                Correct Metrical Level in Timing
            AMLc : float
                Allowed / Any Metrical Level in Continuity
            AMLt : float
                Allowed / Any Metrical Level in Timing
            """
        cmlc = self.calculate_cmlc(predicted_beats, ground_truth_beats)
        cmlt = self.calculate_cmlt(predicted_beats, ground_truth_beats)
        amlc = self.calculate_amlc(predicted_beats, ground_truth_beats)
        amlt = self.calculate_amlt(predicted_beats, ground_truth_beats)

        return cmlc, cmlt, amlc, amlt

    @staticmethod
    def metric_components():
        return ['CMLc', 'CMLt', 'AMLc', 'AMLt']

    def plot_cmlc_cmlt(self, ground_truth_beats, predicted_beats, metrical_level_name):
        plt.ylim(bottom=0, top=1)

        for index, x in enumerate(predicted_beats):
            if index != 0:
                lbl = None
            else:
                lbl = 'Predictions'

            plt.axvline(x=x, color='black', linestyle='-', linewidth=2, ymax=0.5, label=lbl, zorder=2)

        beat_successes = self.get_beat_successes(ground_truth_beats, predicted_beats)
        longest_continuous_segment_bounds = get_longest_continuous_segment_bounds(beat_successes)
        lower_bound, upper_bound = longest_continuous_segment_bounds
        all_except_longest_continuous_segment_bounds = get_all_except_longest_continuous_segment_bounds(beat_successes)
        tolerance_window_list = self.get_tolerance_window_list(ground_truth_beats)

        for bounds in all_except_longest_continuous_segment_bounds:
            plt.fill_betweenx([0, 0.75], x1=predicted_beats[lower_bound] - tolerance_window_list[lower_bound],
                              x2=predicted_beats[upper_bound] + tolerance_window_list[upper_bound], color='gold',
                              alpha=0.3)
        if lower_bound != upper_bound:
            plt.fill_betweenx([0, 0.75], x1=predicted_beats[lower_bound] - tolerance_window_list[lower_bound],
                              x2=predicted_beats[upper_bound] + tolerance_window_list[upper_bound], color='gold', alpha=0.9)

        for index, beat in enumerate(ground_truth_beats):
            if index != 0:
                lbl = None
            else:
                lbl = 'Reference beats'

            plt.axvline(x=beat, color='black', linestyle='--', linewidth=1, ymax=1, label=lbl)

            plt.fill_betweenx([0, 1], x1=beat - tolerance_window_list[index], x2=beat + tolerance_window_list[index],
                              color='forestgreen', alpha=0.25)

        # Create custom legend entries
        legend_handles = [
            Line2D([0], [0], color='black', linestyle='--', linewidth=1),
            Line2D([0], [0], color='black', linestyle='-', linewidth=2),
            Line2D([0], [0], color='forestgreen', linestyle='-', linewidth=0, marker='s', alpha=0.3, markersize=10),
            Line2D([0], [0], color='gold', linestyle='-', linewidth=0, marker='s', alpha=0.3, markersize=10),
            Line2D([0], [0], color='gold', linestyle='-', linewidth=0, marker='s', alpha=0.9, markersize=10),
        ]
        legend_labels = ['Ground Truth Beats', 'Predictions', 'Tolerance Window (+-17.5% of inter-annotation-interval)',
                         'Continuous segments', 'Longest continuous segment']
        plt.title('Predicted Beats and Reference Beats with Tolerance Window, CMLc, CMLt, ' + metrical_level_name)

        # Create a legend object
        legend = plt.legend(legend_handles, legend_labels, loc='upper right')

        # Customize legend position
        plt.gca().add_artist(legend)

        plt.xlabel('Time (s)')
        plt.yticks([])
        # plt.legend()
        # plt.grid(True)
        plt.show()

    def plot_beats(self, predicted_beats, ground_truth_beats):
        metrical_level_variations = get_metrical_level_variations(ground_truth_beats)
        metrical_level_names = get_metrical_level_variations_names()

        for metrical_level_variation, metrical_level_name in zip(metrical_level_variations, metrical_level_names):
            self.plot_cmlc_cmlt(metrical_level_variation, predicted_beats, metrical_level_name)




