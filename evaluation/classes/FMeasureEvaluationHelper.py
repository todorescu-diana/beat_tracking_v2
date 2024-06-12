from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from evaluation.classes.EvaluationHelperBase import EvaluationHelperBase

class FMeasureEvaluationHelper(EvaluationHelperBase):
    def __init__(self, tolerance_ms=70):
        self.tolerance_ms = tolerance_ms

    @staticmethod
    def metric_components():
        return ['precision', 'recall', 'f_score']

    def calculate_tp_fp_fn(self, predicted_beats, ground_truth_beats):
        tp = 0
        fp = 0
        fn = 0

        for pred_beat in predicted_beats:
            found_match = False
            for true_beat in ground_truth_beats:
                if abs(pred_beat - true_beat) <= self.tolerance_ms / 1000:
                    tp += 1
                    found_match = True
                    break
            if not found_match:
                fp += 1

        fn = len(ground_truth_beats) - tp

        return tp, fp, fn

    def calculate_metric(self, predicted_beats=None, ground_truth_beats=None):
        if len(predicted_beats) == 0 or len(ground_truth_beats) == 0:
            return 0., 0., 0.
        else:
            true_positives, false_positives, false_negatives = self.calculate_tp_fp_fn(predicted_beats, ground_truth_beats)
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            if precision + recall:
                f_score = 2. * (precision * recall) / (precision + recall)
            else:
                f_score = 0.
            return round(precision, 2), round(recall, 2), round(f_score, 2)

    def plot_beats(self, predicted_beats, ground_truth_beats, n=None):
        if n is not None:
            predicted_beats = predicted_beats[:n]
            ground_truth_beats = ground_truth_beats[:n]

        plt.ylim(bottom=0)

        for index, x in enumerate(predicted_beats):
            if index != 0:
                lbl = None
            else:
                lbl = 'Predictions'
            plt.axvline(x=x, color='black', linestyle='-', linewidth=2, ymax=0.5, label=lbl)

        for index, beat in enumerate(ground_truth_beats):
            if index != 0:
                lbl = None
            else:
                lbl = 'Ground truth beats'
            plt.axvline(x=beat, color='black', linestyle='--', linewidth=1, ymax=1, label=lbl)

        for beat in ground_truth_beats:
            plt.fill_betweenx(y=[0, 2], x1=beat - self.tolerance_ms / 1000, x2=beat + self.tolerance_ms / 1000,
                              color='forestgreen',
                              alpha=0.3)

        legend_handles = [
            Line2D([0], [0], color='black', linestyle='--', linewidth=1),
            Line2D([0], [0], color='black', linestyle='-', linewidth=2),
            Line2D([0], [0], color='forestgreen', linestyle='-', linewidth=0, marker='s', alpha=0.3, markersize=10),
        ]
        legend_labels = ['Ground Truth Beats', 'Predictions', 'Tolerance Window (70ms)', ]
        plt.title('Predicted Beats and Ground Truth Beats with Cemgil Tolerance Window')

        legend = plt.legend(legend_handles, legend_labels, loc='upper right')
        plt.gca().add_artist(legend)
        plt.xlabel('Time (s)')
        plt.yticks([])
        plt.title('Predicted Beats and Ground Truth Beats with Tolerance Window')

        plt.show()
