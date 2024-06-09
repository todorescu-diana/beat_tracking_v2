import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from evaluation.classes.EvaluationHelperBase import EvaluationHelperBase


class CemgilEvaluationHelper(EvaluationHelperBase):
    def __init__(self, tolerance_ms=40):
        self.tolerance_ms = tolerance_ms

    @staticmethod
    def gaussian_error(x, mu=0, sigma=1):
        return np.exp((-1.) * ((x - mu) ** 2) / (2.0 * (sigma ** 2)))

    @staticmethod
    def gaussian(x, mu=0, sigma=1.):
        return np.exp(-1 * ((x - mu) / sigma) ** 2) / (2.0 * sigma * np.sqrt(2 * np.pi))

    def calculate_metric(self, predicted_beats, ground_truth_beats):
        total_score = 0

        for true_beat in ground_truth_beats:
            min_error = float('inf')

            for pred_beat in predicted_beats:
                error = abs(pred_beat - true_beat)
                min_error = min(error, min_error)

            weight = self.gaussian_error(min_error, mu=0, sigma=self.tolerance_ms / 1000)
            total_score += weight

        if (len(predicted_beats) > 0 and len(ground_truth_beats) > 0):
            cemgil_accuracy = total_score / (.5 * (len(predicted_beats) + len(ground_truth_beats))) 
        else:
            cemgil_accuracy = 0.
        
        return round(cemgil_accuracy, 2)

    @staticmethod
    def metric_components():
        return ['cemgil']

    def plot_beats(self, predicted_beats, ground_truth_beats, n=None):
        if n is not None:
            predicted_beats = predicted_beats[:n]
            ground_truth_beats = ground_truth_beats[:n]

        plt.ylim(bottom=0)

        for index, beat in enumerate(ground_truth_beats):
            if index != 0:
                lbl = None
            else:
                lbl = 'Ground truth beats'
            plt.axvline(x=beat, color='black', linestyle='--', linewidth=1, ymax=1, label=lbl)

            x_values = np.linspace(beat - 10 * self.tolerance_ms / 1000, beat + 10 * self.tolerance_ms / 1000, 1000)
            y_values = [self.gaussian_error(x, mu=beat, sigma=self.tolerance_ms / 1000) for x in x_values]

            plt.plot(x_values, y_values, color='black', linewidth=0.5)
            plt.fill_between(x_values, y_values, color='gray', alpha=0.25)

            tolerance_window_x_linspace = np.linspace(beat - self.tolerance_ms / 1000, beat + self.tolerance_ms / 1000,
                                                      100)
            tolerance_window_y_linspace = [self.gaussian_error(x, mu=beat, sigma=self.tolerance_ms / 1000) for x in
                                           tolerance_window_x_linspace]

            min_x = min(tolerance_window_x_linspace)
            max_x = beat + 6 * self.tolerance_ms / 1000

            plt.xlim(left=0, right=max_x)
            plt.fill_between(tolerance_window_x_linspace, tolerance_window_y_linspace, color='forestgreen', alpha=0.3)

        # Create custom legend entries
        legend_handles = [
            Line2D([0], [0], color='black', linestyle='--', linewidth=1),
            Line2D([0], [0], color='black', linestyle='-', linewidth=2),
            Line2D([0], [0], color='forestgreen', linestyle='-', linewidth=0, marker='s', alpha=0.3, markersize=10),
        ]
        legend_labels = ['Ground Truth Beats', 'Predictions', 'Tolerance Window (40ms)', ]
        plt.title('Predicted Beats and Ground Truth Beats with Cemgil Tolerance Window')

        for index, x in enumerate(predicted_beats):
            if index != 0:
                lbl = None
            else:
                lbl = 'Predictions'
            plt.axvline(x=x, color='black', linestyle='-', linewidth=2, ymax=0.5, label=lbl)

        legend = plt.legend(legend_handles, legend_labels, loc='upper right')
        plt.gca().add_artist(legend)
        plt.xlabel('Time (s)')
        plt.ylabel('Weight')
        plt.title('Predicted Beats and Ground Truth Beats with Cemgil Tolerance Window')

        plt.show()
