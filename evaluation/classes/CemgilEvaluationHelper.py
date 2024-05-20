#  The implementation of the EvaluationHelper_FMeasure class adheres to several coding design principles, including:
# Single Responsibility Principle (SRP):
#
# Each method in the class has a single responsibility. For example, calculate_tp_fp_fn calculates true positives,
# false positives, and false negatives, while calculate_f_score computes the F-score based on these values. The class
# itself encapsulates functionality related to evaluating F-measure, which is a single responsibility. Encapsulation:
#
# The class encapsulates related functionality (calculating F-measure) and data (predicted beats, ground truth beats,
# tolerance). Encapsulation helps to organize code, making it more modular and easier to maintain. Separation of
# Concerns (SoC):
#
# The class separates the concerns of calculating F-measure from other parts of the code. This separation allows for
# better organization and readability of code, as each class or module can focus on a specific aspect of
# functionality. Abstraction:
#
# The class abstracts away the details of F-measure calculation, providing a clear interface (calculate_f_score) for
# clients to use. Abstraction helps in hiding implementation details, allowing clients to interact with the class
# without needing to know the internal workings. Modularity:
#
# The class promotes modularity by encapsulating related functionality within a single unit (EvaluationHelper_FMeasure).
# This modular design makes it easier to understand, test, and maintain the code.
# Ease of Testing:
#
# With the functionality encapsulated within the class, it becomes easier to write unit tests for each method
# independently. This facilitates testing and ensures that each component behaves as expected.
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from evaluation.classes.EvaluationHelperBase import EvaluationHelperBase


# Overall, following these coding design principles leads to cleaner, more maintainable, and easier-to-understand
# code, which can improve productivity and reduce the likelihood of bugs.

class CemgilEvaluationHelper(EvaluationHelperBase):
    def __init__(self, tolerance_ms=40):
        self.tolerance_ms = tolerance_ms

    @staticmethod
    def gaussian_error(x, mu=0, sigma=1):
        """
        Compute the value of the Gaussian error function for a given value of x.

        Args:
            x (float): Input value.
            mu (float): Mean of the Gaussian distribution.
            sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
            float: Value of the Gaussian error function.
        """
        return np.exp((-1.) * ((x - mu) ** 2) / (2.0 * (sigma ** 2)))

    @staticmethod
    def gaussian(x, mu=0, sigma=1.):
        return np.exp(-1 * ((x - mu) / sigma) ** 2) / (2.0 * sigma * np.sqrt(2 * np.pi))

    def calculate_metric(self, predicted_beats, ground_truth_beats):  # 40ms as in the original paper
        """
        Calculate Cemgil accuracy for beat tracking.

        Args:
            predicted_beats (list): List of predicted beat timestamps.
            ground_truth_beats (list): List of ground truth beat timestamps.

        Returns:
            float: Cemgil accuracy.
        """
        total_score = 0

        for true_beat in ground_truth_beats:
            min_error = float('inf')

            for pred_beat in predicted_beats:
                error = abs(pred_beat - true_beat)
                min_error = min(error, min_error)

            weight = self.gaussian_error(min_error, mu=0, sigma=self.tolerance_ms / 1000)
            total_score += weight

            # Dividing the accuracy by this factor ensures that the accuracy metric is scaled appropriately relative
            # to the total number of beats considered.This normalization step helps in making the Cemgil accuracy
            # metric comparable across different scenarios, datasets, or variations of beat tracking algorithms.

        cemgil_accuracy = total_score / (.5 * (len(predicted_beats) + len(ground_truth_beats))) if (len(
            predicted_beats) > 0 and len(
            ground_truth_beats) > 0) else 0.
        return round(cemgil_accuracy, 2)

    @staticmethod
    def metric_components():
        return ['cemgil']

    def plot_beats(self, predicted_beats, ground_truth_beats, n=None):
        # plt.figure(figsize=(10, 6))

        if n is not None:
            predicted_beats = predicted_beats[:n]
            ground_truth_beats = ground_truth_beats[:n]

        # # Plot predicted beats
        # plt.scatter(predicted_beats, [0.4] * len(predicted_beats), color='blue', label='Predicted Beats', zorder=2)

        plt.ylim(bottom=0)
        # plt.xlim(left=0)

        # # Plot ground truth beats
        # plt.scatter(ground_truth_beats, [0.25] * len(ground_truth_beats), color='black', label='Ground Truth Beats',
        #             zorder=2)
        # for x in zip(ground_truth_beats):
        #     plt.axvline(x=x, color='black', linestyle='--', linewidth=2)
        # Plot tolerance window
        for index, beat in enumerate(ground_truth_beats):
            if index != 0:
                lbl = None
            else:
                lbl = 'Ground truth beats'
            plt.axvline(x=beat, color='black', linestyle='--', linewidth=1, ymax=1, label=lbl)

            x_values = np.linspace(beat - 10 * self.tolerance_ms / 1000, beat + 10 * self.tolerance_ms / 1000, 1000)
            # x_values = np.linspace(beat - self.tolerance_ms / 1000, beat + self.tolerance_ms / 1000, 100)

            y_values = [self.gaussian_error(x, mu=beat, sigma=self.tolerance_ms / 1000) for x in x_values]

            # normalize_array = [y_values[0] for _ in range(len(y_values))]
            # y_values = [y - n for y, n in zip(y_values, normalize_array)]
            plt.plot(x_values, y_values, color='black', linewidth=0.5)
            plt.fill_between(x_values, y_values, color='gray', alpha=0.25)

            # plt.axvline(x=beat - self.tolerance_ms / 1000, color='black', linestyle='-', linewidth=2)
            # plt.axvline(x=beat + self.tolerance_ms / 1000, color='black', linestyle='-', linewidth=2)
            # tolerance_window_x_bounds = [beat - self.tolerance_ms / 1000, beat + self.tolerance_ms / 1000]
            # tolerance_window_y_bounds = [
            #     self.gaussian_error(beat - self.tolerance_ms / 1000, mu=beat, sigma=self.tolerance_ms / 1000),
            #     self.gaussian_error(beat + self.tolerance_ms / 1000, mu=beat, sigma=self.tolerance_ms / 1000)]

            tolerance_window_x_linspace = np.linspace(beat - self.tolerance_ms / 1000, beat + self.tolerance_ms / 1000,
                                                      100)
            tolerance_window_y_linspace = [self.gaussian_error(x, mu=beat, sigma=self.tolerance_ms / 1000) for x in
                                           tolerance_window_x_linspace]

            # Get the minimum and maximum x-values
            min_x = min(tolerance_window_x_linspace)
            max_x = beat + 6 * self.tolerance_ms / 1000

            # Set xlim based on the data range
            plt.xlim(left=0, right=max_x)
            # plt.plot([tolerance_window_x_bounds[0], tolerance_window_x_bounds[0]], [0, tolerance_window_y_bounds[0]],
            #          color='red', linestyle='--', linewidth=0.9)
            # plt.plot([tolerance_window_x_bounds[1], tolerance_window_x_bounds[1]],
            #          [0, tolerance_window_y_bounds[1]],
            #          color='red', linestyle='--', linewidth=0.9)
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
        # Create a legend object
        legend = plt.legend(legend_handles, legend_labels, loc='upper right')

        # Customize legend position
        plt.gca().add_artist(legend)

        # plt.grid(True)

        plt.xlabel('Time (s)')
        plt.ylabel('Weight')
        plt.title('Predicted Beats and Ground Truth Beats with Cemgil Tolerance Window')
        # plt.legend()
        plt.show()
