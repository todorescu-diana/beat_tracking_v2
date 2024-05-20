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
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from evaluation.classes.EvaluationHelperBase import EvaluationHelperBase


# Overall, following these coding design principles leads to cleaner, more maintainable, and easier-to-understand
# code, which can improve productivity and reduce the likelihood of bugs.

class FMeasureEvaluationHelper(EvaluationHelperBase):
    def __init__(self, tolerance_ms=70):
        self.tolerance_ms = tolerance_ms

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

    @staticmethod
    def metric_components():
        return ['precision', 'recall', 'f_score']

    def plot_beats(self, predicted_beats, ground_truth_beats, n=None):
        # plt.figure(figsize=(10, 6))
        if n is not None:
            predicted_beats = predicted_beats[:n]
            ground_truth_beats = ground_truth_beats[:n]

        plt.ylim(bottom=0)

        # # Plot predicted beats
        # plt.scatter(predicted_beats, [1] * len(predicted_beats), color='blue', label='Predicted Beats')
        for index, x in enumerate(predicted_beats):
            if index != 0:
                lbl = None
            else:
                lbl = 'Predictions'
            plt.axvline(x=x, color='black', linestyle='-', linewidth=2, ymax=0.5, label=lbl)

        # # Plot ground truth beats
        # plt.scatter(ground_truth_beats, [1.1] * len(ground_truth_beats), color='black', label='Ground Truth '
        #                                                                                       'Beats')
        for index, beat in enumerate(ground_truth_beats):
            if index != 0:
                lbl = None
            else:
                lbl = 'Ground truth beats'
            plt.axvline(x=beat, color='black', linestyle='--', linewidth=1, ymax=1, label=lbl)

        # Plot tolerance window
        for beat in ground_truth_beats:
            # plt.axvline(x=beat - self.tolerance_ms / 1000, color='gray', linestyle='--', linewidth=0.5)
            # plt.axvline(x=beat + self.tolerance_ms / 1000, color='gray', linestyle='--', linewidth=0.5)
            plt.fill_betweenx(y=[0, 2], x1=beat - self.tolerance_ms / 1000, x2=beat + self.tolerance_ms / 1000,
                              color='forestgreen',
                              alpha=0.3)

        # Create custom legend entries
        legend_handles = [
            Line2D([0], [0], color='black', linestyle='--', linewidth=1),
            Line2D([0], [0], color='black', linestyle='-', linewidth=2),
            Line2D([0], [0], color='forestgreen', linestyle='-', linewidth=0, marker='s', alpha=0.3, markersize=10),
        ]
        legend_labels = ['Ground Truth Beats', 'Predictions', 'Tolerance Window (70ms)', ]
        plt.title('Predicted Beats and Ground Truth Beats with Cemgil Tolerance Window')

        # Create a legend object
        legend = plt.legend(legend_handles, legend_labels, loc='upper right')

        # Customize legend position
        plt.gca().add_artist(legend)

        plt.xlabel('Time (s)')
        plt.yticks([])
        plt.title('Predicted Beats and Ground Truth Beats with Tolerance Window')
        # plt.legend()
        # plt.grid(True)
        plt.show()
