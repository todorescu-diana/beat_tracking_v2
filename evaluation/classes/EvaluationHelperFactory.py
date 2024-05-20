from evaluation.classes.CemgilEvaluationHelper import CemgilEvaluationHelper
from evaluation.classes.ContinuityEvaluationHelper import ContinuityEvaluationHelper
from evaluation.classes.FMeasureEvaluationHelper import FMeasureEvaluationHelper


class EvaluationHelperFactory:
    @staticmethod
    def create_evaluation_helper(metric_name, **kwargs):
        if metric_name == 'f_measure':
            return FMeasureEvaluationHelper(**kwargs)
        elif metric_name == 'cemgil':
            return CemgilEvaluationHelper(**kwargs)
        elif metric_name == 'continuity':
            return ContinuityEvaluationHelper(**kwargs)
        # Add more conditions for other types of evaluation helpers
        else:
            raise ValueError("Invalid metric name")
