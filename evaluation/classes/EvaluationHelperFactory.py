from evaluation.classes.CemgilEvaluationHelper import CemgilEvaluationHelper
from evaluation.classes.ContinuityEvaluationHelper import ContinuityEvaluationHelper
from evaluation.classes.FMeasureEvaluationHelper import FMeasureEvaluationHelper


class EvaluationHelperFactory:
    _registry = {}

    @classmethod
    def register_helper(cls, metric_name, helper_class):
        cls._registry[metric_name] = helper_class

    @classmethod
    def create_evaluation_helper(cls, metric_name, **kwargs):
        if metric_name not in cls._registry:
            raise ValueError(f"Invalid metric name: {metric_name}")
        helper_class = cls._registry[metric_name]
        return helper_class(**kwargs)

EvaluationHelperFactory.register_helper('f_measure', FMeasureEvaluationHelper)
EvaluationHelperFactory.register_helper('cemgil', CemgilEvaluationHelper)
EvaluationHelperFactory.register_helper('continuity', ContinuityEvaluationHelper)