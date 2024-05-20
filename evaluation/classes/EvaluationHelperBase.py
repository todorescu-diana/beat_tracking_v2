from abc import ABC, abstractmethod


class EvaluationHelperBase(ABC):
    @abstractmethod
    def calculate_metric(self, detections, annotations):
        pass

    @staticmethod
    @abstractmethod
    def metric_components():
        pass

    @abstractmethod
    def plot_beats(self, predicted_beats, ground_truth_beats):
        pass

    def calculate_mean_metric(self, detections_dict, annotations_dict):
        num_tracks = len(detections_dict)
        total_metrics = None

        # Get the metric components from the evaluation helper object
        metric_components = self.metric_components()

        for track_name in detections_dict.keys():
            detections = detections_dict[track_name]['beats']
            annotations = annotations_dict[track_name]['beats']  # Get annotations for the corresponding track name

            if annotations is None:
                continue  # Skip if annotations are missing for this track

            metric = self.calculate_metric(detections, annotations)

            if isinstance(metric, tuple):  # Check if it's a tuple
                if total_metrics is None:
                    total_metrics = {component: 0. for component in metric_components}

                for component, value in zip(metric_components, metric):
                    total_metrics[component] += value
            else:  # Otherwise, assume it's a single value
                component = metric_components[0]
                if total_metrics is None:
                    total_metrics = {component: 0. for component in metric_components}
                total_metrics[component] += metric

        if total_metrics is not None:
            # Calculate the mean
            for component in total_metrics:
                total_metrics[component] /= num_tracks
                total_metrics[component] = round(total_metrics[component], 2)
        else:
            total_metrics = {component: 0. for component in metric_components}

        return total_metrics
