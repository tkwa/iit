from enum import Enum
import numpy as np

class MetricType(Enum):
    ACCURACY = 1
    LOSS = 2


class MetricStore:
    def __init__(self, name, metric_type: MetricType):
        self._name = name
        self.type = metric_type
        self._store = []
        assert self.type in MetricType, f"Invalid metric type {self.type}"

    def append(self, metric):
        self._store.append(metric)

    def get_value(self):
        if len(self._store) == 0:
            raise ValueError("No values in metric store!")
        if self.type == MetricType.ACCURACY:
            return np.mean(self._store) * 100
        else:
            return np.mean(self._store)
    
    def get_name(self):
        return self._name

    def __str__(self) -> str:
        if self.type == MetricType.ACCURACY:
            return f"{self._name}: {float(self.get_value()):.2f}%"
        return f"{self._name}: {self.get_value():.4f}"
    
    def __len__(self):
        return len(self._store)


class MetricStoreCollection:
    def __init__(self, list_of_metric_stores: list[MetricStore]):
        self.metrics = list_of_metric_stores
    
    def update(self, metrics: dict[str, float]):
        for k, v in metrics.items():
            key_found = False
            for metric in self.metrics:
                if metric.get_name() == k:
                    metric.append(v)
                    key_found = True
                    break
            assert key_found, f"Key {k} not found in metric stores!"
        l = [len(self.metrics[i]) for i in range(len(self.metrics))]
        assert np.unique(l).shape[0] == 1, f"All metric stores should have the same length after update!, got lengths: {l}"
    
    def create_metric_store(self, name, metric_type: MetricType):
        assert [len(metric._store) == 0 for metric in self.metrics], "All metric stores should be empty before creating a new one!"
        new_metric = MetricStore(name, metric_type)
        self.metrics.append(new_metric)
        return new_metric
