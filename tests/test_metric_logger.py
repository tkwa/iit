from iit.utils.metric import *
from iit.model_pairs import IITModelPair


def test_metric_collection():
    mc = MetricStoreCollection(
        [MetricStore("acc", MetricType.ACCURACY), MetricStore("loss", MetricType.LOSS)]
    )
    mc.create_metric_store("new_acc", MetricType.ACCURACY)
    mc.update({"acc": 0.5, "loss": 0.2, "new_acc": 0.6})
    mc.update({"acc": 0.7, "loss": 0.1, "new_acc": 0.8})
    str_arr = [str(metric) for metric in mc.metrics]
    assert str_arr == ["acc: 60.00%", "loss: 0.1500", "new_acc: 70.00%"]
    assert mc.metrics[0].get_value() == ((0.5 + 0.7) / 2) * 100
    assert mc.metrics[1].get_value() == (0.2 + 0.1) / 2
    assert mc.metrics[2].get_value() == ((0.6 + 0.8) / 2) * 100


def test_early_stop():
    mc = MetricStoreCollection(
        [MetricStore("acc", MetricType.ACCURACY), MetricStore("loss", MetricType.LOSS)]
    )
    mc.create_metric_store("new_acc", MetricType.ACCURACY)
    mc.update({"acc": 0.5, "loss": 0.2, "new_acc": 0.6})
    mc.update({"acc": 0.7, "loss": 0.1, "new_acc": 0.8})

    es_condition = IITModelPair._check_early_stop_condition(mc.metrics)
    assert es_condition == False
    mc.update({"acc": 0.991, "loss": 0.1, "new_acc": 0.99})
    es_condition = IITModelPair._check_early_stop_condition(mc.metrics)
    assert es_condition == False
    mc = MetricStoreCollection(
        [
            MetricStore("acc", MetricType.ACCURACY),
            MetricStore("loss", MetricType.LOSS),
            MetricStore("new_acc", MetricType.ACCURACY),
        ]
    )
    mc.update({"acc": 0.991, "loss": 0.1, "new_acc": 0.991})
    es_condition = IITModelPair._check_early_stop_condition(mc.metrics)
    assert es_condition == True
