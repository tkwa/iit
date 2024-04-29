from iit.utils.metric import *
from iit.model_pairs import IITModelPair
from iit.model_pairs import IOI_ModelPair


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


def test_IOI_early_stop():
    per_token_accuracy = [
        1.0, # 0
        1.0, # 1
        0.005, # 2
        0.985, # 3
        0.022, # 4
        0.019, # 5
        1.0, # 6
        1.0, # 7
        0.361, # 8
        1.0, # 9
        0.084, # 10
        0.688, # 11
        1.0, # 12
        0.332, # 13
        1.0, # 14
        1.0, # 15
    ]

    IIA = 100
    accuracy = 60

    mc = IOI_ModelPair.make_test_metrics()
    mc.update(
        {
            "val/iit_loss": 0.2,
            "val/IIA": IIA,
            "val/accuracy": accuracy,
            "val/per_token_accuracy": per_token_accuracy,
        }
    )

    es_condition = IOI_ModelPair._check_early_stop_fn(mc.metrics, non_ioi_thresh=0.9)

    assert es_condition == False

    es_condition = IOI_ModelPair._check_early_stop_fn(mc.metrics, non_ioi_thresh=0.5)

    assert es_condition == True
