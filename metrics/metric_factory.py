import torch

import torchmetrics
from torchmetrics import MetricCollection

from .iou import IOUMetric
from .F1 import F1Metric

from utils import device

def create_metrics(names, conf):
    metrics = list()
    for name in names:
        if name == "IOU":
            metrics.append(create_IOU(conf[name]))
        elif name == "F1":
            metrics.append(create_F1(conf[name]))

    if "0.7" in torchmetrics.__version__:
        return MetricCollection(metrics).to(device)
    else: 
        return MetricCollection(metrics, compute_groups=False).to(device) # fix from github issue

def create_IOU(conf):
    return IOUMetric(num_classes=conf['num_classes'])

def create_F1(conf):
    return F1Metric(num_classes=conf['num_classes'])