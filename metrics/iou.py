import os
import torchmetrics 
import torch
from utils import save_graph

class IOUMetric(torchmetrics.Metric):
    def __init__(self, num_classes):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        if "0.7" in torchmetrics.__version__:
            self.iou = torchmetrics.IoU(num_classes=num_classes,
                                        reduction="elementwise_mean",
                                        compute_on_step=False,
                                        dist_sync_on_step=False)
        else:
            self.iou = torchmetrics.JaccardIndex(num_classes=num_classes,
                                                 reduction="elementwise_mean",
                                                 compute_on_step=False,
                                                 dist_sync_on_step=False)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        return self.iou(preds, targets)

    def compute(self):
        return self.iou.compute().item()

    def reset(self):
        self.iou.reset()

    def to_string(self, results):
        return "{:.5f}".format(results)

    def save_graph(self, out_dir, model_name, epoch, train_records, val_records, test_records, test_sets_names):
        save_graph(os.path.join(out_dir, model_name+"_IOUMetric.png"), epoch, train_records, val_records, test_records, test_sets_names, label="IOUMetric")

    def save_if_best(self, cur, best, check_point, path):
        if best is None or cur >= best:
            path = path[:-4] + "_best-IOUMetric.pth"
            torch.save(check_point, path)
            return cur
        return best
