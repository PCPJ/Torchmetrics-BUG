import os
import torchmetrics 
import torch
from utils import save_graph

class F1Metric(torchmetrics.Metric):
    def __init__(self, num_classes):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        if "0.7" in torchmetrics.__version__:
            self.f1 = torchmetrics.F1(num_classes=num_classes,
                                        average="micro",
                                        mdmc_average="samplewise",
                                        compute_on_step=False,
                                        dist_sync_on_step=False)
        else:
            self.f1 = torchmetrics.F1Score(num_classes=num_classes,
                                            average="micro",
                                            mdmc_average="samplewise",
                                            compute_on_step=False,
                                            dist_sync_on_step=False)
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        return self.f1(preds, targets)

    def compute(self):
        return self.f1.compute().item()

    def reset(self):
        self.f1.reset()

    def to_string(self, results):
        return "{:.5f}".format(results)

    def save_graph(self, out_dir, model_name, epoch, train_records, val_records, test_records, test_sets_names):
        save_graph(os.path.join(out_dir, model_name+"_F1Metric.png"), epoch, train_records, val_records, test_records, test_sets_names,label="F1Metric")

    def save_if_best(self, cur, best, check_point, path):
        if best is None or cur >= best:
            path = path[:-4] + "_best-F1Metric.pth"
            torch.save(check_point, path)
            return cur
        return best
