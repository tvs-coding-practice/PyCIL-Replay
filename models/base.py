import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, accuracy_score
)
from scipy.spatial.distance import cdist
import os

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self._previous_accuracies = []  # For backward transfer and forgetting score

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def _evaluate(self, y_pred, y_true):
        ret = {}
        ret["top1"] = accuracy_score(y_true, y_pred[:, 0]) * 100
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        # ➡️ Class-wise accuracy
        cm = confusion_matrix(y_true, y_pred[:, 0], labels=np.arange(self._total_classes))
        class_wise_acc = cm.diagonal() / (cm.sum(axis=1) + EPSILON) * 100
        ret["class_wise_acc"] = dict(enumerate(class_wise_acc))

        # ➡️ Classification metrics
        ret["precision"] = precision_score(y_true, y_pred[:, 0], average="macro", zero_division=0) * 100
        ret["recall"] = recall_score(y_true, y_pred[:, 0], average="macro", zero_division=0) * 100
        ret["f1"] = f1_score(y_true, y_pred[:, 0], average="macro", zero_division=0) * 100
        ret["auc"] = roc_auc_score(
            np.eye(self._total_classes)[y_true], 
            np.eye(self._total_classes)[y_pred[:, 0]],
            multi_class='ovr'
        )

        return ret

    def eval_task(self, save_conf=False):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        # ➡️ Forgetting Score Calculation
        if self._cur_task > 0:
            forgetting_scores = []
            for i in range(self._cur_task):
                forgetting = self._previous_accuracies[i] - cnn_accy["top1"]
                forgetting_scores.append(forgetting)
            cnn_accy["forgetting_score"] = np.mean(forgetting_scores)

        # ➡️ Backward and Forward Transfer Calculation
        if self._cur_task > 0:
            previous_avg = np.mean(self._previous_accuracies)
            backward_transfer = cnn_accy["top1"] - previous_avg
            cnn_accy["backward_transfer"] = backward_transfer

        if self._cur_task > 0:
            forward_transfer = cnn_accy["top1"] - self._previous_accuracies[-1]
            cnn_accy["forward_transfer"] = forward_transfer

        # ➡️ Save the current accuracy for future use
        self._previous_accuracies.append(cnn_accy["top1"])

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

        return cnn_accy, nme_accy

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(correct * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
