import copy
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
import scipy.stats

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
        self.topk = 1

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

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

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_gradient(data_manager, per_class)
            # self._construct_exemplar_entropy(data_manager, per_class)
            # self._construct_exemplar_coreset(data_manager, per_class)
            # self._construct_exemplar_kcenter(data_manager, per_class)
            # self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass


    def _evaluate_classification_metrics(self, y_pred, y_true):
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        # auc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
        cm = confusion_matrix(y_true, y_pred)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # Sensitivity for class 1
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # Specificity for class 0
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            # "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity
        }

    def _class_wise_accuracy(self, y_pred, y_true):
        unique_classes = np.unique(y_true)
        class_accuracies = {}
        for cls in unique_classes:
            mask = y_true == cls
            acc = np.mean(y_pred[mask] == y_true[mask])
            class_accuracies[cls] = acc
        return class_accuracies

    def _forgetting_score(self, accuracy_old, accuracy_new):
        return accuracy_old - accuracy_new

    def _forward_transfer(self, accuracy_new, accuracy_previous):
        return accuracy_new - accuracy_previous

    def _backward_transfer(self, accuracy_old, accuracy_new):
        return accuracy_old - accuracy_new

    # def _evaluate(self, y_pred, y_true):
    #     ret = {}
    #     grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
    #     ret["grouped"] = grouped
    #     ret["top1"] = grouped["total"]
    #     ret["top{}".format(self.topk)] = np.around(
    #         (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
    #         decimals=2,
    #     )

    #     # New Metrics
    #     class_metrics = self._evaluate_classification_metrics(y_pred, y_true)
    #     ret.update(class_metrics)
    #     class_accuracies = self._class_wise_accuracy(y_pred, y_true)
    #     ret["class_wise_accuracy"] = class_accuracies

    #     return ret

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
    
        # New Metrics
        class_metrics = self._evaluate_classification_metrics(y_pred, y_true)
        class_accuracies = self._class_wise_accuracy(y_pred, y_true)
    
        # Example old and new accuracies for calculating forgetting, FT, and BT
        accuracy_old = 80.0
        accuracy_new = class_metrics["recall"] * 100  # Example
    
        forgetting_score = self._forgetting_score(accuracy_old, accuracy_new)
        forward_transfer = self._forward_transfer(accuracy_new, accuracy_old)
        backward_transfer = self._backward_transfer(accuracy_old, accuracy_new)
    
        ret.update(class_metrics)
        ret["class_wise_accuracy"] = class_accuracies
        ret["forgetting_score"] = forgetting_score
        ret["forward_transfer"] = forward_transfer
        ret["backward_transfer"] = backward_transfer
    
        # Log all results
        logging.info(f"Top-1 Accuracy: {ret['top1']}")
        logging.info(f"Precision: {class_metrics['precision']:.4f}, Recall: {class_metrics['recall']:.4f}, F1: {class_metrics['f1_score']:.4f}")
        logging.info(f"Sensitivity: {class_metrics['sensitivity']:.4f}, Specificity: {class_metrics['specificity']:.4f}")
        logging.info(f"Forgetting Score: {forgetting_score:.4f}, FT: {forward_transfer:.4f}, BT: {backward_transfer:.4f}")
        logging.info(f"Class-wise Accuracy: {class_accuracies}")
    
        return ret

    
    def eval_task(self, save_conf=False):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        logging.info("CNN Metrics")
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            logging.info("NME Metrics")
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        # Optionally save prediction and target for analysis
        if save_conf:
            _pred = y_pred.T[0]
            _target = y_true
            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred},{_target} \n")

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=2
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=2
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=2
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    # Herding Strategy
    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=2
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=2
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=2
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            _class_means[class_idx, :] = mean
        self._class_means = _class_means

    # K-Center strategy
    def _construct_exemplar_kcenter(self, data_manager, m):
        logging.info(f"Constructing exemplars using K-Center clustering... ({m} per class)")
    
        _class_means = np.zeros((self._total_classes, self.feature_dim))
    
        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )
    
            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=2
            )
    
            vectors, _ = self._extract_vectors(class_loader)  # Extract feature vectors
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
    
            _class_means[class_idx, :] = mean
    
        # Construct exemplars using K-Center selection for new classes
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=2
            )
    
            vectors, _ = self._extract_vectors(class_loader)  # Extract feature vectors
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    
            # Step 1: Apply KMeans clustering to find m cluster centers
            kmeans = KMeans(n_clusters=m, random_state=0).fit(vectors)
            cluster_centers = kmeans.cluster_centers_
    
            # Step 2: Select closest samples to each cluster center
            selected_exemplars = []
            selected_indices = []
            for center in cluster_centers:
                i = np.argmin(np.linalg.norm(vectors - center, axis=1))  # Find closest point
                selected_exemplars.append(np.array(data[i]))  # Store selected sample
                selected_indices.append(i)
    
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
    
            # Update memory with selected exemplars
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
    
            # Compute new exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=2
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
    
            _class_means[class_idx, :] = mean
    
        self._class_means = _class_means

    # core-set implementation
    def _construct_exemplar_coreset(self, data_manager, m):
        logging.info(f"Constructing exemplars using Core-set Selection... ({m} per class)")
    
        _class_means = np.zeros((self._total_classes, self.feature_dim))
    
        # Process old classes to compute updated means
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )
    
            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=2
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
    
            _class_means[class_idx, :] = mean
    
        # Process new classes using Core-set selection
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=2
            )
    
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    
            # Step 1: Initialize by selecting a random sample
            selected_exemplars = []
            selected_indices = [np.random.choice(len(data))]  
            selected_exemplars.append(data[selected_indices[0]])
    
            # Step 2: Iteratively select the sample farthest from current set
            for _ in range(m - 1):
                remaining_indices = list(set(range(len(data))) - set(selected_indices))
                
                # Compute distance of remaining samples to selected set
                dist_matrix = cdist(vectors[remaining_indices], vectors[selected_indices], metric="euclidean")
                min_dist = np.min(dist_matrix, axis=1)  # Find min distance to any selected exemplar
    
                # Pick the farthest point
                next_index = remaining_indices[np.argmax(min_dist)]
                selected_indices.append(next_index)
                selected_exemplars.append(data[next_index])
    
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
    
            # Update memory
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
    
            # Compute mean of exemplars
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=2
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
    
            _class_means[class_idx, :] = mean
    
        self._class_means = _class_means


    def _construct_exemplar_entropy(self, data_manager, m):
        logging.info(f"Constructing exemplars using Entropy-based Selection... ({m} per class)")
    
        _class_means = np.zeros((self._total_classes, self.feature_dim))
    
        # Compute feature representations for old classes
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )
    
            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=2)
            vectors, _ = self._extract_vectors(class_loader)  # Extract feature vectors
    
            # Normalize feature vectors
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            _class_means[class_idx, :] = mean
    
        # Compute feature representations for new classes
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=2)
    
            vectors, _ = self._extract_vectors(class_loader)  # Extract feature vectors
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    
            # Ensure we have a classifier for entropy computation
            if not hasattr(self, 'classifier'):
                self.classifier = nn.Sequential(
                    nn.Linear(self.feature_dim, 256),  # Feature input → Hidden layer
                    nn.ReLU(),
                    nn.Linear(256, self._total_classes)  # Hidden → Output (num_classes)
                ).to(self._device)
    
            # Step 1: Compute entropy for each sample
            with torch.no_grad():
                feature_tensors = torch.tensor(vectors).float().to(self._device)  # Convert to tensor
                probs = F.softmax(self.classifier(feature_tensors), dim=1)  # Compute softmax probabilities
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # Shannon entropy
    
            # Step 2: Select top-m high entropy samples
            selected_indices = torch.argsort(entropy, descending=True)[:m]  # Top-m highest entropy
            selected_exemplars = data[selected_indices.cpu().numpy()]
            exemplar_targets = np.full(m, class_idx)
    
            # Store exemplars in memory
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
    
            # Compute new class mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=2)
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            _class_means[class_idx, :] = mean
    
        self._class_means = _class_means


    # gradient-selection strategy
    def _construct_exemplar_gradient(self, data_manager, m):
        """
        Gradient-Based Selection Strategy for Exemplar Construction.
        Args:
            data_manager: Object managing data access.
            m: Number of exemplars per class.
        """
        logging.info("Constructing exemplars using Gradient-Based Selection...({} per class)".format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))
    
        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=2
            )
    
            gradient_norms = []
            all_vectors = []
            
            self._network.eval()
            for inputs, labels in class_loader:
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                self._network.zero_grad()
    
                outputs = self._network(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
    
                # Flatten and concatenate all gradients
                grad_norm = 0.0
                for param in self._network.parameters():
                    if param.grad is not None:
                        grad_norm += torch.norm(param.grad).item()
                gradient_norms.append(grad_norm)
    
                # Extract vectors and add to list
                vectors = tensor2numpy(self._network.extract_vector(inputs))
                all_vectors.append(vectors)
    
            gradient_norms = np.array(gradient_norms)
            all_vectors = np.concatenate(all_vectors, axis=0)
    
            # Select top-m exemplars with the highest gradient norms
            top_indices = np.argsort(-gradient_norms)[:m]  # Negative sign for descending order
            selected_exemplars = np.array([data[i] for i in top_indices])
            exemplar_targets = np.full(m, class_idx)
    
            # Update memory
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
    
            # Calculate mean of exemplars
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=2
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
    
            _class_means[class_idx, :] = mean
    
        self._class_means = _class_means
        logging.info("Gradient-based exemplar construction completed.")
