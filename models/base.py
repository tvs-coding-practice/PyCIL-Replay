import copy
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
import scipy.stats

import os

EPSILON = 1e-8
batch_size = 32


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

        # Track accuracies across tasks for continual learning metrics
        # R[i][j] = accuracy on task i after training on task j
        # where i, j are task indices (0-indexed)
        self._accuracy_matrix = []  # List of lists: R[i][j]
        self._baseline_accuracies = []  # Baseline accuracy for each task (random init)
        self._task_test_loaders = {}  # Store test loaders for each task: {task_id: loader}
        self._data_manager = None  # Store data_manager reference for automatic metrics
        self._enable_continual_learning_metrics = args.get("enable_cl_metrics", True)  # Enable by default

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
            # self._construct_exemplar_entropy(data_manager, per_class)
            # self._construct_exemplar_coreset(data_manager, per_class)
            # self._construct_exemplar_kcenter(data_manager, per_class)
            self._construct_exemplar_unified(data_manager, per_class)
            # self._construct_exemplar_random(data_manager, per_class)

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
        """
        Called after training on each task.
        Automatically computes continual learning metrics if enabled.
        """
        if not self._enable_continual_learning_metrics or self._data_manager is None:
            return

        # Step 1: Evaluate on all tasks and update accuracy matrix
        self._auto_evaluate_all_tasks()

        # Step 2: Compute baseline for NEXT task (if there is one)
        if self._cur_task + 1 < self._data_manager.nb_tasks:
            self._auto_compute_next_task_baseline()

        pass


    def _evaluate_classification_metrics(self, y_pred, y_true):
        """
        Compute classification metrics with multiple averaging methods.

        Args:
            y_pred: 1D array of predicted class labels
            y_true: 1D array of true class labels

        Returns:
            dict: Dictionary containing precision, recall, and f1-score for:
                  - weighted: Accounts for class imbalance by weighting by support
                  - macro: Unweighted mean (treats all classes equally)
                  - micro: Global average (aggregates contributions of all classes)
        """
        # Weighted averaging (accounts for class imbalance)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Macro averaging (unweighted mean - treats all classes equally)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Micro averaging (global average - aggregates contributions)
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        logging.info(f"Confusion Matrix:\n{cm}")

        return {
            # Weighted metrics (default for imbalanced datasets)
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_score_weighted": f1_weighted,

            # Macro metrics (treats all classes equally)
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_score_macro": f1_macro,

            # Micro metrics (global average)
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_score_micro": f1_micro,

            # Backward compatibility (weighted is default)
            "precision": precision_weighted,
            "recall": recall_weighted,
            "f1_score": f1_weighted,
        }

    def _class_wise_accuracy(self, y_pred, y_true):
        """
        Compute per-class accuracy.
        Args:
            y_pred: 1D array of predicted class labels
            y_true: 1D array of true class labels
        """
        unique_classes = np.unique(y_true)
        class_accuracies = {}
        for cls in unique_classes:
            mask = y_true == cls
            acc = np.mean(y_pred[mask] == y_true[mask])
            class_accuracies[int(cls)] = float(acc)
        return class_accuracies

    def _get_task_class_range(self, task_id):
        """
        Get the class range for a specific task.

        Args:
            task_id: Task index (0-based)

        Returns:
            tuple: (start_class, end_class)
        """
        if task_id == 0:
            start_class = 0
            end_class = self._data_manager.get_task_size(0)
        else:
            start_class = sum([self._data_manager.get_task_size(t) for t in range(task_id)])
            end_class = start_class + self._data_manager.get_task_size(task_id)

        return start_class, end_class

    def _create_task_test_loader(self, task_id):
        """
        Create a test loader for a specific task.

        Args:
            task_id: Task index (0-based)

        Returns:
            DataLoader for the task's test data
        """
        from torch.utils.data import DataLoader

        start_class, end_class = self._get_task_class_range(task_id)

        # Create test dataset for this task's classes
        test_dataset = self._data_manager.get_dataset(
            np.arange(start_class, end_class),
            source="test",
            mode="test"
        )

        # Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4
        )

        return test_loader

    def _create_task_test_loaders(self, data_manager):
        """
        Create test loaders for all tasks seen so far.
        Each loader contains only the classes for that specific task.

        Args:
            data_manager: DataManager instance

        Returns:
            dict: {task_id: test_loader} for all tasks from 0 to _cur_task
        """
        task_loaders = {}

        for task_id in range(self._cur_task + 1):
            task_loaders[task_id] = self._create_task_test_loader(task_id)

        return task_loaders

    def _auto_compute_next_task_baseline(self):
        """
        Automatically compute baseline for the NEXT task.
        Called in after_task() to prepare for the next incremental training.

        This computes zero-shot accuracy on the next task's classes
        before any training on that task.
        """
        next_task_id = self._cur_task + 1

        if next_task_id >= self._data_manager.nb_tasks:
            return  # No more tasks

        # Get class range for next task
        start_class, end_class = self._get_task_class_range(next_task_id)

        # Create test dataset for next task's classes
        test_dataset = self._data_manager.get_dataset(
            np.arange(start_class, end_class),
            source="test",
            mode="test"
        )

        # Create test loader
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4
        )

        # Evaluate on next task (zero-shot)
        baseline_acc = self._compute_accuracy(self._network, test_loader)

        # Store baseline
        self._baseline_accuracies.append(baseline_acc)

        logging.info(f"\n{'='*60}")
        logging.info(f"Baseline for Task {next_task_id} (zero-shot): {baseline_acc:.2f}%")
        logging.info(f"{'='*60}\n")

        return baseline_acc

    def _compute_and_store_baseline(self, data_manager):
        """
        Manually compute and store baseline accuracy for the current task.
        This evaluates the model on the current task BEFORE training on it (zero-shot).

        Note: This is now optional. The system automatically computes baselines
        in after_task() for the next task.

        Args:
            data_manager: DataManager instance to get test dataset for current task
        """
        # Store data_manager reference
        self._data_manager = data_manager

        # Create test dataset for ONLY the new classes in current task
        test_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="test",
            mode="test"
        )

        # Create temporary test loader
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=128,  # Use a reasonable batch size
            shuffle=False,
            num_workers=4
        )

        # Evaluate on the new task before training
        if self._cur_task == 0:
            # First task: model is randomly initialized
            baseline_acc = self._compute_accuracy(self._network, test_loader)
            logging.info(f"Task {self._cur_task} baseline (random init): {baseline_acc:.2f}%")
        else:
            # Subsequent tasks: zero-shot evaluation with current model
            baseline_acc = self._compute_accuracy(self._network, test_loader)
            logging.info(f"Task {self._cur_task} baseline (zero-shot): {baseline_acc:.2f}%")

        # Store baseline
        self._baseline_accuracies.append(baseline_acc)

        # Also store the test loader for this task for later evaluation
        self._task_test_loaders[self._cur_task] = test_loader

        return baseline_acc

    def _store_baseline_accuracy(self, accuracy):
        """
        Manually store baseline accuracy for the current task.

        Note: You typically don't need to call this directly.
        Use _compute_and_store_baseline() instead for automatic computation.

        Args:
            accuracy: Baseline accuracy on the current task
        """
        self._baseline_accuracies.append(accuracy)
        logging.info(f"Stored baseline accuracy for task {self._cur_task}: {accuracy:.2f}%")

    def _update_accuracy_matrix(self, task_accuracies):
        """
        Update the accuracy matrix with accuracies on all tasks after current training.

        Args:
            task_accuracies: Dict mapping task_id to accuracy on that task
                            e.g., {0: 95.0, 1: 87.5} means 95% on task 0, 87.5% on task 1
        """
        current_task_idx = self._cur_task

        # Fill in accuracies for all tasks evaluated so far
        for task_id, accuracy in task_accuracies.items():
            task_idx = task_id

            # Ensure we have enough rows in the matrix
            while len(self._accuracy_matrix) <= task_idx:
                self._accuracy_matrix.append([])

            # Ensure the row for this task has enough columns
            while len(self._accuracy_matrix[task_idx]) <= current_task_idx:
                self._accuracy_matrix[task_idx].append(0.0)

            # Store R[task_idx][current_task_idx] = accuracy on task_idx after training on current_task
            self._accuracy_matrix[task_idx][current_task_idx] = accuracy

        logging.info(f"Updated accuracy matrix after task {current_task_idx}")
        self._log_accuracy_matrix()

    def _log_accuracy_matrix(self):
        """
        Log the current state of the accuracy matrix for debugging.
        """
        if len(self._accuracy_matrix) == 0:
            return

        logging.info("Accuracy Matrix R[i][j] (task i after training on task j):")
        header = "Task |" + "".join([f" T{j:2d} |" for j in range(len(self._accuracy_matrix[0]))])
        logging.info(header)
        logging.info("-" * len(header))

        for i, row in enumerate(self._accuracy_matrix):
            row_str = f"  {i:2d} |" + "".join([f" {acc:5.2f}|" for acc in row])
            logging.info(row_str)

    def _compute_continual_learning_metrics(self):
        """
        Compute continual learning metrics according to Lopez-Paz & Ranzato (2017):
        "Gradient Episodic Memory for Continual Learning"

        Metrics computed:
        - Backward Transfer (BWT): Average influence of learning new tasks on old tasks
        - Forward Transfer (FWT): Average influence of learning old tasks on new tasks
        - Forgetting (F): Average maximum drop in accuracy on old tasks

        Returns:
            dict with backward_transfer, forward_transfer, forgetting (or None if not applicable)
        """
        metrics = {
            "backward_transfer": None,
            "forward_transfer": None,
            "forgetting": None,
            "average_accuracy": None
        }

        T = len(self._accuracy_matrix)  # Number of tasks learned so far

        if T == 0:
            return metrics

        # Compute Average Accuracy
        # ACC = (1/T) * Σ(i=1 to T) R_T,i
        avg_acc = np.mean([self._accuracy_matrix[i][T-1] for i in range(T)])
        metrics["average_accuracy"] = avg_acc

        if T == 1:
            # Only one task, no transfer metrics yet
            return metrics

        # Backward Transfer (BWT)
        # BWT = (1/(T-1)) * Σ(i=1 to T-1) [R_T,i - R_i,i]
        # Measures how learning new tasks affects performance on old tasks
        bwt_sum = 0.0
        for i in range(T - 1):
            R_T_i = self._accuracy_matrix[i][T - 1]  # Accuracy on task i after learning all T tasks
            R_i_i = self._accuracy_matrix[i][i]      # Accuracy on task i right after learning it
            bwt_sum += (R_T_i - R_i_i)
        metrics["backward_transfer"] = bwt_sum / (T - 1)

        # Forward Transfer (FWT)
        # FWT = (1/(T-1)) * Σ(i=2 to T) [R_i-1,i - b_i]
        # Measures how learning previous tasks helps with new tasks
        if len(self._baseline_accuracies) >= T:
            fwt_sum = 0.0
            for i in range(1, T):
                R_prev_i = self._accuracy_matrix[i][i - 1]  # Accuracy on task i before training on it
                b_i = self._baseline_accuracies[i]           # Baseline accuracy on task i
                fwt_sum += (R_prev_i - b_i)
            metrics["forward_transfer"] = fwt_sum / (T - 1)

        # Forgetting (F)
        # F = (1/(T-1)) * Σ(i=0 to T-2) max_j∈{i,...,T-1} [R[i][j] - R[i][T-1]]
        # Measures the maximum drop in accuracy on each old task
        forgetting_sum = 0.0
        for i in range(T - 1):
            # Find maximum accuracy on task i across all training stages from i to T-1
            max_acc = max([self._accuracy_matrix[i][j] for j in range(i, T)])
            current_acc = self._accuracy_matrix[i][T - 1]
            forgetting_sum += max(0, max_acc - current_acc)  # Only count drops, not improvements
        metrics["forgetting"] = forgetting_sum / (T - 1)

        return metrics

    def _evaluate(self, y_pred, y_true):
        """
        Evaluate predictions and compute metrics.
        Args:
            y_pred: 2D array of shape [N, topk] containing top-k predictions
            y_true: 1D array of shape [N] containing true labels
        """
        ret = {}

        # Extract top-1 predictions for classification metrics
        y_pred_top1 = y_pred.T[0]  # Shape: [N]

        # Compute grouped accuracy (old/new classes)
        grouped = accuracy(y_pred_top1, y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]

        # Compute top-k accuracy
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        # Compute classification metrics using top-1 predictions
        class_metrics = self._evaluate_classification_metrics(y_pred_top1, y_true)
        class_accuracies = self._class_wise_accuracy(y_pred_top1, y_true)

        ret.update(class_metrics)
        ret["class_wise_accuracy"] = class_accuracies

        # Log all results
        logging.info(f"Top-1 Accuracy: {ret['top1']:.2f}%")
        logging.info(f"Top-{self.topk} Accuracy: {ret['top{}'.format(self.topk)]:.2f}%")

        # Log classification metrics with all averaging methods
        logging.info(f"\nClassification Metrics:")
        logging.info(f"  Weighted - Precision: {class_metrics['precision_weighted']:.4f}, Recall: {class_metrics['recall_weighted']:.4f}, F1: {class_metrics['f1_score_weighted']:.4f}")
        logging.info(f"  Macro    - Precision: {class_metrics['precision_macro']:.4f}, Recall: {class_metrics['recall_macro']:.4f}, F1: {class_metrics['f1_score_macro']:.4f}")
        logging.info(f"  Micro    - Precision: {class_metrics['precision_micro']:.4f}, Recall: {class_metrics['recall_micro']:.4f}, F1: {class_metrics['f1_score_micro']:.4f}")

        logging.info(f"\nClass-wise Accuracy: {class_accuracies}")

        return ret

    def _auto_evaluate_all_tasks(self):
        """
        Automatically evaluate on all tasks and update accuracy matrix.
        Called in after_task() to compute continual learning metrics.

        Creates test loaders on-the-fly for all tasks seen so far.
        """
        if self._data_manager is None:
            logging.warning("DataManager not available. Skipping continual learning metrics.")
            return None

        # Compute baseline for Task 0 if this is the first call (after Task 0)
        if self._cur_task == 0 and len(self._baseline_accuracies) == 0:
            # For Task 0, we use theoretical random baseline since we can't evaluate
            # before the FC layer is initialized
            task_0_size = self._data_manager.get_task_size(0)
            random_baseline = 100.0 / task_0_size  # Random guess accuracy
            self._baseline_accuracies.append(random_baseline)
            logging.info(f"Task 0 baseline (theoretical random): {random_baseline:.2f}%")

        task_accuracies = {}

        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating on all tasks after training task {self._cur_task}")
        logging.info(f"{'='*60}")

        # Evaluate on each task from 0 to current
        for task_id in range(self._cur_task + 1):
            # Create test loader for this task
            test_loader = self._create_task_test_loader(task_id)

            # Evaluate
            y_pred, y_true = self._eval_cnn(test_loader)
            y_pred_top1 = y_pred.T[0]

            # Compute accuracy for this task
            task_acc = np.mean(y_pred_top1 == y_true) * 100
            task_accuracies[task_id] = task_acc

            logging.info(f"Task {task_id} Accuracy: {task_acc:.2f}%")

        # Update the accuracy matrix
        self._update_accuracy_matrix(task_accuracies)

        # Compute continual learning metrics
        cl_metrics = self._compute_continual_learning_metrics()

        # Log continual learning metrics
        logging.info(f"\n{'='*60}")
        logging.info("Continual Learning Metrics:")
        logging.info(f"{'='*60}")

        if cl_metrics["average_accuracy"] is not None:
            logging.info(f"Average Accuracy: {cl_metrics['average_accuracy']:.2f}%")

        if cl_metrics["backward_transfer"] is not None:
            logging.info(f"Backward Transfer (BWT): {cl_metrics['backward_transfer']:.4f}")
            if cl_metrics["backward_transfer"] > 0:
                logging.info("  → Positive: Learning new tasks helped old tasks")
            else:
                logging.info("  → Negative: Catastrophic forgetting occurred")

        if cl_metrics["forward_transfer"] is not None:
            logging.info(f"Forward Transfer (FWT): {cl_metrics['forward_transfer']:.4f}")
            if cl_metrics["forward_transfer"] > 0:
                logging.info("  → Positive: Previous learning helped new tasks")
            else:
                logging.info("  → Negative: Previous learning hurt new tasks")

        if cl_metrics["forgetting"] is not None:
            logging.info(f"Forgetting (F): {cl_metrics['forgetting']:.4f}")
            logging.info(f"  → Average maximum drop in accuracy on old tasks")

        logging.info(f"{'='*60}\n")

        return cl_metrics

    def _evaluate_all_tasks(self, test_loader_dict=None):
        """
        Evaluate on all tasks seen so far and update the accuracy matrix.
        This should be called after training on each task.

        Args:
            test_loader_dict: Optional dictionary mapping task_id to test DataLoader.
                             If None, uses self._task_test_loaders (automatically created)
                             e.g., {0: loader_task0, 1: loader_task1, ...}

        Returns:
            dict with continual learning metrics
        """
        # Use stored test loaders if not provided
        if test_loader_dict is None:
            test_loader_dict = self._task_test_loaders

        if len(test_loader_dict) == 0:
            logging.warning("No test loaders available for evaluation. Skipping continual learning metrics.")
            return {
                "backward_transfer": None,
                "forward_transfer": None,
                "forgetting": None,
                "average_accuracy": None
            }

        task_accuracies = {}

        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating on all tasks after training task {self._cur_task}")
        logging.info(f"{'='*60}")

        # Evaluate on each task
        for task_id, test_loader in test_loader_dict.items():
            y_pred, y_true = self._eval_cnn(test_loader)
            y_pred_top1 = y_pred.T[0]

            # Compute accuracy for this task
            task_acc = np.mean(y_pred_top1 == y_true) * 100
            task_accuracies[task_id] = task_acc

            logging.info(f"Task {task_id} Accuracy: {task_acc:.2f}%")

        # Update the accuracy matrix
        self._update_accuracy_matrix(task_accuracies)

        # Compute continual learning metrics
        cl_metrics = self._compute_continual_learning_metrics()

        # Log continual learning metrics
        logging.info(f"\n{'='*60}")
        logging.info("Continual Learning Metrics:")
        logging.info(f"{'='*60}")

        if cl_metrics["average_accuracy"] is not None:
            logging.info(f"Average Accuracy: {cl_metrics['average_accuracy']:.2f}%")

        if cl_metrics["backward_transfer"] is not None:
            logging.info(f"Backward Transfer (BWT): {cl_metrics['backward_transfer']:.4f}")
            if cl_metrics["backward_transfer"] > 0:
                logging.info("  → Positive: Learning new tasks helped old tasks")
            else:
                logging.info("  → Negative: Catastrophic forgetting occurred")

        if cl_metrics["forward_transfer"] is not None:
            logging.info(f"Forward Transfer (FWT): {cl_metrics['forward_transfer']:.4f}")
            if cl_metrics["forward_transfer"] > 0:
                logging.info("  → Positive: Previous learning helped new tasks")
            else:
                logging.info("  → Negative: Previous learning hurt new tasks")

        if cl_metrics["forgetting"] is not None:
            logging.info(f"Forgetting (F): {cl_metrics['forgetting']:.4f}")
            logging.info(f"  → Average maximum drop in accuracy on old tasks")

        logging.info(f"{'='*60}\n")

        return cl_metrics

    
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

    def incremental_train(self, data_manager):
        """
        Base incremental_train method.
        Subclasses should override this but call super().incremental_train(data_manager)
        to enable automatic continual learning metrics.

        Args:
            data_manager: DataManager instance
        """
        # Store data_manager reference for automatic metrics computation
        self._data_manager = data_manager

        # Note: Baseline computation for Task 0 is now handled in _auto_evaluate_all_tasks()
        # during the first after_task() call, after the FC layer has been initialized.
        # This avoids the issue of trying to evaluate before update_fc() is called.

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


    #random selection strategy
    def _construct_exemplar_random(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )

        # For new classes, simply select random samples
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, _ = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )

            # Random Selection - Just pick m random samples
            num_samples = len(data)
            if num_samples <= m:
                selected_indices = list(range(num_samples))
            else:
                np.random.seed(215)
                selected_indices = np.random.choice(num_samples, size=m, replace=False)

            selected_exemplars = np.array([data[i] for i in selected_indices])
            exemplar_targets = np.full(len(selected_exemplars), class_idx)

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
