import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List

class BaseMetric:
    def __init__(self):
        self._cache = {}

    def update(self, preds, targets):
        """
        Optional update method for batch-wise computation. By default, it does nothing.
        Subclasses can override this method to implement batch-wise computation.

        Parameters
        - preds: torch.Tensor of shape (batch_size, ...)
        - targets: torch.Tensor of shape (batch_size, ...)
        """
        pass # do nothing by default for non-batch-wise metrics

    def compute_final(self):
        """
        Compute the final metric value based on the stored intermediate values.
        Must be implemented by subclasses.

        Returns
        - final_value: float
        """
        raise NotImplementedError('Subclasses must implement this method')

class ConfusionMatrixMetric(BaseMetric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.cumulative_cm = np.zeros((num_classes, num_classes))

    def update(self, preds, targets):
        batch_cm = confusion_matrix(targets, preds, labels=range(self.num_classes)) # what if preds have no positive class?
        self.cumulative_cm += batch_cm
    
    def compute_final(self):
        self._cache['confusion_matrix'] = self.cumulative_cm
        return self.cumulative_cm.ravel()
    
class PrecisionMetric(BaseMetric):
    def compute_final(self):
        if 'confusion_matrix' not in self._cache:
            raise ValueError('Confusion matrix not found in _cache. Make sure to compute ConfusionMatrixMetric first.')

        cm = self._cache['confusion_matrix']
        precision = self._compute_precision(cm)

        self._cache['precision'] = precision

        return precision
    
    def _compute_precision(self, cm):
        if cm.shape[0] == 2:
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp)
        else:
            tp = np.diag(cm)
            fp = cm.sum(axis=0) - tp
            precision = tp / (tp + fp) 
        
        # fix nans
        precision = np.nan_to_num(precision)
        return np.mean(precision)

class RecallMetric(BaseMetric):
    def compute_final(self):
        if 'confusion_matrix' not in self._cache:
            raise ValueError('Confusion matrix not found in _cache. Make sure to compute ConfusionMatrixMetric first.')

        cm = self._cache['confusion_matrix']
        recall = self._compute_recall(cm)
        self._cache['recall'] = recall

        return recall
    
    def _compute_recall(self, cm):
        if cm.shape[0] == 2:
            tn, fp, fn, tp = cm.ravel()
            recall = tp / (tp + fn)
        else:
            tp = np.diag(cm)
            fn = cm.sum(axis=1) - tp
            recall = tp / (tp + fn) 
        
        # fix nans
        recall = np.nan_to_num(recall)
        return np.mean(recall)

class F1ScoreMetric(BaseMetric):
    def compute_final(self):
        if 'precision' not in self._cache:
            raise ValueError('Precision not found in _cache. Make sure to compute PrecisionMetric first.')
        if 'recall' not in self._cache:
            raise ValueError('Recall not found in _cache. Make sure to compute RecallMetric first.')

        precision = self._cache['precision']
        recall = self._cache['recall']
        f1_score = self._compute_f1_score(precision, recall)

        self._cache['f1_score'] = f1_score

        return f1_score
    
    def _compute_f1_score(self, precision, recall):
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

class AccuracyMetric(BaseMetric):
    def compute_final(self):
        if 'confusion_matrix' not in self._cache:
            raise ValueError('Confusion matrix not found in _cache. Make sure to compute ConfusionMatrixMetric first.')

        cm = self._cache['confusion_matrix']
        accuracy = self._compute_accuracy(cm)

        self._cache['accuracy'] = accuracy

        return accuracy
    
    def _compute_accuracy(self, cm):
        if cm.shape[0] == 2:
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
        else:
            tp = np.diag(cm)
            accuracy = tp.sum() / cm.sum()
        
        accuracy = np.nan_to_num(accuracy)
        return accuracy

class AUCMetric(BaseMetric):
    def compute_final(self):
        raise NotImplementedError('AUCMetric is not implemented yet.')

class IOUMetric(BaseMetric):
    def compute_final(self):
        if 'confusion_matrix' not in self._cache:
            raise ValueError('Confusion matrix not found in _cache. Make sure to compute ConfusionMatrixMetric first.')
        
        cm = self._cache['confusion_matrix']
        iou = self._compute_iou(cm)

        self._cache['iou'] = iou

        return iou
    
    def _compute_iou(self, cm):
        if cm.shape[0] == 2:
            _, fp, fn, tp = cm.ravel()
        else:
            tp = np.diag(cm)
            fn = cm.sum(axis=1) - tp
            fp = cm.sum(axis=0) - tp

        iou = tp / (tp + fp + fn)
        iou = np.nan_to_num(iou)
        return np.mean(iou)
    

class MetricsPipeline:
    def __init__(self, metrics: List[BaseMetric]):
        self.metrics = metrics

        # share cache
        self._cache = {}
        for metric in self.metrics:
            metric._cache = self._cache

    def reset(self):
        """
        Reset the intermediate values stored in the cache. 

        This method should be called at the beginning of each epoch.
        """
        self._cache = {}
        for metric in self.metrics:
            metric._cache = self._cache
    
    def update(self, preds, targets):
        """
        Update the intermediate values of all metrics based on the current batch.

        Parameters
        - preds: torch.Tensor of shape (batch_size, ...)
        - targets: torch.Tensor of shape (batch_size, ...)
        """
        # move to cpu
        preds = preds.cpu().view(-1).numpy()
        targets = targets.cpu().view(-1).numpy()
        
        for metric in self.metrics:
            metric.update(preds, targets)

    def compute_final(self):
        """
        Compute the final metric values based on the stored intermediate values.
        """
        final_values = {}
        for metric in self.metrics:
            final_values[metric.__class__.__name__] = metric.compute_final()
        return final_values
    