from torchmetrics import Metric
import torch
import numpy as np

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes: int = 200):
        super().__init__()
        # Initialize true positive, false positive, false negative for each class
        self.true_positive = np.zeros(num_classes)
        self.false_positive = np.zeros(num_classes)
        self.false_negative = np.zeros(num_classes)
        self.num_classes = num_classes
    
    def update(self, preds, target, num_classes: int = 200):
        # The preds (B x C tensor), so take argmax to get index with highest confidence
        rep_prediction = torch.argmax(preds, dim=1)

        # check if preds and target have equal shape
        if rep_prediction.shape != target.shape:
            raise ValueError("preds and target must have the same shape")     

        # Count true positive, false positive, false negative
        self.num_classes = num_classes
        for pred_idx in range(rep_prediction.shape[0]):
            if(rep_prediction[pred_idx] == target[pred_idx]):
                self.true_positive[target[pred_idx]] += 1
            else: # False positive and false negative
                self.false_positive[rep_prediction[pred_idx]] += 1
                self.false_negative[target[pred_idx]] += 1
    
    def compute(self):
        # Calculate the average F1 score
        f1_score = {}
        for class_idx in range(self.num_classes):
            precision = 0
            if (self.true_positive[class_idx] + self.false_positive[class_idx]) != 0:
                precision = self.true_positive[class_idx] / (self.true_positive[class_idx] + self.false_positive[class_idx])
            recall = 0
            if (self.true_positive[class_idx] + self.false_negative[class_idx]) != 0:
                recall = self.true_positive[class_idx] / (self.true_positive[class_idx] + self.false_negative[class_idx])
            # Check dividing zero
            if (precision + recall) == 0:
                f1_score[class_idx] = 0
            else:
                f1_score[class_idx] = 2 * precision * recall / (precision + recall)

        #  Return F1 score for each class
        return sum(f1_score.values()) / len(f1_score)

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        rep_prediction = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if rep_prediction.shape != target.shape:
            raise ValueError("preds and target must have the same shape")     

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(rep_prediction == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        if(self.total.float() == 0):
            return torch.tensor(0)
        return self.correct.float() / self.total.float()
