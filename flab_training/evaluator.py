from tqdm import tqdm
import time
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, balanced_accuracy_score,
    matthews_corrcoef, f1_score, fbeta_score
)
from flab_training.utils import format_dict

class Evaluator:
    def __init__(self, args, batcher):
        self.args = args
        self.batcher = batcher
        self.splits = batcher.splits
        self.io = {}
        
    def prepare_batches(self, split, cache=False, rep=1):
        """ Prepares batches for evaluation. If cache=True, stores batches in self.io[split] to avoid recomputation. """
        
        if cache and split in self.io:
            return self.io[split]

        eval_ind = self.splits[split]
        batches = []
        num_samples = len(eval_ind)
        for start in range(0, num_samples, self.args.eval_batch_size):
            batch_ind = eval_ind[start:min(num_samples, start + self.args.eval_batch_size)]
            #batches.append(self.batcher.get_batch(batch_ind))
            # data augmentation for pretraining
            for _ in range(rep):
                batches.append(self.batcher.get_batch(batch_ind))
        if cache:
            self.io[split] = batches
        return batches

    def evaluate(self, model, split, train_step):
        raise NotImplementedError("Subclasses should implement this!")


class EvaluatorTrain(Evaluator):
    def __init__(self, args, batcher):
        super().__init__(args, batcher)
        self.thresh = 0.5

    def evaluate(self, model, split, train_step):
        self.args.logger.write(f"\nEvaluating on split = {split}")
        batches = self.prepare_batches(split, cache=True, rep=1) # these can be cached because deterministic
        model.eval()

        true, pred = [], []
        cum_loss, num_batches, total_time = 0.0, 0, 0.0

        for batch in tqdm(batches, desc="running forward pass"):
            # get data and labels
            labels = batch['labels'].to(self.args.device)
            true.append(labels.cpu())
            batch = {k: v.to(self.args.device) for k, v in batch.items()}  
            
            with torch.no_grad():
                # forward time
                logits, _ = model(**batch)
                # compute loss
                loss = model.compute_loss(logits, labels)
                cum_loss += loss.item()
                pred.append(torch.sigmoid(logits).cpu())
                num_batches += 1

        avg_loss = cum_loss / num_batches if num_batches else None
        
        true, pred = torch.cat(true).numpy(), torch.cat(pred).numpy()
        result = self._compute_classification_metrics(true, pred, avg_loss)
        
        self.args.logger.write(f"Result on {split} split at train step {train_step}: {format_dict(result)}")

        return result

    def _compute_classification_metrics(self, true, pred, loss):
        if len(np.unique(true)) < 2:
            self.args.logger.write(f"WARNING: Only one class present in y_true (unique={np.unique(true)}). Metrics set to NaN.")
            nan = float('nan')
            return {'loss': loss, 'auroc': nan, 'auprc': nan, 'minrp': nan,
                    'precision': nan, 'recall': nan, 'balanced_acc': nan,
                    'mcc': nan, 'f1': nan, 'f2': nan}
        precision, recall, _ = precision_recall_curve(true, pred)
        pr_auc = auc(recall, precision)
        minrp = np.minimum(precision, recall).max()
        roc_auc = roc_auc_score(true, pred)
        pred_binary = (pred >= self.thresh).astype(int)

        return {
            'loss': loss, 'auroc': roc_auc, 'auprc': pr_auc, 'minrp': minrp,
            'precision': precision_score(true, pred_binary, zero_division=0),
            'recall': recall_score(true, pred_binary, zero_division=0),
            'balanced_acc': balanced_accuracy_score(true, pred_binary),
            'mcc': matthews_corrcoef(true, pred_binary),
            'f1': f1_score(true, pred_binary),
            'f2': fbeta_score(true, pred_binary, beta=2, zero_division=0),
        }
        
        
class EvaluatorPretrain(Evaluator):
    def __init__(self, args, batcher):
        super().__init__(args, batcher)

    ## TO DO: ADD augmentation by 3 for cohort pretraining > check if all different shuffles or 3 fixed shuffles
    def evaluate(self, model, split, train_step):
        self.args.logger.write(f"\nEvaluating on split = {split}")

        batches = self.prepare_batches(split, cache=True, rep=3)
        model.eval()
        
        loss, count = 0.0, 0.0

        for batch in tqdm(batches, desc="running forward pass"):
            batch = {k:v.to(self.args.device) for k,v in batch.items()}
            num_pred = batch['forecast_mask'].sum()
            if num_pred == 0:
                self.args.logger.write('Skipped empty batch in evaluation.')
                continue  # skip empty batches
        
            with torch.no_grad():
                train_loss, _ = model(**batch)
                loss += train_loss * num_pred
                count += num_pred
                
        result = {'loss': (loss / count).item() if count else None}
        self.args.logger.write(f"Result on {split} split at train step {train_step}: {format_dict(result)}")
        return result


"""
        if split == "val" and self.args.calibrate_threshold:
            best_thresh = 0.5
            best_metric = -float('inf')
            for thresh in thresholds:
                pred_bin = (pred >= thresh).int()
                #f1 = f1_score(true, pred_bin, zero_division=0)
                f2 = fbeta_score(true, pred_bin, beta=2, zero_division=0)
                if f2 > best_metric:
                    best_metric = f2
                    best_thresh = thresh   
            self.thresh = best_thresh   
            self.args.logger.write(f"Calibrated threshold set to {self.thresh:.3f}")
"""  