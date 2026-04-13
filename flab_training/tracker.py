import time
import torch
import psutil
import os
from tqdm import tqdm

class Tracker:
    def __init__(self, args):
        self.args = args
        self.train_peak_memory = 0 #computed over each epoch in train set
        self.eval_peak_memory = 0 #computed over test set
        self.epoch_train_times = [] 
        self.epoch_train_memory = []
        self.total_eval_time = 0
        self.inference = False
    
    # ----- internal helpers -----
    def test_memory(self, peak_memory):
        """Track current memory and update peak memory for the epoch."""
        if self.args.device == "cuda":
            torch.cuda.synchronize()
            current_mem = torch.cuda.max_memory_allocated(self.args.device)
        else:
            process = psutil.Process(os.getpid())
            current_mem = process.memory_info().rss
        return max(peak_memory, current_mem)

    def reset_memory(self):
        """Reset memory counters for the current device."""
        if self.args.device == "cuda":
            torch.cuda.reset_max_memory_allocated(torch.device("cuda:0")) 
    
    # ----- training tracking -----   
    def update_memory_batch(self):     
        """Update peak memory during a batch."""
        self.train_peak_memory = self.test_memory(self.train_peak_memory)
        
    def reset_train_epoch(self):
        """Reset timers and memory trackers at the start of an epoch."""
        self.epoch_train_start = time.time()
        self.train_peak_memory = 0
        self.reset_memory()

    def update_train_epoch(self):
        """Store duration and peak memory at the end of an epoch."""
        epoch_train_end = time.time()
        epoch_train_duration = epoch_train_end - self.epoch_train_start
        self.epoch_train_times.append(epoch_train_duration)
        self.epoch_train_memory.append(self.train_peak_memory)

    def update_train_total(self):
        """Compute total and mean metrics across all epochs."""
        self.best_epoch = len(self.epoch_train_times)
        self.total_train_time = sum(self.epoch_train_times)
        self.mean_epoch_train_time = self.total_train_time / self.best_epoch
        self.max_train_peak_memory = max(self.epoch_train_memory)
        self.mean_train_peak_memory = sum(self.epoch_train_memory) / self.best_epoch

    # ----- evaluation / inference -----
    
    def compute_inference_metrics(self, model, batches):
        model.eval()
        self.inference = True
        self.total_eval_time = 0
        self.eval_peak_memory = 0
        preds = []
        self.reset_memory()
        
        for batch in tqdm(batches, desc="running forward pass"):
            batch = {k: v.to(self.args.device) for k, v in batch.items()}  
            with torch.no_grad():
                start_time = time.time()
                logits, _ = model(**batch)
                preds.append(torch.sigmoid(logits).cpu())
                end_time = time.time()
                self.total_eval_time += end_time - start_time # compute time
                self.eval_peak_memory = self.test_memory(self.eval_peak_memory) # test memory

        self.num_test_samples = len(preds)
        self.args.logger.write(f"Inference completed.")
    
    
    # ----- summary logging -----
    def log_final(self):
        """Return a dictionary of overall training (and optionally inference) metrics."""
        final_dict = {
            # Time and memory metrics at best model
            'epoch': self.best_epoch,                     # number of epochs
            'total_train_time': self.total_train_time,   # total training time
            'epoch_train_time': self.mean_epoch_train_time,  # avg training time per epoch
            'total_peak_memory': self.max_train_peak_memory / 1e6, # max training peak memory
            'mean_peak_memory': self.mean_train_peak_memory / 1e6, # avg training peak memory per epoch
        }
        # add inference metrics if inference was applied (no inference in self supervised learning)
        if self.inference:
            final_dict.update({
                'total_inference_time': self.total_eval_time,                 # total inference time
                'sample_inference_time': self.total_eval_time / self.num_test_samples,  # time per sample
                'inference_peak_memory': self.eval_peak_memory / 1e6,               # max inference memory
                'sample_inference_memory': self.eval_peak_memory / (self.num_test_samples*1e6) # avg memory per sample
            })
        return final_dict