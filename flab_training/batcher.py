import numpy as np
import torch
from flab_training.cycler import CycleIndex, CycleIndexBalanced
from flab_training.utils import compute_event_masks
import pandas as pd
class Batcher:
    def __init__(self, args, input_dict):
        self.args = args
        self.input_dict = input_dict
        self.splits = self.input_dict.get("splits")
        self.demo = self.input_dict.get("demo_norm")
        self.y = self.input_dict.get("target") # none if attribute missing
        self.set_cycler()


    def _get_indices(self, ind):
        if ind is None:
            return self.train_cycler.get_batch_ind()
        return ind

    def get_batch(self, ind=None):
        raise NotImplementedError
    
    def set_cycler(self):    
        if self.args.train_mode == "pretrain":
            self.train_cycler = CycleIndex(self.splits['train'], self.args.train_batch_size)
        elif self.args.stratify_batch > 0:
            self.train_cycler = CycleIndexBalanced(self.splits['train'], self.y[self.splits['train']], self.args.train_batch_size, self.args.stratify_batch)
        else:
            self.train_cycler = CycleIndex(self.splits['train'], self.args.train_batch_size)

class BatcherA(Batcher):
    def __init__(self, args, input_dict):
        super().__init__(args, input_dict) 
        self.X = self.input_dict.get("X")

    def get_batch(self, ind=None):
        ind = self._get_indices(ind)
        return {
            'ts': torch.FloatTensor(self.X[ind]),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }

class BatcherB(Batcher):
    def __init__(self, args, input_dict):
        super().__init__(args, input_dict) 
        self.values = self.input_dict.get("values")
        self.mask = self.input_dict.get("mask")
        if self.args.model_type == "grud":
            self.deltas = self.input_dict.get("deltas")
        elif self.args.model_type == "interpnet":
            self.times = self.input_dict.get("times")
            self.holdout_masks = self.input_dict.get("holdout_masks")
        else:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")  

    def _pad_and_stack(self, sequences, pad_mats):
        return torch.FloatTensor(
            np.stack([
                np.concatenate((seq, pad), axis=0) if len(seq) > 0 else pad
                for seq, pad in zip(sequences, pad_mats)
            ])
        )

    def _make_pad_mats(self, num_timestamps, V):
        max_timestamps = max(num_timestamps)
        pad_lens = max_timestamps - num_timestamps
        pad_mats = [np.zeros((l, V)) for l in pad_lens]
        return pad_mats, pad_lens

    def get_batch(self, ind=None):
        ind = self._get_indices(ind)
        if self.args.model_type == "grud":
            return self.get_batch_grud(ind)
        elif self.args.model_type == "interpnet":
            return self.get_batch_interpnet(ind)
        else:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")  

    def get_batch_grud(self, ind=None):
        deltas = [self.deltas[i] for i in ind]
        values = [self.values[i] for i in ind]
        masks = [self.mask[i] for i in ind]

        num_timestamps = np.array([len(d) for d in deltas])
        pad_mats, _ = self._make_pad_mats(num_timestamps, self.args.V)

        return {
            'x_t': self._pad_and_stack(values, pad_mats),
            'm_t': self._pad_and_stack(masks, pad_mats),
            'delta_t': self._pad_and_stack(deltas, pad_mats),
            'seq_len': torch.LongTensor(num_timestamps),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }

    def get_batch_interpnet(self, ind=None):
        times = [self.times[i] for i in ind]
        values = [self.values[i] for i in ind]
        masks = [self.mask[i] for i in ind]
        hmasks = [self.holdout_masks[i] for i in ind]

        num_timestamps = np.array(list(map(len, times)))
        pad_mats, pad_lens = self._make_pad_mats(num_timestamps, self.args.V)

        return {
            't': torch.FloatTensor([t + [0] * p for t, p in zip(times, pad_lens)]),
            'x': self._pad_and_stack(values, pad_mats),
            'm': self._pad_and_stack(masks, pad_mats),
            'h': self._pad_and_stack(hmasks, pad_mats),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }

class BatcherC_unsup(Batcher):
    def __init__(self, args, input_dict):
        super().__init__(args, input_dict) 
        self.values = self.input_dict.get("values")
        self.times = self.input_dict.get("times")
        self.varis = self.input_dict.get("varis")
        self.timestamps = self.input_dict.get("timestamps")     
        self.max_minute = self.args.window_forecast #7*24*60 #args.window_forecast*24*60 if not None else 7*24*60
        self.pred_int = self.args.window_pred #1*24*60 #12*60 1 day forecasting window as opposed to 12 hours

    def get_batch(self, ind=None):
        ind = self._get_indices(ind)

        input_values = []
        input_times = []
        input_varis = []
        forecast_values = torch.zeros((len(ind),self.args.V))
        forecast_mask = torch.zeros((len(ind),self.args.V), dtype=torch.int)
        for b,i in enumerate(ind):
            t1 = np.random.choice(self.timestamps[i]) # minutes
            curr_times = self.times[i]
            for ix in range(len(curr_times)-1,-1,-1):
                if curr_times[ix]==t1:
                    t1_ix = ix+1 # start of prediction window
                    break
            t0_ix = max(0,t1_ix-self.args.max_obs)

            while curr_times[t0_ix]<t1-self.max_minute:
                t0_ix += 1
            if t1>self.max_minute: # shift times
                diff = t1-self.max_minute
                input_times.append(list(np.array(self.times[i][t0_ix:t1_ix])-diff))
            else:
                input_times.append(self.times[i][t0_ix:t1_ix])
            input_values.append(self.values[i][t0_ix:t1_ix])
            input_varis.append(self.varis[i][t0_ix:t1_ix])

            t2 = t1+self.pred_int
            for t2_ix in range(t1_ix, len(curr_times)):
                if curr_times[t2_ix]>t2:
                    break
            # t2_ix: last+1 for prediction window
            curr_varis = self.varis[i]
            curr_values = self.values[i]
            for ix in range(t2_ix-1,t1_ix-1,-1):
                vari = curr_varis[ix]
                val = curr_values[ix]
                forecast_mask[b,vari] = 1
                forecast_values[b,vari] = val

        num_obs = list(map(len, input_values))
        max_obs = max(num_obs)
        pad_lens = max_obs-np.array(num_obs)
        values = [x+[0]*(l) for x,l in zip(input_values,pad_lens)]
        times = [x+[0]*(l) for x,l in zip(input_times,pad_lens)]
        varis = [x+[0]*(l) for x,l in zip(input_varis,pad_lens)]
        values, times = torch.FloatTensor(values), torch.FloatTensor(times)
        times = times/self.max_minute*2-1
        varis = torch.IntTensor(varis)
        obs_mask = [[1]*l1+[0]*l2 for l1,l2 in zip(num_obs,pad_lens)]
        obs_mask = torch.IntTensor(obs_mask)

        #self.args.logger.write(num_obs)

        return {'values':values, 'times':times, 'varis':varis,
                'obs_mask':obs_mask, 
                'demo':torch.FloatTensor(self.demo[ind]),
                'forecast_values':forecast_values,
                'forecast_mask':forecast_mask}


class BatcherC_sup(Batcher):
    def __init__(self, args, input_dict):
        super().__init__(args, input_dict) 
        self.values = self.input_dict.get("values")
        self.times = self.input_dict.get("times")
        self.varis = self.input_dict.get("varis")

    def get_batch(self, ind=None):
        ind = self._get_indices(ind)

        num_obs = [len(self.values[i]) for i in ind]
        max_obs = max(num_obs)
        pad_lens = max_obs - np.array(num_obs)

        values = [self.values[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        times = [self.times[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        varis = [self.varis[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        values, times = torch.FloatTensor(values), torch.FloatTensor(times)
        varis = torch.IntTensor(varis)
        obs_mask = [[1]*l1+[0]*l2 for l1,l2 in zip(num_obs,pad_lens)]
        obs_mask = torch.IntTensor(obs_mask)

        return {
            'values': values,
            'times': times,
            'varis': varis,
            'obs_mask': obs_mask,
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }


class BatcherD_unsup(Batcher):
    # Dynamic similar to STraTs 
    def __init__(self, args, input_dict):
        super().__init__(args, input_dict) 
        self.values = self.input_dict.get("values")
        self.times = self.input_dict.get("times")
        self.varis = self.input_dict.get("varis")
        self.timestamps = self.input_dict.get("timestamps")     
        self.max_minute = self.args.window_forecast
        self.pred_int = self.args.window_pred 
        
        self.event_mask_threshold = self.args.event_mask_threshold
        self.insignificant_prob = self.args.insignificant_prob
        self.train_ts_inds = set(self.splits.get("train", []))
    
    def get_batch(self, ind=None):
        ind = self._get_indices(ind)
        input_times, input_values, input_varis = [], [], []
        
        forecast_values = torch.zeros((len(ind), self.args.V))
        forecast_mask = torch.zeros((len(ind), self.args.V), dtype=torch.int)
        
        for b, i in enumerate(ind):
            t1 = np.random.choice(self.timestamps[i]) 
            curr_times = self.times[i]
            
            # Find index of t1 in times
            #t1_ix = len(curr_times)  # Default to end if not found
            for ix in range(len(curr_times)-1, -1, -1):
                if curr_times[ix] == t1:
                    t1_ix = ix + 1  # start of prediction window
                    break
            t0_ix = max(0, t1_ix - self.args.max_obs)

            # Shift lookback window start to ensure within max_minute
            while curr_times[t0_ix] < t1 - self.max_minute:
                t0_ix += 1
            
            # TIME SHIFTING (like STRATS) - normalize time reference point
            if t1 > self.max_minute:
                diff = t1 - self.max_minute
                input_times.append(list(np.array(self.times[i][t0_ix:t1_ix]) - diff))
            else:
                input_times.append(self.times[i][t0_ix:t1_ix])
            
            input_values.append(self.values[i][t0_ix:t1_ix])
            input_varis.append(self.varis[i][t0_ix:t1_ix])

            t2 = t1+self.pred_int
            for t2_ix in range(t1_ix, len(curr_times)):
                if curr_times[t2_ix]>t2:
                    break

            
            curr_varis = self.varis[i]
            curr_values = self.values[i]
            for ix in range(t2_ix-1,t1_ix-1,-1):
                vari = curr_varis[ix]
                val = curr_values[ix]
                forecast_mask[b,vari] = 1
                forecast_values[b,vari] = val

        # Compute event masks on-the-fly: stochastic for train, deterministic for val (stable val curve)
        is_train = any(i in self.train_ts_inds for i in ind)
        insig_prob = self.insignificant_prob if is_train else 0.0
        input_event_masks = compute_event_masks(
            input_times, input_values, input_varis,
            V=self.args.V,
            threshold=self.event_mask_threshold,
            insignificant_prob=insig_prob,
            logger=self.args.logger
        )

        # Pad sequences
        num_obs = list(map(len, input_values))
        max_obs = max(num_obs)
        pad_lens = max_obs - np.array(num_obs)
        values = [x + [0]*l for x, l in zip(input_values, pad_lens)]
        times = [x + [0]*l for x, l in zip(input_times, pad_lens)]
        varis = [x + [0]*l for x, l in zip(input_varis, pad_lens)]
        event_masks = [list(x) + [False]*l for x, l in zip(input_event_masks, pad_lens)]
        values, times = torch.FloatTensor(values), torch.FloatTensor(times)
        times = times / self.max_minute * 2 - 1
        varis = torch.IntTensor(varis)
        event_masks = torch.BoolTensor(event_masks)
        obs_mask = [[1]*l1 + [0]*l2 for l1, l2 in zip(num_obs, pad_lens)]
        obs_mask = torch.IntTensor(obs_mask)

        return {
            'values': values,
            'times': times,
            'varis': varis,
            'obs_mask': obs_mask,
            'event_mask': event_masks,
            'demo': torch.FloatTensor(self.demo[ind]),
            'forecast_values': forecast_values,
            'forecast_mask': forecast_mask
        }


class BatcherD_unsup_fixed(Batcher):
    # Fixed timestepping EMIT original approach
    def __init__(self, args, input_dict):
        super().__init__(args, input_dict)
        
        self.values = self.input_dict.get("values")
        self.times = self.input_dict.get("times")  # Already in hours 
        self.varis = self.input_dict.get("varis")
        self.timestamps = self.input_dict.get("timestamps")
        self.obs_window  = self.args.window_forecast / 60
        self.pred_window = self.args.window_pred / 60
        
        self.fore_max_len = 880  # max events in observation window (matching original EMIT)
        
        self.event_mask_threshold = self.args.event_mask_threshold
        self.insignificant_prob = self.args.insignificant_prob
        
        self.train_ts_inds = set(self.splits.get("train", []))
        self.val_ts_inds = set(self.splits.get("val", []))
        
        self.window_indices = {i: 0 for i in range(len(self.timestamps))}

    def get_batch(self, ind=None):

        ind = self._get_indices(ind) #get batch indices
        
        input_values, input_times, input_varis = [], [], []
        input_event_masks = []
        
        forecast_values = torch.zeros((len(ind), self.args.V))
        forecast_mask = torch.zeros((len(ind), self.args.V), dtype=torch.int)
        
        for b, i in enumerate(ind):

            if len(self.timestamps[i]) == 0:
                continue  # Skip if no valid windows (shouldn't happen after filtering)
            
            
            window_idx = np.random.randint(0, len(self.timestamps[i]))
            w_hour = self.timestamps[i][window_idx]
            
            
            times_arr = np.array(self.times[i])
            values_arr = np.array(self.values[i])
            varis_arr = np.array(self.varis[i])
            
            obs_start = w_hour - self.obs_window
            obs_mask_bool = (times_arr >= obs_start) & (times_arr < w_hour)
            
            obs_times = times_arr[obs_mask_bool]
            obs_values = values_arr[obs_mask_bool]
            obs_varis = varis_arr[obs_mask_bool]
            
                                  
            # Limit to first fore_max_len events (matching original EMIT)
            if len(obs_times) > self.fore_max_len:
                obs_times = obs_times[:self.fore_max_len]
                obs_values = obs_values[:self.fore_max_len]
                obs_varis = obs_varis[:self.fore_max_len]
                
                
            input_times.append(list(obs_times))  # Keep in hours (no normalization)
            input_values.append(list(obs_values))
            input_varis.append(list(obs_varis))
        
    
            
            
            #Build forecast targets
            pred_end = w_hour + self.pred_window
            pred_mask_bool = (times_arr >= w_hour) & (times_arr <= pred_end)
            

            pred_times = times_arr[pred_mask_bool]
            pred_values = values_arr[pred_mask_bool]
            pred_varis = varis_arr[pred_mask_bool]
                
            # Build forecast targets: FIRST value per variable
            for idx in range(len(pred_times)):
                v = pred_varis[idx]
                if 0 <= v < self.args.V and forecast_mask[b, v] == 0:  # Not set yet first value per variable
                    forecast_values[b, v] = pred_values[idx]
                    forecast_mask[b, v] = 1
                        
        # Compute event masks: stochastic for train, deterministic for val (stable val curve)
        is_train = any(i in self.train_ts_inds for i in ind)
        insig_prob = self.insignificant_prob if is_train else 0.0
        input_event_masks = compute_event_masks(
            input_times, input_values, input_varis,
            V=self.args.V,
            threshold=self.event_mask_threshold,
            insignificant_prob=insig_prob,
            logger=self.args.logger
        )
        

        num_obs = list(map(len, input_values))
        max_obs = max(num_obs)
        pad_lens = max_obs - np.array(num_obs)
        values = [x + [0]*l for x, l in zip(input_values, pad_lens)]
        times = [x + [0]*l for x, l in zip(input_times, pad_lens)]
        varis = [x + [0]*l for x, l in zip(input_varis, pad_lens)]
        event_masks = [list(x) + [False]*l for x, l in zip(input_event_masks, pad_lens)]
        values = torch.FloatTensor(values)
        times = torch.FloatTensor(times)  # Keep in hours (NO normalization)
        varis = torch.IntTensor(varis)
        event_masks = torch.BoolTensor(event_masks)
        obs_mask = [[1]*l1 + [0]*l2 for l1, l2 in zip(num_obs, pad_lens)]
        obs_mask = torch.IntTensor(obs_mask)
        
        return {
            'values': values,
            'times': times,  # In hours, not normalized
            'varis': varis,
            'obs_mask': obs_mask,
            'event_mask': event_masks,
            'demo': torch.FloatTensor(self.demo[ind]),
            'forecast_values': forecast_values,
            'forecast_mask': forecast_mask
        }