import os
import pickle
import numpy as np
import pandas as pd
from flab_training.utils import discrete_tensors, fill_impute, fill_mean, compute_means_stds_df, compute_deltas, compute_holdout, compute_event_masks
from pathlib import Path
class Preprocessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.args = dataset.args
        self.data = dataset.data
        self.train_ind = self.dataset.splits["train"]

    def set_variables(self):          
        self.variables = self.get_vars()
        self.var_to_ind = {v: i for i, v in enumerate(self.variables)}
        # remove vars not in pretrain
        self.data = self.data[self.data.itemid.isin(self.variables)]
        self.data["var_ind"] = self.data.itemid.map(self.var_to_ind)
        self.args.V = len(self.variables)
        self.args.logger.write('\nTemporal variables: ' + ', '.join(self.variables))

    def get_vars(self):
        return sorted(self.data.itemid.unique())

    def trim(self):
        raise NotImplementedError

    def compute_means_stds(self):
        raise NotImplementedError

    def normalise(self):
        raise NotImplementedError

    def prepare_inputs(self):
        raise NotImplementedError
    
    def save_inputs(self):
        # merge current input_dict with splits and ts_id_to_ind
        self.input_dict = {
            **self.input_dict,
            "demo_norm": self.dataset.demo,
            "demo_raw": self.dataset.demo_raw,
            "demo_means" : self.dataset.demo_means, 
            "demo_stds" : self.dataset.demo_stds,           
            "splits": self.dataset.splits,
            "ts_id_to_ind": self.dataset.ts_id_to_ind,
            "var_to_ind": self.var_to_ind
        } # if the task is supervised, save also the target
        if self.args.train_mode != "pretrain":
            self.input_dict["target"] = self.dataset.y
            
        # pickle dump only in stats mode, for these cohorts, and not in nested crossvalidation
        save_cohorts = ["A08-K52","C25-D63"]
        if self.args.variant == "stats" and self.args.cohort in save_cohorts and self.args.cv_mode != "grid":
            output_path = Path(self.dataset.args.paths["output_path"]) / "input_dict.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(self.input_dict, f)


class PreprocessorA(Preprocessor):

    # preprocessing params
    # args.agg_int (default 24)
    def trim(self):
        self.data = self.data.sort_values("minute")
        self.data['int'] = (self.data['minute'] // (60 * self.args.agg_int)).astype(int)
        self.args.T = self.data.int.max() + 1
        self.args.logger.write(f'Time bins: {self.args.T} x {self.args.agg_int/24:.1f} days (from the observed data range)')

    def normalise(self):
        self.means, self.stds = self.compute_means_stds()
        self.values = (self.values-self.means)/self.stds
        self.args.logger.write('Data normalised')

    def compute_means_stds(self):
        means = self.values[self.train_ind].mean(axis=(0, 1), keepdims=True)
        stds = self.values[self.train_ind].std(axis=(0, 1), keepdims=True)
        stds = np.where(stds == 0, 1.0, stds)
        return means, stds

    def prepare_inputs(self):

        ## DISCRETISATION
        self.set_variables()
        self.trim()
        last, avgs, self.obs, self.delta, sums, counts = discrete_tensors(self.data, self.args.N, self.args.T, self.args.V)

        ## AGGREGATION 
        if self.args.agg == "mean": # aggregation by average value
            self.values = avgs
        else: # default is aggregation by last recorded value
            self.values = last

        ## IMPUTATION
        if self.args.impute == "fill":
            self.values = fill_impute(self.values, self.obs)
        else: # default is mean imputation
            self.values = fill_mean(self.values, self.obs, self.train_ind)     
        self.input_dict = {"values_raw" : self.values, "obs" : self.obs, "delta" : self.delta}
        # compute means and stds
        self.normalise()
        self.input_dict.update({"values_norm": self.values, "values_means": self.means, "values_stds": self.stds})

        
        # VARIANTS (concatenate data)
        variant = self.args.variant
        if variant == "V":
            self.X = np.concatenate((self.values,), axis=-1)
        elif variant == "M":
            self.X = np.concatenate((self.obs,), axis=-1)
        elif variant == "D":
            self.X = np.concatenate((self.delta,), axis=-1)
        elif variant == "MD":
            self.X = np.concatenate((self.obs, self.delta), axis=-1)
        elif variant == "VM":
            self.X = np.concatenate((self.values, self.obs), axis=-1)
        elif variant == "VD":
            self.X = np.concatenate((self.values, self.delta), axis=-1)
        else:  # default: VMD
            self.X = np.concatenate((self.values, self.obs, self.delta), axis=-1)
        self.input_dict["X"] = self.X
        self.args.F = self.X.shape[-1]
        self.args.logger.write(f'Input prepared. Shape: {self.X.shape} (N, T, F={self.args.F})')


class PreprocessorB(Preprocessor):
    # preprocessing params
    # args.max_timesteps (default 1000)

    def trim(self):
        # eliminate duplicate lab measurements
        self.data = self.data.groupby(["hadm_id", "ts_ind", "itemid","var_ind","minute"]).value.mean().reset_index()
        if self.args.max_timesteps != -1:
            timestamps = self.data[['hadm_id', 'minute']].drop_duplicates().sample(frac=1)
            timestamps = timestamps.groupby('hadm_id').head(self.args.max_timesteps)
            self.data = self.data.merge(timestamps, on=['hadm_id', 'minute'], how='inner')
            self.args.logger.write('\nData trimmed to max length')
    
    def compute_means_stds(self):  
        return compute_means_stds_df(self.data, self.train_ind)

    def normalise(self):
        means_stds = self.compute_means_stds()
        self.data = self.data.merge(means_stds, on='itemid', how='left')
        self.data['value'] = (self.data['value'] - self.data['mean']) / self.data['std']
        self.args.logger.write('Data normalised')

    def prepare_inputs(self):
        model_type = self.args.model_type
        
        self.set_variables()
        self.trim()
        self.normalise()

        N = self.args.N
        V = self.args.V
        
        if model_type == 'grud':
            deltas = [[] for _ in range(N)]
        elif model_type == 'interpnet':
            times = [[] for _ in range(N)]
            holdout_masks = [[] for _ in range(N)]
        values = [[] for _ in range(N)]
        mask = [[] for _ in range(N)]

        for ts_ind, curr_data in self.data.groupby('ts_ind'):

            # get observed time points
            curr_times = sorted(curr_data.minute.unique())

            # construct value/mask matrices
            pivot_data = (
                curr_data
                .pivot(index="var_ind", columns="minute", values="value")
                .reindex(list(range(V)))  # consistent ordering
                .T
            )
            curr_values = pivot_data.fillna(0).to_numpy()
            curr_mask = pivot_data.notna().astype(int).to_numpy()

            # get deltas for grud model
            if model_type == 'grud':
                max_time = (self.args.days_before_discharge + 1) * 24 * 60  # minutes
                deltas[ts_ind] = compute_deltas(curr_times, curr_mask, max_time)

            # get masks for interpnet model
            elif model_type == 'interpnet':
                times[ts_ind] = list(np.array(curr_times) / 60)  # convert to hours
                curr_mask[0, :] = 1  # ensure at least one observation per feature
                holdout_masks[ts_ind] = compute_holdout(curr_mask)

            values[ts_ind] = curr_values
            mask[ts_ind] = curr_mask
            
        # save results to self
        self.values = values
        self.mask = mask
        self.input_dict = {"values" : self.values, "mask" : self.mask}
        if model_type == 'grud':
            self.deltas = deltas
            self.input_dict["deltas"] = self.deltas
        elif model_type == 'interpnet':
            self.times = times
            self.holdout_masks = holdout_masks
            self.input_dict["times"] = self.times
            self.input_dict["holdout_masks"] = self.holdout_masks
        self.args.logger.write('Input prepared')

class PreprocessorC(Preprocessor): # strats
    # preprocessing params
    # args.max_obs (default 1000) - can be set to -1 for no trimming
        
    def get_vars(self):
        raise NotImplementedError

    def compute_means_stds(self):
        raise NotImplementedError
            
    def normalise(self):
        means_stds = self.compute_means_stds()
        self.data = self.data.merge(means_stds, on='itemid', how='left')
        self.data['value'] = (self.data['value'] - self.data['mean']) / self.data['std']
        self.args.logger.write('Data normalised')
        
    def prepare_inputs(self):
        raise NotImplementedError
    
    
class PreprocessorC_unsup(PreprocessorC): # unsupervised
    
    def trim(self):
        # eliminate duplicate lab measurements
        self.data = self.data.groupby(["hadm_id", "ts_ind", "itemid","var_ind","minute"]).value.mean().reset_index()
        #self.data = self.data.sample(frac=1)
        #self.data = self.data.groupby('hadm_id').head(self.args.max_obs) # trimming is applied at batching stage
    
    def get_vars(self):
        self.pt_variables = sorted(self.data.itemid.unique())
        return self.pt_variables

    def compute_means_stds(self):     
        self.pt_means_stds = compute_means_stds_df(self.data, self.train_ind)
        return self.pt_means_stds

    def dump_stats(self):
        pt_var_path = os.path.join(self.args.paths["output_path"], 'pt_saved_variables.pkl')
        with open(pt_var_path, 'wb') as f:
            pickle.dump((self.pt_variables, self.pt_means_stds), f)

    def prepare_inputs(self):
        self.set_variables()
        self.trim()
        self.normalise()
        self.dump_stats()

        N = self.args.N
        values = [[] for _ in range(N)]
        times = [[] for _ in range(N)]
        varis = [[] for _ in range(N)]

        self.data = self.data.sample(frac=1).sort_values(by='minute')
        for row in self.data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(row.var_ind)
        self.values, self.times, self.varis = values, times, varis
        
        # unique sorted timestamps except the last one for each patient
        self.timestamps = [np.array(sorted(list(set(x)))[:-1]) for x in self.times]
        # only keep timepoints that occur 12h or later
        self.timestamps = [x[x>=720] for x in self.timestamps] 
        self.input_dict = {"values" : self.values, "times" : self.times, "varis": self.varis, "timestamps": self.timestamps}
        self.args.logger.write('Input prepared.')
        
        # get all admissions where there are no valid timestamps left after filtering
        delete = [i for i in range(self.args.N) if len(self.timestamps[i])==0]
        # remove from splits
        self.dataset.splits = {k:np.setdiff1d(v,delete) for k,v in self.dataset.splits.items()}    
        self.args.logger.write(str(len(delete)) + ' admissions removed.')   
        

class PreprocessorC_sup(PreprocessorC): # supervised

    def __init__(self, dataset):
        super().__init__(dataset)
        # If finetuning, load precomputed variables and normalization stats
        if self.args.train_mode == "finetune":
            self.pt_variables, self.pt_means_stds = pickle.load(open(self.args.pt_var_path, 'rb'))   
            
    def trim(self):
        # eliminate duplicate lab measurements
        self.data = self.data.groupby(["hadm_id", "ts_ind", "itemid","var_ind","minute"]).value.mean().reset_index()
        self.data = self.data.sample(frac=1)
        self.data = self.data.groupby('hadm_id').head(self.args.max_obs) 

    def get_vars(self):
        if self.args.train_mode == "finetune":  
            return self.pt_variables
        else:
            return sorted(self.data.itemid.unique())
        
    def compute_means_stds(self):  
        # strats can be pretrained and normalisation should use precomputed stats
        if self.args.train_mode == "finetune":
            return self.pt_means_stds
        else:
            return compute_means_stds_df(self.data, self.train_ind)

    def prepare_inputs(self):
        self.set_variables()
        self.trim()
        self.normalise()

        N = self.args.N
        values = [[] for _ in range(N)]
        times = [[] for _ in range(N)]
        varis = [[] for _ in range(N)]

        max_time = (self.args.days_before_discharge + 1) * 24 * 60
        self.data['minute'] = self.data['minute']/max_time*2-1

        for row in self.data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(row.var_ind)
        self.values, self.times, self.varis = values, times, varis
        self.input_dict = {"values" : self.values, "times" : self.times, "varis": self.varis}
        self.args.logger.write('Input prepared')


class PreprocessorD(Preprocessor):  # EMIT

    def use_fixed_windows(self):
        return getattr(self.args, "emit_fixed_windows", False)

    def get_vars(self):
        raise NotImplementedError

    def compute_means_stds(self):
        raise NotImplementedError
            
    def normalise(self):
        means_stds = self.compute_means_stds()
        self.data = self.data.merge(means_stds, on='itemid', how='left')
        self.data['value'] = (self.data['value'] - self.data['mean']) / self.data['std']
        self.args.logger.write('Data normalised')
        
    def prepare_inputs(self):
        raise NotImplementedError
    

class PreprocessorD_unsup(PreprocessorD):  # EMIT unsupervised (pretraining)

    
    def trim(self):
        self.data = self.data.groupby(["hadm_id", "ts_ind", "itemid","var_ind","minute"]).value.mean().reset_index()
    
    def get_vars(self):
        self.pt_variables = sorted(self.data.itemid.unique())
        return self.pt_variables

    def compute_means_stds(self):     
        self.pt_means_stds = compute_means_stds_df(self.data, self.train_ind)
        return self.pt_means_stds

    def dump_stats(self):
        pt_var_path = os.path.join(self.args.paths["output_path"], 'pt_saved_variables.pkl')
        with open(pt_var_path, 'wb') as f:
            pickle.dump((self.pt_variables, self.pt_means_stds), f)

    def prepare_inputs(self):
              
        self.set_variables()
        self.trim()
        self.normalise()
        self.dump_stats()
        
        N = self.args.N
        values = [[] for _ in range(N)]
        times = [[] for _ in range(N)]
        varis = [[] for _ in range(N)]

        self.data = self.data.sample(frac=1).sort_values(by='minute')
        for row in self.data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(row.var_ind)
        self.values, self.times, self.varis = values, times, varis
        
        
        if self.use_fixed_windows():
            self.args.logger.write('Using FIXED-WINDOW EMIT preprocessing (original EMIT approach)')
            self.prepare_fixed_window_samples()
        else:
            self.args.logger.write('Using DYNAMIC-WINDOW EMIT preprocessing (STRATS-style)')
            self.prepare_dynamic_window_inputs()
    
    def prepare_dynamic_window_inputs(self):


        self.timestamps = [np.array(sorted(list(set(x)))[:-1]) for x in self.times]
        # only keep timepoints that occur 12h or later
        self.timestamps = [x[x>=720] for x in self.timestamps]
        self.input_dict = {"values" : self.values, "times" : self.times, "varis": self.varis, "timestamps": self.timestamps}
        self.args.logger.write('Input prepared EMIT (dynamic windowing).')
        
        # get all admissions where there are no valid timestamps left after filtering
        delete = [i for i in range(self.args.N) if len(self.timestamps[i])==0]
        # remove from splits
        self.dataset.splits = {k:np.setdiff1d(v,delete) for k,v in self.dataset.splits.items()}    
        self.args.logger.write(str(len(delete)) + ' admissions removed.')  
    
    def prepare_fixed_window_samples(self):

        self.times = [[t / 60 for t in times_list] for times_list in self.times]

        max_minute = self.args.window_forecast
        pred_int = self.args.window_pred

        obs_windows_param = self.args.emit_obs_windows
        if obs_windows_param is not None and isinstance(obs_windows_param, (list, tuple)) and len(obs_windows_param) == 3:
            obs_windows = range(*[int(x) for x in obs_windows_param])
        elif obs_windows_param is not None:
            raise ValueError("obs_windows_param must be a 3-element list")

        self.args.logger.write(f'Using observation windows: {list(obs_windows)}')
        self.args.logger.write(f'Using window_forecast (minutes): {max_minute}')
        self.args.logger.write(f'Using window_pred (minutes): {pred_int}')

        self.timestamps = []
        for ts_ind in range(self.args.N):
            if len(self.times[ts_ind]) == 0:
                self.timestamps.append(np.array([]))
                continue
            
            times_arr = np.array(self.times[ts_ind])
            valid_windows = []
            
            for w in obs_windows:
                w_hour = w  # Window boundary in hours 
                obs_start = w_hour - max_minute / 60
                pred_end = w_hour + pred_int / 60
                # Check if admission has data in both observation and prediction windows
                has_obs = np.any((times_arr >= obs_start) & (times_arr < w_hour))
                has_pred = np.any((times_arr >= w_hour) & (times_arr <= pred_end))
                
                if has_obs and has_pred:
                    valid_windows.append(w_hour)
            self.timestamps.append(np.array(valid_windows))

        self.input_dict = {"values" : self.values, "times" : self.times, "varis": self.varis, "timestamps": self.timestamps}
        self.args.logger.write('Input prepared EMIT (fixed windowing).')
        # Remove admissions with no valid timestamps
        delete = [i for i in range(self.args.N) if len(self.timestamps[i]) == 0]
        self.dataset.splits = {k: np.setdiff1d(v, delete) for k, v in self.dataset.splits.items()}
        self.args.logger.write(f'{len(delete)} admissions removed (no valid windows).')




class PreprocessorD_sup(PreprocessorD):  # EMIT supervised (finetuning)

    def __init__(self, dataset):
        super().__init__(dataset)
        # If finetuning, load precomputed variables and normalization stats
        if self.args.train_mode == "finetune":
            self.pt_variables, self.pt_means_stds = pickle.load(open(self.args.pt_var_path, 'rb'))   

        
        
            
    def trim(self):
        # eliminate duplicate lab measurements
        self.data = self.data.groupby(["hadm_id", "ts_ind", "itemid","var_ind","minute"]).value.mean().reset_index()
        self.data = self.data.sample(frac=1)
        self.data = self.data.groupby('hadm_id').head(self.args.max_obs) 

    def get_vars(self):
        if self.args.train_mode == "finetune":  
            return self.pt_variables
        else:
            return sorted(self.data.itemid.unique())
        
    def compute_means_stds(self):  
        # EMIT can be pretrained and normalisation should use precomputed stats
        if self.args.train_mode == "finetune":
            return self.pt_means_stds
        else:
            return compute_means_stds_df(self.data, self.train_ind)

    def prepare_inputs(self):
        self.set_variables()
        self.trim()
        self.normalise()
        
        if self.use_fixed_windows():
            self.args.logger.write('Using FIXED-WINDOW EMIT preprocessing Time is in hours')
            self.data['minute'] = self.data['minute'] / 60
        else:   
            max_time = (self.args.days_before_discharge + 1) * 24 * 60
            self.args.logger.write('Using DYNAMIC-WINDOW EMIT preprocessing Time is in minutes normalized to [-1, 1]')
            self.data['minute'] = self.data['minute']/max_time*2-1


        N = self.args.N
        values = [[] for _ in range(N)]
        times = [[] for _ in range(N)]
        varis = [[] for _ in range(N)]


        for row in self.data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(row.var_ind)
        self.values, self.times, self.varis = values, times, varis
        self.input_dict = {"values" : self.values, "times" : self.times, "varis": self.varis}
        self.args.logger.write('Input prepared')




'''class PreprocessorML(Preprocessor):

    def get_feature_names(self, variant):
        vars = self.variables
        T = self.args.T
        per_t = []
        if "V" in variant:
            per_t += [f"{v}_V" for v in vars]
        if "M" in variant:
            per_t += [f"{v}_M" for v in vars]
        if "D" in variant:
            per_t += [f"{v}_D" for v in vars]
        if self.args.feature_combination_method == "concatenate":
            ts_cols = [f"{col}_Bin{t}" for t in range(T) for col in per_t]
        else:
            ts_cols = per_t
        demo_cols = list(self.dataset.static_data.columns)
        return ts_cols + demo_cols

    def trim(self):
        # mean of duplicate measurements
        #self.data = self.data.groupby(["hadm_id", "itemid","minute","ts_ind","var_ind"]).value.mean().reset_index()

        self.data = self.data.sort_values("minute")
        self.data['int'] = (self.data['minute'] // (60 * self.args.agg_int)).astype(int)
        self.args.T = self.data.int.max() + 1
        self.args.logger.write('\nData discretised')
        self.args.logger.write('# intervals: '+str(self.args.T))
        
    def normalise(self):
        self.means, self.stds = self.compute_means_stds()
        self.values = (self.values-self.means)/self.stds
        self.args.logger.write('Data normalised')

    def compute_means_stds(self):
        means = self.values[self.train_ind].mean(axis=(0, 1), keepdims=True)
        stds = self.values[self.train_ind].std(axis=(0, 1), keepdims=True)
        stds = np.where(stds == 0, 1.0, stds)
        return means, stds

    def prepare_inputs(self):
        self.set_variables()
        self.trim()
        last, avgs, self.obs, self.delta, sums, counts = discrete_tensors(self.data, self.args.N, self.args.T, self.args.V)
        self.values = avgs
        
        ## IMPUTATION
        if self.args.impute == "fill":
            self.values = fill_impute(self.values, self.obs)
        else: # default is mean imputation
            self.values = fill_mean(self.values, self.obs, self.train_ind)     
            
        self.input_dict = {"values_raw" : self.values, "obs" : self.obs, "delta" : self.delta}
        # compute means and stds
        self.normalise()
        self.input_dict.update({"values_norm": self.values, "values_means": self.means, "values_stds": self.stds})
        
        
        variant = self.args.variant
        if variant == "V":
            X_3d = self.values
        elif variant == "M":
            X_3d = self.obs
        elif variant == "D":
            X_3d = self.delta
        elif variant == "MD":
            X_3d = np.concatenate((self.obs, self.delta), axis=-1)
        elif variant == "VM":
            X_3d = np.concatenate((self.values, self.obs), axis=-1)
        elif variant == "VD":
            X_3d = np.concatenate((self.values, self.delta), axis=-1)
        else:
            X_3d = np.concatenate((self.values, self.obs, self.delta), axis=-1)
        

        if self.args.feature_combination_method == 'concatenate':
            X_flat_ts = X_3d.reshape(X_3d.shape[0], -1)
        else:
            X_flat_ts = X_3d.mean(axis=1)
            

        X = np.concatenate([X_flat_ts, self.dataset.demo], axis=1) # Demo raw or normalized?
        #if self.args.model_type == 'logistic_regression':
        #    from sklearn.preprocessing import StandardScaler
        #    scaler = StandardScaler()
        #    scaler.fit(X[self.train_ind])
        #    X = scaler.transform(X)


        feature_names = self.get_feature_names(variant)
        self.input_dict["X_flat"] = pd.DataFrame(X, columns=feature_names)
        self.input_dict["feature_names"] = feature_names
        self.args.logger.write('ML flat matrix prepared. OLD')'''


class PreprocessorML(PreprocessorA): # same as A only we flatten the input

    def get_feature_names(self, variant):
        vars = self.variables
        T = self.args.T
        per_t = []
        if "V" in variant:
            per_t += [f"{v}_V" for v in vars]
        if "M" in variant:
            per_t += [f"{v}_M" for v in vars]
        if "D" in variant:
            per_t += [f"{v}_D" for v in vars]
        if self.args.feature_combination_method == "concatenate":
            ts_cols = [f"{col}_Bin{t}" for t in range(T) for col in per_t]
        else:
            ts_cols = per_t
        demo_cols = list(self.dataset.static_data.columns)
        return ts_cols + demo_cols

    def flatten(self, X_3d):
        if self.args.feature_combination_method == "concatenate":
            X_flat_ts = X_3d.reshape(X_3d.shape[0], -1)
        else:
            X_flat_ts = X_3d.mean(axis=1)
        return np.concatenate([X_flat_ts, self.dataset.demo], axis=1)

    def prepare_inputs(self):
        PreprocessorA.prepare_inputs(self)
        X = self.flatten(self.input_dict.pop("X"))
        if self.args.model_type == 'logistic_regression':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X[self.train_ind])
            X = scaler.transform(X)
            self.args.logger.write('Flat matrix standardised for logistic regression')
        feature_names = self.get_feature_names(self.args.variant)
        self.input_dict["X_flat"] = pd.DataFrame(X, columns=feature_names)
        self.input_dict["feature_names"] = feature_names
        self.args.logger.write('ML flat matrix prepared. USING new ML PREPROCESSOR')


'''class PreprocessorMLStats(PreprocessorML):

    STATS = ["mean", "std", "min", "max", "last", "count"]

    def trim(self): # original trim bin doesn't clip so we migh have partial bins but here I merge the partial bin to the last full bin
        PreprocessorML.trim(self)
        self.args.T = int(np.ceil(self.args.days_before_discharge * 24 / self.args.agg_int))
        self.data["int"] = self.data["int"].clip(upper=self.args.T - 1)
        self.args.logger.write(f'Time bins clipped to {self.args.T} x {self.args.agg_int/24:.1f} days, '
                               f'covering the {self.args.days_before_discharge} days before discharge')

    def prepare_inputs(self):
        self.set_variables()
        self.trim()


        #n_bins = getattr(self.args, "ml_stats_bins", 1)
        #window_minutes = self.args.days_before_discharge * 24 * 60
        #bin_width = window_minutes / n_bins
        #self.data = self.data.assign(
        #    stat_bin=np.minimum((self.data["minute"] // bin_width).astype(int), n_bins - 1)
        #)
        #self.args.logger.write('\nStats binning from ml_stats_bins: '+str(n_bins)+' bins of '+str(bin_width/1440)+' days')

        g = self.data.sort_values("minute").groupby(["ts_ind", "var_ind", "int"])["value"]
        stat_df = g.agg(["mean", "std", "min", "max", "count"])
        stat_df["last"] = g.last()
        stat_df = stat_df.reset_index()

        normed = {}
        filled_means = {}
        for b in range(self.args.T):
            bin_df = stat_df[stat_df["int"] == b]
            for stat in self.STATS:
                m = bin_df.pivot(index="ts_ind", columns="var_ind", values=stat)
                m = m.reindex(index=np.arange(self.args.N), columns=np.arange(self.args.V))
                m = m.to_numpy(dtype=float)

                if stat == "count":
                    m = np.nan_to_num(m, nan=0.0)

                col_mean = np.nanmean(m[self.train_ind], axis=0)
                col_mean = np.nan_to_num(col_mean, nan=0.0)
                col_std = np.nanstd(m[self.train_ind], axis=0)
                col_std = np.where(np.isnan(col_std) | (col_std == 0), 1.0, col_std)
                filled = np.where(np.isnan(m), col_mean, m)
                normed[(b, stat)] = (filled - col_mean) / col_std

                if stat == "mean":
                    filled_means[b] = filled

        blocks = [normed[(b, stat)] for b in range(self.args.T) for stat in self.STATS]
        block_names = [f"{v}_{stat}_bin{b}" for b in range(self.args.T) for stat in self.STATS for v in self.variables]

        if self.args.T > 1: #add across bins features 
            bin_means_stack = np.stack([filled_means[b] for b in range(self.args.T)], axis=0)
            bin_range = bin_means_stack.max(axis=0) - bin_means_stack.min(axis=0)
            col_mean = np.nanmean(bin_range[self.train_ind], axis=0)
            col_mean = np.nan_to_num(col_mean, nan=0.0)
            col_std = np.nanstd(bin_range[self.train_ind], axis=0)
            col_std = np.where(np.isnan(col_std) | (col_std == 0), 1.0, col_std)
            normed_bin_range = (bin_range - col_mean) / col_std
            blocks.append(normed_bin_range)
            block_names += [f"{v}_bin_range" for v in self.variables]
            self.args.logger.write(f'Between-bin variability features added: {len(self.variables)}')
            

        X_ts = np.concatenate(blocks, axis=1)
        X = np.concatenate([X_ts, self.dataset.demo], axis=1)

        feature_names = block_names + list(self.dataset.static_data.columns)

        self.input_dict = {"X_flat": pd.DataFrame(X, columns=feature_names)}
        self.input_dict["feature_names"] = feature_names
        self.args.logger.write(f'ML stats matrix prepared ({self.args.T} bin(s)). Shape: {X.shape}')'''


class PreprocessorStats(PreprocessorML):

    ML_MODELS = ['random_forest', 'logistic_regression', 'gradient_boosting', 'xgboost', 'catboost']
    STATS = ["mean", "std", "min", "max", "last", "count"]
    NORMALISE = True
    DEMO = True

    def trim(self):
        PreprocessorA.trim(self)
        self.args.T = int(np.ceil(self.args.days_before_discharge * 24 / self.args.agg_int))
        overflow = self.data["int"] > self.args.T - 1
        if overflow.any():
            extra = self.data["minute"].max() - self.args.T * 60 * self.args.agg_int
            self.args.logger.write(f'merge midnight-to-discharge to last bin. Merged tail: {int(overflow.sum())} rows, {extra/60:.1f} hours midnight to discharge')
        self.data["int"] = self.data["int"].clip(upper=self.args.T - 1) 
        self.args.logger.write(f'Time bins clipped to {self.args.T} x {self.args.agg_int/24:.1f} days, 'f'covering the {self.args.days_before_discharge} days before discharge')
        
    def stat_blocks(self):
        g = self.data.sort_values("minute").groupby(["ts_ind", "var_ind", "int"])["value"]
        stat_df = g.agg(["mean", "std", "min", "max", "count"])
        stat_df["last"] = g.last()
        stat_df = stat_df.reset_index()

        normed = {}
        for b in range(self.args.T):
            bin_df = stat_df[stat_df["int"] == b]
            train_bin = bin_df[bin_df["ts_ind"].isin(self.train_ind)]
            max_count = train_bin.groupby("var_ind")["count"].max()
            print(f"[dbg] bin {b} across all admissions: observed ({bin_df['ts_ind'].nunique()}, {bin_df['var_ind'].nunique()}) "
                  f"-> padded ({self.args.N}, {self.args.V}) | "
                  f"{self.args.V - len(max_count)} variables never measured in train, "
                  f"{int((max_count == 1).sum())} never measured twice in the same admission | "
                  f"nan {100 * (1 - len(bin_df) / (self.args.N * self.args.V)):.1f}%")
            for stat in self.STATS:
                m = bin_df.pivot(index="ts_ind", columns="var_ind", values=stat)
                m = m.reindex(index=np.arange(self.args.N), columns=np.arange(self.args.V))
                m = m.to_numpy(dtype=float)

                if stat == "count":
                    m = np.nan_to_num(m, nan=0.0)

                if not self.NORMALISE:
                    normed[(b, stat)] = np.nan_to_num(m, nan=0.0)
                    continue

                # normalize the STATS bin wise
                n_obs = (~np.isnan(m[self.train_ind])).sum(axis=0) # admissions with a value, per variable
                col_mean = np.nansum(m[self.train_ind], axis=0) / np.maximum(n_obs, 1) # 0 where no admission has that variable
                col_std = np.sqrt(np.nansum((m[self.train_ind] - col_mean) ** 2, axis=0) / np.maximum(n_obs, 1))
                col_std = np.where(col_std == 0, 1.0, col_std)
                filled = np.where(np.isnan(m), col_mean, m) # fill nan values with mean 
                normed[(b, stat)] = (filled - col_mean) / col_std # filled values become zeros 
  

        return normed

    def get_feature_names(self, variant):
        ts_cols = [f"{v}_{stat}_bin{b}" for b in range(self.args.T) for stat in self.STATS for v in self.variables]
        return ts_cols + (list(self.dataset.static_data.columns) if self.DEMO else [])

    def prepare_inputs(self):
        self.set_variables()
        self.trim()
        normed = self.stat_blocks()
        #self.args.logger.write("ohne ML trim")
        X_3d = np.stack([np.concatenate([normed[(b, stat)] for stat in self.STATS], axis=1) for b in range(self.args.T)], axis=1)
        print(f"[dbg] X_3d={X_3d.shape} zero_frac={(X_3d == 0).mean():.3f} nan_frac={np.isnan(X_3d).mean():.3f}")

        if self.args.model_type in self.ML_MODELS:
            X = np.concatenate([X_3d.reshape(X_3d.shape[0], -1)] + ([self.dataset.demo] if self.DEMO else []), axis=1)
            feature_names = self.get_feature_names(self.args.variant)
            print(f"[dbg] X={X.shape} names={len(feature_names)} "f"const_zero_cols={int((X == 0).all(axis=0).sum())}")
            
            self.input_dict = {"X_flat": pd.DataFrame(X, columns=feature_names)}
            self.input_dict["feature_names"] = feature_names
            self.args.logger.write(f'Feature matrix: {X.shape[0]} samples x {X.shape[1]} features = '
                                   f'{self.args.V} variables x {len(self.STATS)} stats x {self.args.T} bins '
                                   f'+ {self.dataset.demo.shape[1] if self.DEMO else 0} static')
        else:
            self.X = X_3d
            self.input_dict = {"X": X_3d}
            self.args.F = X_3d.shape[-1]
            self.args.logger.write(f'PreprocessorStats: TS stats matrix prepared ({self.args.T} bin(s)). '
                                   f'Shape: {X_3d.shape} (N, T, F={self.args.F} = {self.args.V} variables x {len(self.STATS)} stats)')


class PreprocessorCount(PreprocessorStats):

    STATS = ["count"]
    NORMALISE = False
    DEMO = False
