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
            
        # pickle dump only if not training in nested crossvalidation
        if not (self.args.cv_mode == "grid"): #self.args.grid == "nested" and 
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
        self.args.logger.write('Input prepared')


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




class PreprocessorML(Preprocessor):

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
        #self.normalise()
        #self.input_dict.update({"values_norm": self.values, "values_means": self.means, "values_stds": self.stds})
        
        
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
            

        X = np.concatenate([X_flat_ts, self.dataset.demo], axis=1) # already normalized demo and features
        '''if self.args.model_type == 'logistic_regression':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X[self.train_ind])
            X = scaler.transform(X)'''


        self.input_dict["X_flat"] = X
        self.input_dict["feature_names"] = self.get_feature_names(variant)
        self.args.logger.write('ML flat matrix prepared.')


