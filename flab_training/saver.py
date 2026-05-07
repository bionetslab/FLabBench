import os
import pandas as pd
from flab_training.utils import format_dict

class ResultSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def save(self, results):
        """Save results dictionary. Must be implemented by subclass."""
        raise NotImplementedError

    def _save_csv(self, df, filename):
        """Helper to save a DataFrame to CSV, appending if file exists."""
        filepath = os.path.join(self.output_dir, filename)
        file_exists = os.path.isfile(filepath)
        df.to_csv(filepath, mode='a', index=False, header=not file_exists)


class ResultSaverBest(ResultSaver):

    def save(self, results):
        # Save losses
        self._save_csv(pd.DataFrame(results["losses"]), "losses.csv")

        # Save metrics
        rows = []
        for split, res in results["final_metrics"].items():
            if res is None:
                continue
            row = {**results["args"], **results["param_info"], "split": split, **results["time_memory"], **res}
            rows.append(row)
        if rows:
            self._save_csv(pd.DataFrame(rows), "results_final.csv")


class ResultSaverGrid(ResultSaver):
    def save(self, results):
        # Prepare result row
        row = {
            "split": "val",
            "epoch": results["losses"]["epoch"][-1],  # last epoch
            "inner_fold": results["args"].get("inner_fold", None), # inner CV fold if available
            "search": results["args"].get("search", "custom"),
            **results["args"]["model_params"],
            **results["final_metrics"]["val"],
        }        

        df = pd.DataFrame([row])
        self._save_csv(df, "grid_results.csv")
        
        
        
        # save params to CSV and pickle
        #pd.DataFrame([vars(self.args)]).to_csv(os.path.join(self.output_dir, "params.csv"), index=False)
        #with open(os.path.join(self.output_dir, "params.pkl"), 'wb') as f:
        #    pickle.dump(self.args, f)