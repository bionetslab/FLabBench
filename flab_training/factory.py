import ast
import inspect
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier

from flab_training.ts_models.ts_gru import GRU_TS
from flab_training.ts_models.ts_sand import SAND_TS
from flab_training.ts_models.ts_tcn import TCN_TS
from flab_training.ts_models.ts_lstm import LSTM_TS
from flab_training.ts_models.ts_grud import GRUD_TS
from flab_training.ts_models.ts_interpnet import INTERPNET_TS
from flab_training.ts_models.ts_strats import STRATS_TS
from flab_training.ts_models.mlp import MLP
from flab_training.ts_models.ts_emit import EMIT_TS

from flab_training.batcher import Batcher, BatcherA, BatcherB, BatcherC_sup, BatcherC_unsup, BatcherD_unsup, BatcherD_unsup_fixed

TS_MODEL_CLASSES = {
    'gru': GRU_TS,
    'tcn': TCN_TS,
    'lstm': LSTM_TS,
    'sand': SAND_TS,
    'grud': GRUD_TS,
    'interpnet': INTERPNET_TS,
    'strats': STRATS_TS,
    'emit': EMIT_TS,
    'mlp': MLP,
}

CL_MODEL_CLASSES = {
    'random_forest': RandomForestClassifier,
    'logistic_regression': LogisticRegression,
    'gradient_boosting': HistGradientBoostingClassifier,
    'xgboost': xgb.XGBClassifier,
    'catboost': CatBoostClassifier,
}

CL_MODELS = list(CL_MODEL_CLASSES.keys())

def build_ts_model(args):
    model_cls = TS_MODEL_CLASSES.get(args.model_type.lower())
    if model_cls is None:
        raise ValueError(
            f"Unknown model type '{args.model_type}'. "
            f"Available: {list(TS_MODEL_CLASSES)}"
        )
    model = model_cls(args)
    return model.to(args.device)

def build_cl_model(args):
    model_cls = CL_MODEL_CLASSES.get(args.model_type)
    if model_cls is None:
        raise ValueError(
            f"Unknown model type '{args.model_type}'. "
            f"Available: {list(CL_MODEL_CLASSES)}"
        )
    valid_keys = set(inspect.signature(model_cls).parameters)
    def _parse(v):
        if isinstance(v, str) and v.startswith('{'):
            return ast.literal_eval(v)
        if isinstance(v, float) and v == int(v):
            return int(v)
        return v
    params = {k: _parse(v) for k, v in dict(args.model_params).items() if k in valid_keys}
    params['random_state'] = args.seed
    return model_cls(**params)

def build_batcher(args, input_dict):
    # get batcher based on model
    model_type = args.model_type
    if model_type in ['gru', 'lstm', 'tcn', 'sand', 'mlp']:
        batcher = BatcherA(args, input_dict)
    elif model_type in ['grud', 'interpnet']:
        batcher = BatcherB(args, input_dict)
    elif model_type in ['strats', 'istrats'] and args.train_mode == "pretrain":
        batcher = BatcherC_unsup(args, input_dict)
    elif model_type in ['strats', 'istrats'] and args.train_mode != "pretrain":
        batcher = BatcherC_sup(args, input_dict)
    elif model_type in ['emit'] and args.train_mode == "pretrain":
        # Choose EMIT batcher based on windowing mode
        use_fixed_windows = getattr(args, 'emit_fixed_windows', False)
        if use_fixed_windows:
            batcher = BatcherD_unsup_fixed(args, input_dict)
            args.logger.write('Using BatcherD_unsup_fixed (original EMIT fixed windows)')
        else:
            batcher = BatcherD_unsup(args, input_dict)
            args.logger.write('Using BatcherD_unsup (STRATS-style dynamic windows)')
    elif model_type in ['emit'] and args.train_mode != "pretrain":
        batcher = BatcherC_sup(args, input_dict)  # Reuse STRATS supervised batcher for EMIT finetuning
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    args.logger.write('\nBatching module assigned.')
    return batcher