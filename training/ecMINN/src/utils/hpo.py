# inner loop for the k-fold cross validation
from functools import partial
import numpy as np
from omegaconf import DictConfig
import optuna

from src.utils.training import kfold_evaluation


def objective(trial: optuna.Trial, cfg, task, X_train, y_train, Vin_train, n_distribution, fit_model):
    # Define hyperparameters to optimize
    if cfg.hpo.hpo_name == "hpo_non_cathegorical":
        params = {'hidden_size': trial.suggest_int('hidden_size', **cfg.hpo.search_space.hidden_size),
                    'drop_rate': trial.suggest_float('drop_rate',  **cfg.hpo.search_space.drop_rate),
                    'learning_rate': trial.suggest_float('learning_rate', **cfg.hpo.search_space.learning_rate),
                    'weight_decay': trial.suggest_float('weight_decay', **cfg.hpo.search_space.weight_decay)}
        
        if "l1_constant" in cfg.hpo.search_space.keys():
            params.update({'l1_constant': trial.suggest_float('l1_constant', **cfg.hpo.search_space.l1_constant)})
        if "data_driven_weight" in cfg.hpo.search_space.keys():
            data_driven_weight = trial.suggest_float('data_driven_weight', **cfg.hpo.search_space.data_driven_weight)
            mechanistic_weight = 1 - data_driven_weight
            params.update({'data_driven_weight': data_driven_weight, 'mechanistic_weight': mechanistic_weight})
        if "mechanistic_bound" in cfg.hpo.search_space.keys():
            params.update({'mechanistic_bound': trial.suggest_float('mechanistic_bound', **cfg.hpo.search_space.mechanistic_bound)})
            
    elif cfg.hpo.hpo_name == "hpo_ecMINN_reservoir":  
        params = {
            #'depth': trial.suggest_categorical('depth', cfg.hpo.depth),
            'l1_constant': trial.suggest_int('l1_constant', **cfg.hpo.search_space.l1_constant),
            
            # rest are fixed from cfg
            'hidden_size': trial.suggest_categorical('hidden_size', cfg.hpo.hidden_size),
            'drop_rate': trial.suggest_categorical('drop_rate',  cfg.hpo.drop_rate),
            'learning_rate': trial.suggest_categorical('learning_rate', cfg.hpo.learning_rate),
            'weight_decay': trial.suggest_categorical('weight_decay', cfg.hpo.weight_decay)}

    elif cfg.hpo.hpo_name == "hpo_depth_hidden":  
        params = {
            'depth': trial.suggest_categorical('depth', cfg.hpo.depth),
            'hidden_size': trial.suggest_categorical('hidden_size', cfg.hpo.hidden_size),
            # rest are fixed from cfg
            'drop_rate': trial.suggest_categorical('drop_rate',  cfg.hpo.drop_rate),
            'learning_rate': trial.suggest_categorical('learning_rate', cfg.hpo.learning_rate),
            'weight_decay': trial.suggest_categorical('weight_decay', cfg.hpo.weight_decay),
            'l1_constant': trial.suggest_categorical('l1_constant', cfg.hpo.l1_constant)}
    else:
        params = {'hidden_size': trial.suggest_categorical('hidden_size', cfg.hpo.hidden_size),
                    'drop_rate': trial.suggest_categorical('drop_rate',  cfg.hpo.drop_rate),
                    'learning_rate': trial.suggest_categorical('learning_rate', cfg.hpo.learning_rate),
                    'weight_decay': trial.suggest_categorical('weight_decay', cfg.hpo.weight_decay),
                    'l1_constant': trial.suggest_categorical('l1_constant', cfg.hpo.l1_constant)}
        

    if 'l1_constant' not in params.keys():
        params['l1_constant'] = 1
        
    
    fold_losses = kfold_evaluation(cfg, task, X_train, y_train, Vin_train, n_distribution, fit_model, params)

    return np.mean(fold_losses)

# def objective(trial: optuna.Trial, cfg, task, X_train, y_train, Vin_train, n_distribution, fit_model):
#     # Define hyperparameters to optimize
#     if cfg.hpo.hpo_name == "hpo_non_cathegorical":
#         params = {
#             'hidden_size': trial.suggest_int('hidden_size', **cfg.hpo.search_space.hidden_size),
#             'drop_rate': trial.suggest_float('drop_rate',  **cfg.hpo.search_space.drop_rate),
#             'learning_rate': trial.suggest_float('learning_rate', **cfg.hpo.search_space.learning_rate),
#             'weight_decay': trial.suggest_float('weight_decay', **cfg.hpo.search_space.weight_decay)
#         }
        
#         if "l1_constant" in cfg.hpo.search_space.keys():
#             params.update({'l1_constant': trial.suggest_float('l1_constant', **cfg.hpo.search_space.l1_constant)})
#         if "data_driven_weight" in cfg.hpo.search_space.keys():
#             data_driven_weight = trial.suggest_float('data_driven_weight', **cfg.hpo.search_space.data_driven_weight)
#             mechanistic_weight = 1 - data_driven_weight
#             params.update({'data_driven_weight': data_driven_weight, 'mechanistic_weight': mechanistic_weight})
#         if "mechanistic_bound" in cfg.hpo.search_space.keys():
#             params.update({'mechanistic_bound': trial.suggest_float('mechanistic_bound', **cfg.hpo.search_space.mechanistic_bound)})

#     elif cfg.hpo.hpo_name == "hpo_baseline":
#         params = {
#             'hidden_size': trial.suggest_categorical('hidden_size', cfg.hpo.hidden_size),
#             'drop_rate': trial.suggest_categorical('drop_rate',  cfg.hpo.drop_rate),
#             'learning_rate': trial.suggest_categorical('learning_rate', cfg.hpo.learning_rate),
#             'weight_decay': trial.suggest_categorical('weight_decay', cfg.hpo.weight_decay),
#             'l1_constant': trial.suggest_categorical('l1_constant', cfg.hpo.l1_constant)
#         }

#     elif cfg.hpo.hpo_name == "hpo_depth_hidden":  
#         params = {
#             'depth': trial.suggest_int('depth', 1, 6),
#             'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512, 1024, 2048]),
#             # rest are fixed from cfg
#             'drop_rate': cfg.model.drop_rate,
#             'learning_rate': cfg.training.lr,
#             'weight_decay': cfg.training.weight_decay,
#             'l1_constant': getattr(cfg.model, 'l1_constant', 50)
#         }

#     if 'l1_constant' not in params.keys():
#         params['l1_constant'] = 1

#     fold_losses = kfold_evaluation(cfg, task, X_train, y_train, Vin_train, n_distribution, fit_model, params)

#     return np.mean(fold_losses)



# def hpo(cfg: DictConfig, **objective_args):
#     # Create an Optuna study
#     sampler = optuna.samplers.TPESampler(seed=cfg.seed)
#     study = optuna.create_study(direction='minimize', sampler=sampler)
#     # You can adjust the number of trials
        
#     objective_func = partial(objective, cfg=cfg, **objective_args)

#     study.optimize(objective_func, n_jobs=1, n_trials=cfg.hpo.max_trials)

#     # Get the best hyperparameters
#     return study.best_params

import json
import os
import optuna

def hpo(cfg: DictConfig, **objective_args):
    # Use SQLite so Optuna dashboard can connect
    storage_url = "sqlite:///optuna_study.db"
    sampler = optuna.samplers.TPESampler(seed=cfg.seed)


    # depth_vals = [2,3,4,5,6]
    # hidden_vals = [512, 1024, 2048]
    # search_space = {
    #        "depth": depth_vals,
    #        "hidden_size": hidden_vals
    #          }
    # sampler = optuna.samplers.GridSampler(search_space)

    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True
    )
    
    objective_func = partial(objective, cfg=cfg, **objective_args)

    study.optimize(objective_func, n_jobs=1, n_trials=cfg.hpo.max_trials)

    # Save best params to JSON
    best_params = study.best_params
    os.makedirs("hpo_results_reservoir", exist_ok=True)
    with open("hpo_results_reservoir/best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    return best_params
