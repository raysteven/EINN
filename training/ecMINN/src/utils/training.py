from clearml import Task
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from src.nn_model.amn_qp import *
from src.schedulers.loss_aggregator import get_loss_aggregator
from src.schedulers.loss_balancer import LossBalancer, get_loss_balancer
from src.schedulers.weights_schedulers import get_loss_weight_scheduler
from src.utils.import_data import *
from src.utils.import_GEM import *
from src.utils.loggers import DataFrameLogger
from src.utils.utils import assign_hyperparams_to_confg, log_losses_balanced_to_clearml, log_losses_to_clearml, log_losses_with_l1_constant_to_clearml, log_weights_to_clearml, plot_modified_loss_with_aggregate

def check_nan(label, arr):
    if torch.is_tensor(arr):
        finite_mask = torch.isfinite(arr)
        if not finite_mask.all():
            print(f"NaN detected at {label}")
            raise ValueError(f"Non-finite values in {label}")
    else:
        if not np.all(np.isfinite(arr)):
            print(f"NaN detected at {label}")
            raise ValueError(f"Non-finite values in {label}")


def train_step(model, criterion, optimizer, train_loader, loss_balancer: LossBalancer=None, device='cpu'):
    model.train()
    loss_tot = 0
    len_train = 0
    Vref_pred = []
    Vref_true = []
    losses_n = np.zeros(4)
        
    # Lists to store loss for each batch
    batch_losses = []
    batch_component_losses = []

    for i, (x, Vref, Vin) in enumerate(train_loader):
        optimizer.zero_grad()
        # forward pass
        x, Vref, Vin = x.to(device), Vref.to(device), Vin.to(device)

        #check_nan("Vin inside train_step", Vin)
        #check_nan("Vref inside train_step", Vref)

        # V = Vref predicted
        V, V0 = model(x, Vref, Vin)
        #print("V:", V)
        #print("V0:", V0)
        for i in range(V.size(0)):
            Vref_pred.append(V[i].tolist())
            Vref_true.append(Vref[i].tolist())
        #print("Vref_pred:", Vref_pred)
        #print("Vref_true:", Vref_true)
        #check_nan("V inside train_step", V)
        # compute loss
        loss, losses = criterion(V, Vref, Vin)
        # back-prop
        loss.backward()

        # ADD GRADIENT CLIPPING HERE - This should fix the NaN explosion
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
        
        # Check for NaN gradients after clipping (optional debugging)
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name} after clipping!")

        optimizer.step()
        # gather statistics
        #loss_tot += loss.item()
            
        current_loss = loss.item()
        loss_tot += current_loss
        batch_losses.append(current_loss)

        #losses_n = losses_n + np.array([losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()])
        #print("losses_n:", losses_n)

        component_items = [losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()]
        losses_n = losses_n + np.array(component_items)
        batch_component_losses.append(component_items)

        len_train += x.size(0)
        
        if loss_balancer is not None:
            loss_balancer.balance_losses([losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()])
        
    #return {'loss': loss_tot/len_train, 'Vref_pred':Vref_pred, 'loss_tot':loss_tot,'losses': losses_n/len_train, 'Vref_true': Vref_true}
    return {'loss': loss_tot/len_train, 'Vref_pred':Vref_pred, 'loss_tot':loss_tot,'losses': losses_n/len_train, 'Vref_true': Vref_true, 'batch_losses':batch_losses, 'batch_component_losses':batch_component_losses}
# def train_step_debug(model, criterion, optimizer, train_loader, loss_balancer: LossBalancer=None, device='cpu'):
#     model.train()
#     loss_tot = 0
#     len_train = 0
#     Vref_pred = []
#     Vref_true = []
#     losses_n = np.zeros(4)
    
#     for batch_idx, (x, Vref, Vin) in enumerate(train_loader):
#         print(f"\n=== BATCH {batch_idx} ===")
#         optimizer.zero_grad()
        
#         # forward pass
#         x, Vref, Vin = x.to(device), Vref.to(device), Vin.to(device)

#         # Check inputs
#         print(f"Input x: finite={torch.isfinite(x).all()}, range=[{x.min():.6f}, {x.max():.6f}]")
#         print(f"Input Vref: finite={torch.isfinite(Vref).all()}, range=[{Vref.min():.6f}, {Vref.max():.6f}]")
#         print(f"Input Vin: finite values={torch.isfinite(Vin).sum().item()}/{Vin.numel()}")

#         # V = Vref predicted
#         V, V0 = model(x, Vref, Vin)
        
#         # Check model outputs
#         print(f"Model output V: finite={torch.isfinite(V).all()}, has_nan={torch.isnan(V).any()}")
#         if torch.isnan(V).any():
#             print("NaN detected in model output V - stopping here!")
#             return None
            
#         for i in range(V.size(0)):
#             Vref_pred.append(V[i].tolist())
#             Vref_true.append(Vref[i].tolist())
        
#         # compute loss - THIS IS WHERE NaN MIGHT ORIGINATE
#         print("Computing loss...")
#         loss, losses = criterion(V, Vref, Vin)
        
#         # Check loss values
#         print(f"Loss: {loss.item()}, finite={torch.isfinite(loss)}")
#         print(f"Individual losses: {[l.item() for l in losses]}")
#         print(f"Individual losses finite: {[torch.isfinite(l).all().item() for l in losses]}")
        
#         # DIAGNOSTIC CODE - Add this right after loss computation
#         print("\n=== Constraint Analysis ===")
#         # Get the constraint matrices from your model
#         #S = torch.from_numpy(np.float32(criterion.S)).to(device)
#         S = torch.as_tensor(criterion.S, device=V.device, dtype=V.dtype)
#         SV = torch.matmul(V, S.T)
        
#         print(f"SV (should be ~0): mean={SV.abs().mean():.6f}, max={SV.abs().max():.6f}")
#         print(f"SV range: [{SV.min():.6f}, {SV.max():.6f}]")
#         print(f"Negative fluxes: {(V < 0).sum().item()}/{V.numel()} ({100*(V < 0).float().mean():.1f}%)")
#         print(f"Most negative flux: {V.min():.6f}, Most positive: {V.max():.6f}")
#         print(f"S matrix shape: {S.shape}")
#         print(f"V matrix shape: {V.shape}")
#         print("=========================\n")
        
#         if not torch.isfinite(loss):
#             print("INFINITE/NaN LOSS DETECTED!")
#             print("Investigating loss components...")
            
#             # Debug the loss function components
#             # You might need to modify your criterion class to expose intermediate values
#             return None
        
#         # back-prop
#         print("Starting backward pass...")
#         loss.backward()
        
#         # Check gradients before clipping
#         print("Checking gradients before clipping...")
#         max_grad_norm = 0
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 grad_norm = param.grad.norm().item()
#                 max_grad_norm = max(max_grad_norm, grad_norm)
#                 if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
#                     print(f"NaN/Inf gradient in {name}! Grad norm: {grad_norm}")
        
#         print(f"Max gradient norm before clipping: {max_grad_norm}")
        
#         # Gradient clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
#         optimizer.step()
        
#         # Check model weights after update
#         print("Checking model weights after update...")
#         for name, param in model.named_parameters():
#             if torch.isnan(param).any() or torch.isinf(param).any():
#                 print(f"NaN/Inf in model weights {name} after update!")
#                 return None
        
#         # gather statistics
#         loss_tot += loss.item()
#         losses_n = losses_n + np.array([losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()])
#         len_train += x.size(0)
        
#         if loss_balancer is not None:
#             loss_balancer.balance_losses([losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()])
        
#         print(f"Batch {batch_idx} completed successfully")
        
#         # Stop after first batch for debugging
#         if batch_idx == 0:
#             print("Stopping after first batch for debugging")
#             breaktrans
        
#     return {'loss': loss_tot/len_train, 'Vref_pred':Vref_pred, 'losses': losses_n/len_train, 'Vref_true': Vref_true}

def test_step(model, criterion, test_loader, device='cpu'):
    model.eval()
    loss_tot = 0
    len_test = 0
    Vref_pred = []
    Vref_true = []
    Vin_all = []
    Vin_reservoir = []
    losses_n = np.zeros(4)
    with torch.no_grad():
        for i, (x, Vref, Vin) in enumerate(test_loader):
            # forward pass
            x, Vref, Vin = x.to(device), Vref.to(device), Vin.to(device)
            # V = Vref predicted
            V, V0 = model(x, Vref, Vin)
            for i in range(V.size(0)):
                Vref_pred.append(V[i].tolist())
                Vref_true.append(Vref[i].tolist())
                Vin_all.append(Vin[i].tolist())
                Vin_reservoir.append(V0[i].tolist())
            loss, losses = criterion(V, Vref, Vin)
            # gather statistics
            loss_tot += loss.item()
            losses_n = losses_n + np.array([losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()])
            len_test += x.size(0)
    return {'loss': loss_tot/len_test, 'Vref_pred':Vref_pred, 'loss_tot':loss_tot, 'losses': losses_n/len_test, 'Vref_true': Vref_true, 'Vin':Vin_all,
            'Vin_reservoir': Vin_reservoir}


# def test_step(model, criterion, test_loader, device='cpu', scaler=None):
#     model.eval()
#     loss_tot = 0
#     len_test = 0
#     Vref_pred = []
#     Vref_true = []
#     Vin_all = []
#     Vin_reservoir = []
#     losses_n = np.zeros(4)

#     # ---- pull Pref once (no API change to the caller) ----
#     if hasattr(criterion, "fit_model") and hasattr(criterion.fit_model, "Pref"):
#         Pref_np = criterion.fit_model.Pref
#     elif hasattr(criterion, "Pref"):
#         Pref_np = criterion.Pref
#     else:
#         raise ValueError("Pref matrix not found in criterion. Pass it or store it in criterion.fit_model.Pref.")
#     Pref_t = torch.as_tensor(Pref_np, dtype=torch.float32, device=device)

#     with torch.no_grad():
#         for i, (x, Vref, Vin) in enumerate(test_loader):
#             # forward pass
#             x, Vref, Vin = x.to(device), Vref.to(device), Vin.to(device)
#             V, V0 = model(x, Vref, Vin)

#             # ---- project to supervised target space ----
#             Vref_pred_batch = torch.matmul(V, Pref_t.T)  # shape: (batch, n_targets)

#             for j in range(V.size(0)):
#                 Vref_pred.append(Vref_pred_batch[j].tolist())  # <-- append projected preds
#                 Vref_true.append(Vref[j].tolist())             # these are (normalized) targets
#                 Vin_all.append(Vin[j].tolist())
#                 Vin_reservoir.append(V0[j].tolist())

#             loss, losses = criterion(V, Vref, Vin)
#             # gather statistics
#             loss_tot += loss.item()
#             losses_n = losses_n + np.array([
#                 losses[0].item(), losses[1].item(), losses[2].item(), losses[3].item()
#             ])
#             len_test += x.size(0)

#     # ---- inverse transform ONLY the target space ----
#     Vref_pred = np.array(Vref_pred)
#     Vref_true = np.array(Vref_true)
#     if scaler is not None:
#         Vref_pred = scaler.inverse_transform(Vref_pred)
#         Vref_true = scaler.inverse_transform(Vref_true)

#     return {
#         'loss': loss_tot/len_test,
#         'Vref_pred': Vref_pred,
#         'losses': losses_n/len_test,
#         'Vref_true': Vref_true,
#         'Vin': Vin_all,
#         'Vin_reservoir': Vin_reservoir
#     }



# kfold evaluation
def kfold_evaluation(cfg, task: Task, X, y, Vin, n_distribution, fit_model, params, loo_count=None):
    # Initialize KFold
    kf = KFold(n_splits=cfg.hpo.n_fold, shuffle=True, random_state=cfg.seed)

    # Lists to store results for each fold
    fold_losses = []

    # Perform k-fold cross-validation
    for fold, (train_idx_fold, val_idx_fold) in enumerate(kf.split(X, y)):

        # Split data into training and validation sets
        X_train_fold, X_val_fold = X[train_idx_fold], X[val_idx_fold]
        y_train_fold, y_val_fold = y[train_idx_fold], y[val_idx_fold]
        Vin_train_fold, Vin_val_fold = Vin[train_idx_fold], Vin[val_idx_fold]

        # Train and evaluate model
        results = train_test_evaluation(cfg, task, X_train_fold, X_val_fold, y_train_fold, y_val_fold, Vin_train_fold, Vin_val_fold, n_distribution, fit_model, params, fold_id=fold, loo_count=loo_count)
        
        #RMSE_value_fold = np.sqrt(np.mean(np.square(Vref_true-Vref_pred_te)))
        #print(f'Fold_{fold} TRAIN losses: {results["train"]["losses"]}')
        #print(f'Fold_{fold} TEST losses: {results["test"]["losses"]}')
        fold_losses.append(results["test"]['losses'][0])

    return fold_losses

def train_test_evaluation(cfg, task: Task, X_train, X_test, y_train, y_test, Vin_train, Vin_test, n_distribution, fit_model, params, fold_id=None, loo_count=None):
    
    assign_hyperparams_to_confg(cfg, params)

    # # --- 🔹 Fit scaler on TRAIN y (fluxes) ---
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # y_train = scaler.fit_transform(y_train)
    # y_test  = scaler.transform(y_test)

    # Save scaler into config so you can inverse-transform later during evaluation
    #cfg.model.scaler = scaler

    # Initialize AMN_QP module
    if cfg.model.model_reservoir:
        
        # amn_qp = AMN_QP_reservoir(input_size=X_train.shape[1], hidden_size=params['hidden_size'], output_size=5, 
        #             drop_rate= params['drop_rate'], hyper_params = cfg, model=fit_model).to(cfg.model.device)
        amn_qp = MINN_reservoir(input_size=X_train.shape[1], hidden_size=params['hidden_size'], output_size=5, 
                    drop_rate= params['drop_rate'], hyper_params = cfg, model=fit_model).to(cfg.model.device)
    else:
        #  amn_qp = MINN(input_size=X_train.shape[1], output_size=n_distribution, 
        #              drop_rate= params['drop_rate'], hyper_params = cfg, model=fit_model,
        #                  hidden_size1=1000, hidden_size2=1000,).to(cfg.model.device)
        amn_qp = MINN(input_size=X_train.shape[1], hidden_size=params['hidden_size'], output_size=n_distribution, 
                   drop_rate= params['drop_rate'], hyper_params = cfg, model=fit_model).to(cfg.model.device) #, depth=params['depth']
        # amn_qp = MINN_Scaled(input_size=X_train.shape[1], hidden_size=params['hidden_size'], output_size=n_distribution, 
        #             drop_rate= params['drop_rate'], hyper_params = cfg, model=fit_model).to(cfg.model.device) 
           
    
    loss_aggregator = get_loss_aggregator(cfg.loss_aggregator)
    if cfg.loss_aggregator.type == "double_linear_bound" and loo_count == 1 and fold_id is None:
        plot_modified_loss_with_aggregate(task, loss_aggregator)
        
    #criterion = MechanisticLossWeighted(fit_model, params['l1_constant'], cfg, loss_aggregator=loss_aggregator)
    criterion = MechanisticLossWeighted(
        fit_model,
        params['l1_constant'],
        cfg,
        loss_aggregator=loss_aggregator,
        biomass_rxn_id="R_BIOMASS_Ec_iAF1260_core_59p81M",  # <-- put your biomass ID here
        biomass_weight= 40                      # <-- strength of emphasis
    )



    # Define optimizer for backpropagation
    optimizer = torch.optim.Adam(amn_qp.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    # Also check the raw X before scaling
    #print("X_train raw stats before scaling:")
    #print("Min:", X_train.min(), "Max:", X_train.max(), "Mean:", X_train.mean())

    # # Basic Data Preprocessing
    # sc_X_fold = MinMaxScaler()
    # #sc_y_fold = MinMaxScaler()
    # X_train = sc_X_fold.fit_transform(X_train)
    # #y_train_fold = sc_y_fold.fit_transform(y_train_fold)
    # X_test = sc_X_fold.transform(X_test)
    # #y_val_fold = sc_y_fold.transform(y_val_fold)

    # Add this right before creating the train_loader
    #print("X_train stats after scaling:")
    #print("Min:", X_train.min(), "Max:", X_train.max(), "Mean:", X_train.mean())
    #print("NaN in X_train:", np.isnan(X_train).any())
    #print("Inf in X_train:", np.isinf(X_train).any())

    #check_nan("Vin train inside train_test_eval", Vin_train)

    # Create data loaders for the training and validation sets
    train_dataset = CustomTensorDataset(data=(X_train, y_train, Vin_train))
    valid_dataset = CustomTensorDataset(data=(X_test, y_test, Vin_test))

    # def debug_forward(model, x, Vref, Vin):
    #     hooks = []
    #     def hook_fn(module, input, output):
    #         if torch.is_tensor(output):
    #             if not torch.isfinite(output).all():
    #                 print(f"NaN detected in {module.__class__.__name__}")
    #         elif isinstance(output, (list, tuple)):
    #             for idx, o in enumerate(output):
    #                 if torch.is_tensor(o) and not torch.isfinite(o).all():
    #                     print(f"NaN detected in {module.__class__.__name__} output {idx}")

    #     for layer in model.modules():
    #         if len(list(layer.children())) == 0:  # leaf layer
    #             hooks.append(layer.register_forward_hook(hook_fn))
        
    #     # run a single forward pass
    #     with torch.no_grad():
    #         model.eval()
    #         model(x, Vref, Vin)

    #     for h in hooks:
    #         h.remove()



    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.hpo.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.hpo.batch_size)

    # best epoch is chosen in based on the first fold metric
    loss_balancer = get_loss_balancer(cfg.loss_balancer, criterion)
    loss_weight_scheduler = get_loss_weight_scheduler(cfg.loss_weight_scheduler, cfg.hpo.epochs, criterion)
    
    for epoch in range(cfg.hpo.epochs):
        # update epoch and loss weights
        loss_weight_scheduler.update_current_epoch(epoch)
        loss_weight_scheduler.adjust_weights()
        
        tr_loss = train_step(amn_qp, criterion, optimizer, train_loader, loss_balancer=loss_balancer, device=cfg.model.device)
        # tr_loss = train_step_debug(amn_qp, criterion, optimizer, train_loader, loss_balancer=loss_balancer, device=cfg.model.device)
        
        if task != None:
            if fold_id is None:
                # log losses and weights to clearml
                log_losses_to_clearml(task, epoch, tr_loss["losses"], loo_counter=loo_count)
                if criterion.l1_constant != 1:
                    log_losses_with_l1_constant_to_clearml(task, epoch, criterion.l1_constant, tr_loss["losses"], loo_counter=loo_count)
                if cfg.loss_balancer.type != 'no_balancer':
                    log_losses_balanced_to_clearml(task, epoch, loss_balancer, tr_loss["losses"], loo_counter=loo_count)
                log_weights_to_clearml(task, epoch, loss_weight_scheduler.get_weights(), loo_counter=loo_count)
            
    te_loss = test_step(amn_qp, criterion, valid_loader, device=cfg.model.device) #, scaler=scaler
    
    return {"train": tr_loss, "test": te_loss, "model": amn_qp}