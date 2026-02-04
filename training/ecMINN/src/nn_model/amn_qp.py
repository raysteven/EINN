import numpy as np
import numpy as np
import torch.nn as nn
import torch
from src.nn_model.amn_qp_old_code import *
from src.utils.import_GEM import *
import pandas as pd

from src.schedulers.loss_aggregator import LossAggregator

def normalized_error(true, pred, axis):
    error_norm = torch.norm(true - pred, dim=axis)
    true_norm = torch.norm(true, dim=axis)
    normalized_error = torch.nan_to_num(error_norm / true_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized_error

def get_loss(V, Vref, Pref, Pin, Vin, S, gradient=False, hyper_params=None, loss_aggregator: LossAggregator = None, l1_constant=None, data_driven_weight=1, mechanistic_weight=1):
    
    #2nd element of the loss (steady-state costraint)cle
    S = torch.from_numpy(np.float32(S)).to(hyper_params.model.device)
    #print("S: ", S)
    SV = torch.matmul(V, S.T)
    #print("SV: ", SV)
    L2 = (torch.norm(SV, dim=1, p=2)**2)/S.shape[0]**2
    #L2 = (torch.norm(SV, dim=1, p=2) / S.shape[0])**2


    #3rd element of the loss (upper bound costraint)
    Pin = torch.from_numpy(np.float32(Pin)).to(hyper_params.model.device)
    #Vin = torch.from_numpy(np.float32(Vin)).to(args.device)
    Vin_pred = torch.matmul(V, Pin.T)

    # print("Pin Shape:", Pin.shape)
    # print("Pin: \n", Pin)
    
    # print("Vin_pred Shape:", Vin_pred.shape)
    # print("Vin_pred: \n", Vin_pred)

    # print("Vin Shape:", Vin.shape)
    # print("Vin: \n", Vin)
    

    relu_Vin= torch.relu(Vin_pred - Vin)
    L3 = (torch.norm(relu_Vin, dim=1, p=2)**2)/Pin.shape[0] #**2

    #L3 = (torch.norm(relu_Vin, dim=1, p=2) / Pin.shape[0])**2



    #4rd element of the loss (flux positivity costraint)
    relu_V = torch.relu(-V)
    L4 = (torch.norm(relu_V, dim=1, p=2)**2)/V.shape[1]**2

    #L4 = (torch.norm(relu_V, dim=1, p=2) / V.shape[1])**2

    if gradient == False:
        #print("gradient ==>", gradient)
        #1st element of the loss (to fit reference fluxes)
        Pref = torch.from_numpy(np.float32(Pref)).to(hyper_params.model.device)
        #print('Pref shape: ', Pref.shape)
        Vref_pred = torch.matmul(V, Pref.T)
        if hyper_params.model.l1_type=='MSE':
            L1 = (torch.norm((Vref_pred - Vref), dim=1, p=2)**2)/Pref.shape[0]
        elif hyper_params.model.l1_type == 'NE':
            L1= normalized_error(Vref, Vref_pred, axis=1)
        # Replace the MSE branch with SmoothL1 (Huber) as an option
        elif hyper_params.model.l1_type == 'SmoothL1':
            diff = Vref_pred - Vref
            # beta = 1.0 (delta) is fine; tune if needed
            L1 = torch.nn.functional.smooth_l1_loss(
                Vref_pred, Vref, reduction='none', beta=0.1
            ).mean(dim=1)  # per-sample mean over targets
            

        #TODO: da capire la normalizzazione delle loss se vogliamo usare un'altra loss rispetto a MSE
        #if hyper_params.model.model_name == 'amn_qp':
        #L = l1_constant*(L1) + (1-l1_constant)*(L2+L3+L4)
        if hyper_params.model.model_name in ['amn_qp', 'amn_qp_MSE']:
            if loss_aggregator is not None:
                L = loss_aggregator.aggregate_losses(losses=[L1, L2+L3+L4], weights=[data_driven_weight*l1_constant, mechanistic_weight])
            else:
                L = data_driven_weight*l1_constant*(L1) + mechanistic_weight*(L2+L3+L4)
                #L = data_driven_weight * L1 + (1.0 - data_driven_weight) * ((1* L2) + (1 * L3) + (1 * L4)) #/ 3.0
        else:
            L=L1

        #print(f"L1: {L1.mean().item():.6f}, L2: {L2.mean().item():.6f}, L3: {L3.mean().item():.6f}, L4: {L4.mean().item():.6f}")


    # when used by QP to refine the solution
    if gradient ==True:
        # ADD THIS SECTION
        # 1. Re-calculate Vref_pred inside the loop
        #Pref = torch.from_numpy(np.float32(Pref)).to(hyper_params.model.device)
        #Vref_pred = torch.matmul(V, Pref.T)
        
        # 2. Calculate the data-driven gradient, dV1
        # The gradient of (Vref_pred - Vref)^2 w.r.t V
        #dV1 = torch.matmul(Vref_pred - Vref, Pref) * (2 / Pref.shape[0])
        # END OF ADDED SECTION
    
    
        #print("gradient ==>", gradient)
        #TODO way to do this automatically
        # old MM_Qp_solver
        #dV1 = torch.matmul(Veb_pred - Veb, Peb)*(2/Pref.shape[0])
        dV2 = torch.matmul(SV, S)*(2/S.shape[0])

        dV3 = torch.where(relu_Vin != 0, 1, torch.zeros_like(relu_Vin))
        dV3 = (relu_Vin*dV3)
        dV3 = torch.matmul(dV3, Pin)*(2/Pin.shape[0]) 

        dV4 = torch.where(relu_V != 0, 1, torch.zeros_like(relu_V))
        dV4 = (relu_V*dV4)*(-2/V.shape[1])

        dV = dV2+dV3+dV4
        #refinement_data_weight = 0.2
        #dV = refinement_data_weight * dV1 + (1 - refinement_data_weight) * (dV2 + dV3 + dV4)
        #print(f"L1: {L1.mean().item():.6f}, L2: {L2.mean().item():.6f}, L3: {L3.mean().item():.6f}, L4: {L4.mean().item():.6f}")
        
        return dV
    

    # when used to backpropagate the error through the NN
    else:
        #print(f"L1: {L1.mean().item():.6f}, L2: {L2.mean().item():.6f}, L3: {L3.mean().item():.6f}, L4: {L4.mean().item():.6f}")
        return L, [L1.sum(), L2.sum(), L3.sum(), L4.sum()]
    
# def get_loss_safe(V, Vref, Pref, Pin, Vin, S, gradient=False, hyper_params=None, loss_aggregator: LossAggregator = None, l1_constant=None, data_driven_weight=1, mechanistic_weight=1):
#     import torch, numpy as np

#     #2nd element of the loss (steady-state constraint)
#     S = torch.from_numpy(np.float32(S)).to(hyper_params.model.device)

#     # S: torch tensor shape (m, r)
#     # V_pred: torch tensor (batch, r) or (r,)
#     # ensure no grad tracking on S
#     print("S dtype, device, requires_grad:", S.dtype, S.device, S.requires_grad)

#     # basic checks
#     print("S finite all:", torch.isfinite(S).all().item())
#     print("S max/min:", S.max().item(), S.min().item())

#     # per-row and per-column norms
#     row_norms = torch.norm(S, dim=1)   # shape (m,)
#     col_norms = torch.norm(S, dim=0)   # shape (r,)
#     print("row_norms max/min/mean:", row_norms.max().item(), row_norms.min().item(), row_norms.mean().item())
#     print("col_norms max/min/mean:", col_norms.max().item(), col_norms.min().item(), col_norms.mean().item())

#     # find extreme rows/cols
#     row_thresh = float(row_norms.max().item()) * 1e-3  # or set absolute threshold like 1e6
#     col_thresh = float(col_norms.max().item()) * 1e-3

#     extreme_rows = torch.where(row_norms > row_thresh)[0].cpu().numpy()
#     extreme_cols = torch.where(col_norms > col_thresh)[0].cpu().numpy()
#     print("num extreme rows:", len(extreme_rows), "example:", extreme_rows[:10])
#     print("num extreme cols:", len(extreme_cols), "example:", extreme_cols[:10])

#     # show top offending entries by absolute value
#     absS = torch.abs(S)
#     vals, idx = torch.topk(absS.view(-1), k=20)
#     rows = (idx // S.size(1)).cpu().numpy()
#     cols = (idx % S.size(1)).cpu().numpy()
#     for v, r, c in zip(vals.cpu().numpy(), rows, cols):
#         print(f"S[{r},{c}] = {v:e}")

#     SV = torch.matmul(V, S.T)
#     print("SV.max:", SV.max())
#     print("SV.min:", SV.min())
#     print("SV.mean:", SV.mean())
#     print("SV.std:", SV.std())
#     (torch.isfinite(S).all())/(torch.isfinite(SV).all())
#     # SAFE L2 computation with clipping to prevent overflow
#     SV_norm = torch.norm(SV, dim=1, p=2)
#     SV_norm_clipped = torch.clamp(SV_norm, max=1e6)  # Prevent overflow
#     L2 = (SV_norm_clipped**2)/S.shape[0]

#     #3rd element of the loss (upper bound constraint)
#     Pin = torch.from_numpy(np.float32(Pin)).to(hyper_params.model.device)
#     Vin_pred = torch.matmul(V, Pin.T)
#     relu_Vin = torch.relu(Vin_pred - Vin)
    
#     # SAFE L3 computation
#     relu_Vin_norm = torch.norm(relu_Vin, dim=1, p=2)
#     relu_Vin_norm_clipped = torch.clamp(relu_Vin_norm, max=1e6)
#     L3 = (relu_Vin_norm_clipped**2)/Pin.shape[0]

#     #4rd element of the loss (flux positivity constraint)
#     relu_V = torch.relu(-V)
    
#     # SAFE L4 computation with clipping
#     relu_V_norm = torch.norm(relu_V, dim=1, p=2)
#     relu_V_norm_clipped = torch.clamp(relu_V_norm, max=1e6)  # Prevent overflow
#     L4 = (relu_V_norm_clipped**2)/V.shape[1]

#     if gradient == False:
#         #1st element of the loss (to fit reference fluxes)
#         Pref = torch.from_numpy(np.float32(Pref)).to(hyper_params.model.device)
#         Vref_pred = torch.matmul(V, Pref.T)
#         if hyper_params.model.l1_type=='MSE':
#             L1_norm = torch.norm((Vref_pred - Vref), dim=1, p=2)
#             L1_norm_clipped = torch.clamp(L1_norm, max=1e6)
#             L1 = (L1_norm_clipped**2)/Pref.shape[0]
#         elif hyper_params.model.l1_type == 'NE':
#             L1= normalized_error(Vref, Vref_pred, axis=1)

#         # Debug prints to see which losses are causing issues
#         print(f"L1: {L1.mean().item():.6f}, L2: {L2.mean().item():.6f}, L3: {L3.mean().item():.6f}, L4: {L4.mean().item():.6f}")

#         if hyper_params.model.model_name in ['amn_qp', 'amn_qp_MSE']:
#             if loss_aggregator is not None:
#                 L = loss_aggregator.aggregate_losses(losses=[L1, L2+L3+L4], weights=[data_driven_weight*l1_constant, mechanistic_weight])
#             else:
#                 L = data_driven_weight*l1_constant*(L1) + mechanistic_weight*(L2+L3+L4)
#         else:
#             L=L1

#     # when used by QP to refine the solution
#     if gradient ==True:
#         dV2 = torch.matmul(SV, S)*(2/S.shape[0])

#         dV3 = torch.where(relu_Vin != 0, 1, torch.zeros_like(relu_Vin))
#         dV3 = (relu_Vin*dV3)
#         dV3 = torch.matmul(dV3, Pin)*(2/Pin.shape[0]) 

#         dV4 = torch.where(relu_V != 0, 1, torch.zeros_like(relu_V))
#         dV4 = (relu_V*dV4)*(-2/V.shape[1])

#         dV = dV2+dV3+dV4
        
#         return dV
    
#     # when used to backpropagate the error through the NN
#     else:
#         return L, [L1.sum(), L2.sum(), L3.sum(), L4.sum()]

# def get_loss_safe(
#     V, Vref, Pref, Pin, Vin, S,
#     gradient=False,
#     hyper_params=None,
#     loss_aggregator=None,
#     l1_constant=None,
#     data_driven_weight=1.0,
#     mechanistic_weight=1.0,
#     row_drop_quantile=None  # e.g., 0.995 to drop top 0.5% heaviest rows
# ):
#     import torch

#     device = hyper_params.model.device
#     dtype  = V.dtype  # match model output dtype

#     # --- Ensure tensors are on same dtype/device as V ---
#     V      = V.to(device=device, dtype=dtype)
#     Vref_t = torch.as_tensor(Vref, device=device, dtype=dtype)
#     Pref_t = torch.as_tensor(Pref, device=device, dtype=dtype)
#     Pin_t  = torch.as_tensor(Pin,  device=device, dtype=dtype)
#     Vin_t  = torch.as_tensor(Vin,  device=device, dtype=dtype)
#     S_t    = torch.as_tensor(S,    device=device, dtype=dtype)

#     # --- Column normalization (float64 for stability) ---
#     S64 = S_t.to(torch.float64)
#     col_norms = torch.linalg.norm(S64, dim=0)
#     col_norms[col_norms == 0] = 1.0
#     Dinv = 1.0 / col_norms
#     Stilde = S64 * Dinv

#     # Optional row masking for heavy rows
#     if row_drop_quantile is not None:
#         row_norms = torch.linalg.norm(Stilde, dim=1)
#         q = torch.quantile(row_norms, row_drop_quantile)
#         keep_rows = row_norms <= q
#         Stilde_mech = Stilde[keep_rows]
#         m_mech = Stilde_mech.shape[0]
#     else:
#         Stilde_mech = Stilde
#         m_mech = S64.shape[0]

#     # --- Vtilde for mechanistic path ---
#     V64 = V.to(torch.float64)
#     Vtilde = V64 * col_norms  # elementwise scale

#     # --- L2: steady-state constraint ---
#     SV = Stilde_mech @ Vtilde.T  # (m_mech, batch)
#     SV = SV.T                    # (batch, m_mech)
#     SV_norm = torch.linalg.norm(SV, dim=1)
#     SV_norm = torch.clamp(SV_norm, max=1e12)
#     L2 = (SV_norm**2) / m_mech

#     # --- L3: upper bound constraint ---
#     Pin_tilde = Pin_t.to(torch.float64) * Dinv
#     Vin_pred = Vtilde @ Pin_tilde.T
#     relu_Vin = torch.relu(Vin_pred - Vin_t.to(torch.float64))
#     relu_Vin_norm = torch.linalg.norm(relu_Vin, dim=1)
#     relu_Vin_norm = torch.clamp(relu_Vin_norm, max=1e12)
#     L3 = (relu_Vin_norm**2) / Pin_tilde.shape[0]

#     # --- L4: flux positivity constraint ---
#     relu_V = torch.relu(-Vtilde)
#     relu_V_norm = torch.linalg.norm(relu_V, dim=1)
#     relu_V_norm = torch.clamp(relu_V_norm, max=1e12)
#     L4 = (relu_V_norm**2) / Vtilde.shape[1]

#     if not gradient:
#         # --- L1: data-fitting term (original space) ---
#         Vref_pred = V @ Pref_t.T
#         if getattr(hyper_params.model, 'l1_type', 'MSE') == 'MSE':
#             diff = Vref_pred - Vref_t
#             diff_norm = torch.linalg.norm(diff, dim=1)
#             diff_norm = torch.clamp(diff_norm, max=1e12)
#             L1 = (diff_norm**2) / Pref_t.shape[0]
#         else:
#             L1 = normalized_error(Vref_t, Vref_pred, axis=1)

#         # Debug
#         print(f"L1: {L1.mean().item():.6e}, L2: {L2.mean().item():.6e}, "
#               f"L3: {L3.mean().item():.6e}, L4: {L4.mean().item():.6e}")

#         mech_loss = L2 + L3 + L4
#         if hyper_params.model.model_name in ['amn_qp', 'amn_qp_MSE']:
#             if loss_aggregator is not None:
#                 L = loss_aggregator.aggregate_losses(
#                     losses=[L1, mech_loss],
#                     weights=[data_driven_weight * (l1_constant or 1.0), mechanistic_weight]
#                 )
#             else:
#                 L = data_driven_weight * (l1_constant or 1.0) * L1 + mechanistic_weight * mech_loss
#         else:
#             L = L1

#         return L, [L1.sum(), L2.sum(), L3.sum(), L4.sum()]

#     else:
#         # --- Gradients for QP refinement ---
#         dV2_tilde = (SV @ Stilde_mech) * (2.0 / m_mech)
#         dV2 = dV2_tilde * col_norms

#         mask3 = (relu_Vin != 0).to(Vtilde.dtype)
#         dL3_dVinpred = (relu_Vin * mask3) * (2.0 / Pin_tilde.shape[0])
#         dV3_tilde = dL3_dVinpred @ Pin_tilde
#         dV3 = dV3_tilde * col_norms

#         mask4 = (relu_V != 0).to(Vtilde.dtype)
#         dV4_tilde = -(relu_V * mask4) * (2.0 / Vtilde.shape[1])
#         dV4 = dV4_tilde * col_norms

#         dV = dV2 + dV3 + dV4
#         return dV


class MechanisticLoss(nn.Module):
    def __init__(self, model, l1_constant, hyper_params):
        super(MechanisticLoss, self).__init__()
        # S [mxn]: stochiometric matrix
        # Pin [n_in x n]: to go from reactions to medium fluxes
        # Pref [n_ref x n]: to go from reactions to measured fluxes
        # Vin [n_batch x n_in]
        # V  [n_batch x n]
        # Vref  [n_batch x n_ref]
        self.Pref = model.Pref
        self.Pin = model.Pin
        self.S = model.S
        self.l1_constant = l1_constant
        self.hyper_params = hyper_params
       
    
    def forward(self, V, Vref, Vin):
        #print(">>>>> forward pass on MechanisticLoss")

        L, losses = get_loss(V, Vref, Pref = self.Pref, Pin=self.Pin, Vin=Vin, S=self.S, gradient=False, 
                             l1_constant=self.l1_constant, hyper_params=self.hyper_params)
        
        return L.sum(), losses
    


class MechanisticLossWeighted(MechanisticLoss):
    def __init__(self, model, l1_constant, hyper_params, loss_aggregator: LossAggregator,
                 data_driven_weight=0.5, mechanistic_weight=1,
                 data_driver_loss_balance=1, mechanistic_loss_balance=1,
                 biomass_rxn_id: str = None, biomass_weight: float = 10.0):
        super().__init__(model, l1_constant, hyper_params)

        self.loss_aggregator = loss_aggregator

        self.data_driven_weight = data_driven_weight
        self.mechanistic_weight = mechanistic_weight

        self.data_driven_loss_balance = data_driver_loss_balance
        self.mechanistic_loss_balance = mechanistic_loss_balance

        # --- biomass weighting ---
        self.biomass_rxn_id = biomass_rxn_id
        self.biomass_weight = biomass_weight

        # Try to resolve biomass index right away (if reaction IDs available)
        self.biomass_index = -1
        # if self.biomass_rxn_id is not None and hasattr(model, "ref_rxn_ids"):
        #     try:
        #         self.biomass_index = model.ref_rxn_ids.index(self.biomass_rxn_id)
        #         print(f"[Loss] Biomass flux found at index {self.biomass_index}")
        #     except ValueError:
        #         print(f"[Loss] WARNING: Biomass reaction {self.biomass_rxn_id} not found in ref_rxn_ids")

    def forward(self, V, Vref, Vin):
        #print(">>>>> forward pass on MechanisticLoss")
        # Compute base loss
        L, losses = get_loss(
            V, Vref,
            Pref=self.Pref, Pin=self.Pin, Vin=Vin, S=self.S,
            gradient=False,
            loss_aggregator=self.loss_aggregator,
            l1_constant=self.l1_constant,
            hyper_params=self.hyper_params,
            mechanistic_weight=self.mechanistic_weight * self.mechanistic_loss_balance,
            data_driven_weight=self.data_driven_weight * self.data_driven_loss_balance
        )

        # --- apply biomass weighting inside L1 ---
        if self.biomass_index is not None:
            #print(" >>>>> Applying biomass weighting >>>>> ")
            # Recompute Vref_pred just for L1 adjustment
            Pref_t = torch.as_tensor(self.Pref, device=V.device, dtype=V.dtype)
            Vref_pred = torch.matmul(V, Pref_t.T)

            flux_weights = torch.ones(Pref_t.shape[0], device=V.device, dtype=V.dtype)
            flux_weights[self.biomass_index] = self.biomass_weight

            diff = (Vref_pred - Vref) * flux_weights
            L1_biased = (torch.norm(diff, dim=1, p=2) ** 2) / Pref_t.shape[0]
 
            # Replace losses[0] (the old L1 sum) with weighted one
            losses[0] = L1_biased.sum()

            # Rebuild total loss with new L1
            mech_loss = losses[1] + losses[2] + losses[3]
            if self.loss_aggregator is not None:
                L = self.loss_aggregator.aggregate_losses(
                    losses=[L1_biased, mech_loss],
                    weights=[self.data_driven_weight * self.l1_constant,
                             self.mechanistic_weight]
                )
            else:
                L = (self.data_driven_weight * self.l1_constant * L1_biased
                     + self.mechanistic_weight * mech_loss)

        return L.sum(), losses



# class MechanisticLossWeighted(MechanisticLoss):
#     def __init__(self, model, l1_constant, hyper_params, loss_aggregator: LossAggregator, data_driven_weight=0.5, mechanistic_weight=1, data_drivern_loss_balance=1, mechanistic_loss_balance=1):
#         super().__init__(model, l1_constant, hyper_params)
        
#         self.loss_aggregator = loss_aggregator
        
#         self.data_driven_weight = data_driven_weight
#         self.mechanistic_weight = mechanistic_weight
        
#         self.data_driven_loss_balance = data_drivern_loss_balance
#         self.mechanistic_loss_balance = mechanistic_loss_balance
    
#     def forward(self, V, Vref, Vin):
#         print(">>>>> forward pass on the original MechanisticLossWeighted")

#         Pref = self.Pref
#         Pin = self.Pin

#         # print("V shape:", V.shape)
#         # print("V: \n", V)
#         # print("Vref shape:", Vref.shape)
#         # print("Vref: \n", Vref)
#         # print("Vin shape:", Vin.shape)
#         # print("Vin: \n", Vin)
#         # print("Pref shape:", Pref.shape)
#         # print("Pref: \n", Pref)
#         # print("Pin shape:", Pin.shape)
#         # print("Pin: \n", Pin)

#         L, losses = get_loss(V, Vref, Pref = self.Pref, Pin=self.Pin, Vin=Vin, S=self.S, gradient=False, loss_aggregator=self.loss_aggregator, 
#                              l1_constant=self.l1_constant, hyper_params=self.hyper_params, 
#                              mechanistic_weight=self.mechanistic_weight*self.mechanistic_loss_balance, 
#                              data_driven_weight=self.data_driven_weight*self.data_driven_loss_balance)
        
#         return L.sum(), losses


def Gradientdescent_QP(V0, Vref, Pref, Pin, Vin, S, lr=0.01, n_iteration=8, decay_rate=0.9, hyper_params=None): #iteration default = 8
    V = V0
    diff = 0
    #lambda_reg = 10 # Example value, tune this    
    for i in range(n_iteration):
        #print(">>>>> forward pass on Gradientdescent_QP")

        # print("V shape:", V.shape)
        # print("V: \n", V)
        # print("Vref shape:", Vref.shape)
        # print("Vref: \n", Vref)
        # print("Vin shape:", Vin.shape)
        # print("Vin: \n", Vin)
        # print("Pref shape:", Pref.shape)
        # print("Pref: \n", Pref)
        # print("Pin shape:", Pin.shape)
        # print("Pin: \n", Pin)


        dV = get_loss(V, Vref, Pref, Pin, Vin, S, gradient=True, hyper_params=hyper_params)
        #dV_reg = lambda_reg * (V - V0)
        #dV = dV + dV_reg
        diff = decay_rate*diff - lr*dV
        V = V + diff

    return V

# ## MULTIPLE HIDDEN LAYER

# class MINN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, drop_rate, hyper_params, model, depth=1):
#         """
#         MINN model with configurable depth.
        
#         Args:
#             input_size (int): number of input features
#             hidden_size (int): width of hidden layers
#             output_size (int): number of outputs
#             drop_rate (float): dropout probability
#             hyper_params: hyperparameter config
#             model: fit_model (mechanistic component)
#             depth (int): number of hidden layers (>=1)
#         """
#         super(MINN, self).__init__()
#         self.model = model
#         self.hyper_params = hyper_params

#         layers = []
#         # First hidden layer
#         layers.append(nn.Linear(input_size, hidden_size))
#         layers.append(nn.ReLU())

#         # Additional hidden layers if depth > 1
#         for _ in range(depth - 1):
#             layers.append(nn.Linear(hidden_size, hidden_size))
#             layers.append(nn.ReLU())

#         # Dropout before output
#         layers.append(nn.Dropout(drop_rate))

#         # Output layer + Softplus
#         layers.append(nn.Linear(hidden_size, output_size))
#         layers.append(nn.Softplus())

#         self.layers = nn.Sequential(*layers)

#     def forward(self, input, Vref, Vin):
#         # output of pure NN
#         V0 = self.layers(input)
#         #print("---> V0:", V0)
#         #print("LEARNING RATE ===>", self.hyper_params.model.qp_lr)
#         self.hyper_params.model.qp_lr = 1e-5
#         #print("LEARNING RATE ===>", self.hyper_params.model.qp_lr)
#         #return V0, V0
#         if self.hyper_params.model.model_type == 'AMN':
#             Vout = Gradientdescent_QP(V0, Vref, self.model.Pref, self.model.Pin, Vin, self.model.S, 
#                                       lr=self.hyper_params.model.qp_lr, 
#                                       n_iteration=self.hyper_params.model.qp_iter, 
#                                       decay_rate=self.hyper_params.model.qp_decay_rate,  
#                                       hyper_params=self.hyper_params)
#             return Vout, Vout
#         else:
#             return V0, V0


# class MINN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, drop_rate, hyper_params, model):
#         super(MINN, self).__init__()

#         self.model = model
#         self.hyper_params = hyper_params

#         self.layers = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             #nn.Linear(hidden_size, hidden_size),
#             #nn.ReLU(),
#             #nn.Linear(hidden_size, hidden_size),
#             #nn.ReLU(),
#             nn.Dropout(drop_rate),
#             nn.Linear(hidden_size, output_size),
#             nn.Softplus()
#            )
        
#     def forward(self, input, Vref, Vin):
#         # output of pure NN
#         V0 = self.layers(input)
#         #print("---> V0:", V0)
#         #print("LEARNING RATE ===>", self.hyper_params.model.qp_lr)
#         self.hyper_params.model.qp_lr = 1e-5
#         #print("LEARNING RATE ===>", self.hyper_params.model.qp_lr)
#         #return V0, V0
#         if self.hyper_params.model.model_type == 'AMN':
#             Vout = Gradientdescent_QP(V0, Vref, self.model.Pref, self.model.Pin, Vin, self.model.S, 
#                                       lr=self.hyper_params.model.qp_lr, 
#                                       n_iteration=self.hyper_params.model.qp_iter, 
#                                       decay_rate=self.hyper_params.model.qp_decay_rate,  
#                                       hyper_params=self.hyper_params)
#             return Vout, Vout
#         else:
#             return V0, V0



## TWO TAPERED HIDDEN LAYERS

class MINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_rate, hyper_params, model):
        super(MINN, self).__init__()

        self.model = model
        self.hyper_params = hyper_params

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(hidden_size, 1000),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(1000, output_size),
            
            #nn.Linear(hidden_size, output_size),
            nn.Softplus()
        )

    # def forward(self, x, Vref=None, Vin=None):
    #     return self.layers(x), Vin
    def forward(self, input, Vref, Vin):
        # output of pure NN
        V0 = self.layers(input)
        #print("---> V0:", V0)
        #print("LEARNING RATE ===>", self.hyper_params.model.qp_lr)
        self.hyper_params.model.qp_lr = 1e-5
        #print("LEARNING RATE ===>", self.hyper_params.model.qp_lr)
        #return V0, V0
        if self.hyper_params.model.model_type == 'AMN':
            Vout = Gradientdescent_QP(V0, Vref, self.model.Pref, self.model.Pin, Vin, self.model.S, 
                                      lr=self.hyper_params.model.qp_lr, 
                                      n_iteration=self.hyper_params.model.qp_iter, 
                                      decay_rate=self.hyper_params.model.qp_decay_rate,  
                                      hyper_params=self.hyper_params)
            return Vout, Vout
        else:
            return V0, V0


# class MINN_Scaled(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, drop_rate, hyper_params, model):
#         super(MINN_Scaled, self).__init__()

#         self.model = model
#         self.hyper_params = hyper_params

#         self.layers = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(drop_rate),
#             nn.Linear(hidden_size, output_size),
#         )
        
#         # Add proper weight initialization
#         self.apply(self._init_weights)
        
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # Xavier initialization with small scale for extreme outputs
#             torch.nn.init.xavier_uniform_(m.weight, gain=0.01)  # Much smaller gain
#             m.bias.data.fill_(0.0)
        
#     def forward(self, input, Vref, Vin):
#         # output of pure NN
#         V0_raw = self.layers(input)
        
#         # CRITICAL: Scale the output to reasonable range
#         # Instead of letting NN output extreme values, constrain them
#         V0 = torch.tanh(V0_raw) * 100  # Scale to [-100, 100] range
        
#         print("---> V0_raw range:", V0_raw.min().item(), "to", V0_raw.max().item())
#         print("---> V0_scaled range:", V0.min().item(), "to", V0.max().item())
        
#         if self.hyper_params.model.model_type == 'AMN':
#             Vout = Gradientdescent_QP(V0, Vref, self.model.Pref, self.model.Pin, Vin, self.model.S, 
#                                       lr=self.hyper_params.model.qp_lr, 
#                                       n_iteration=self.hyper_params.model.qp_iter, 
#                                       decay_rate=self.hyper_params.model.qp_decay_rate,  
#                                       hyper_params=self.hyper_params)
#             return Vout, Vout
#         else:
#             return V0, V0


class PretrainedBlock(nn.Module):
    def __init__(self, model_GEM, hyper_params=None):
        super(PretrainedBlock, self).__init__()
        
        self.model_GEM = model_GEM
        self.hyper_params = hyper_params

        #self.nn = AMN_QP(input_size=5, hidden_size=500, output_size=587, drop_rate= 0.25, model=self.model_GEM)
        self.nn = MINN(input_size=5, hidden_size=1000, output_size=6542, drop_rate=0.25, hyper_params=hyper_params, model=self.model_GEM)
        # Load the pretrained state dictionary into self.nn
        self.nn.load_state_dict(torch.load("model_fold5.pth"))
        #self.nn.load_state_dict(torch.load("pretrained_block_reservoir_state_dict.pth"))
        # Assign self.nn to self.model so it can be called in forward
        self.model = self.nn
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, Vref, Vin):
        out = self.model(x, Vref, Vin)
        
        # if the inner model returned a tuple (MINN case), take the first tensor
        if isinstance(out, tuple):
            # optional: detect if it's specifically a MINN instance
            if isinstance(self.model, MINN):
                out = out[0]
        return out




class MINN_reservoir(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_rate, hyper_params, model):
        super(MINN_reservoir, self).__init__()

        self.model = model
        self.hyper_params = hyper_params

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(hidden_size, 1000),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            # #####
            # nn.Linear(1000, 2000),
            # nn.ReLU(),
            # nn.Dropout(drop_rate),

            # nn.Linear(2000, 1000),
            # nn.ReLU(),
            # nn.Dropout(drop_rate),
            # #####

            nn.Linear(1000, output_size),
            
            #nn.Linear(hidden_size, output_size),
            nn.Softplus()
        )

        # self.layers = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.ReLU(),

        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),

        #     nn.Dropout(drop_rate),
        #     nn.Linear(hidden_size, output_size),
        #     )
        
        self.pretrained_block = PretrainedBlock(self.model, hyper_params=hyper_params)
        
    def forward(self, input, Vref, Vin):
        # output of pure NN
        V0 = self.layers(input)
       
        
        Vout = self.pretrained_block(V0, Vref, Vin)
              
        return Vout, V0        



class AMN_QP_reservoir(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_rate, hyper_params, model):
        super(AMN_QP_reservoir, self).__init__()

        self.model = model
        self.hyper_params = hyper_params

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, output_size),
            )
        
        self.pretrained_block = PretrainedBlock(self.model)
        

        
    def forward(self, input, Vref, Vin):
        # output of pure NN
        V0 = self.layers(input)
       
        
        Vout = self.pretrained_block(V0, Vref, Vin)
              
        return Vout, V0


if __name__ == "__main__":
    pass