import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from model import SimpleCNN
from utils import (
    get_cifar10_airplane_automobile_loaders_noniid,
    train_client, evaluate, compute_client_cosine_similarity,reshape_weight
)


def federated_learning(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client_loaders, test_loader = get_cifar10_airplane_automobile_loaders_noniid(cfg)
    global_model = SimpleCNN().to(device)
    k = cfg['rank']
    compressed_keys = ['conv1.weight', 'conv2.weight']
    acc_per_round = []
    local_models = [deepcopy(global_model).to(device) for _ in range(cfg['num_clients'])]
    cosine_sims_W = {key: {f'Client {i+1}': [] for i in range(cfg['num_clients'])} for key in compressed_keys}
    
    for rnd in range(cfg['num_rounds']):
        state_dicts = []
        
        # Local training
        for i in range(cfg['num_clients']):
            train_client(local_models[i], client_loaders[i], device, epochs=cfg['epochs_per_round'])
            state_dicts.append(local_models[i].state_dict())
        
        # Global aggregation and SVD
        new_weights_dicts = [deepcopy(state_dicts[i]) for i in range(cfg['num_clients'])]
        for key in compressed_keys:
            weights = [state_dicts[i][key] for i in range(cfg['num_clients'])]
            weights_2d = [reshape_weight(w)[0] for w in weights]
            original_shape = weights[0].shape
            p = weights_2d[0].shape[1]
            
            # Compute W_bar
            W_bar = torch.stack(weights_2d, dim=0).mean(dim=0)
            
            # Compute centered weights
            centered_weights = [w - W_bar for w in weights_2d]
            
            # Form M matrix
            M = torch.cat(centered_weights, dim=1)  # [q x 3p]
            
            # SVD
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            sqrt_S = torch.sqrt(S[:k])
            A = U[:, :k] @ torch.diag(sqrt_S)
            Vh_r = Vh[:k, :]
            
            # Compute new weights
            for i in range(cfg['num_clients']):
                B_k_T = torch.diag(sqrt_S) @ Vh_r[:, i*p:(i+1)*p]  # [r x p]
                weight_2d = A @ B_k_T + W_bar
                if len(original_shape) == 4:
                    weight = weight_2d.view(original_shape)
                else:
                    weight = weight_2d
                new_weights_dicts[i][key] = weight
            
            # Compute cosine similarities
            for i in range(cfg['num_clients']):
                mean_cos_sim = compute_client_cosine_similarity(weights_2d, i)
                cosine_sims_W[key][f'Client {i+1}'].append(mean_cos_sim)
        
        # Update non-compressed layers
        for key in state_dicts[0].keys():
            if key not in compressed_keys:
                avg_param = torch.stack([sd[key] for sd in state_dicts], dim=0).mean(dim=0)
                for i in range(cfg['num_clients']):
                    new_weights_dicts[i][key].copy_(avg_param)
        
        # Update local models
        for i in range(cfg['num_clients']):
            local_models[i].load_state_dict(new_weights_dicts[i])
        
        # Evaluate
        acc = np.mean([evaluate(local_models[i], test_loader, device) for i in range(cfg['num_clients'])])
        acc_per_round.append(acc)
        print(f"Round {rnd+1}/{cfg['num_rounds']} - Avg Local Test Accuracy: {acc:.4f}")
    
    return {
    "accuracy": acc_per_round,
    "cosine_sims": cosine_sims_W
}
