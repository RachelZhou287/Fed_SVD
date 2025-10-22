import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from model import SimpleCNN
from utils import (
    get_cifar10_airplane_automobile_loaders_noniid,
    train_client, evaluate, compute_client_cosine_similarity,decompose_weight,reconstruct_weight
)

def federated_learning(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client_loaders, test_loader = get_cifar10_airplane_automobile_loaders_noniid(cfg)
    global_model = SimpleCNN().to(device)
    k = cfg['rank']
    compressed_keys = ['conv1.weight', 'conv2.weight']
    acc_per_round = []
    # Initialize local models, B, and row means
    local_models = [deepcopy(global_model).to(device) for _ in range(cfg['num_clients'])]
    local_Bs = [{} for _ in range(cfg['num_clients'])]
    local_row_means = [{} for _ in range(cfg['num_clients'])]
    # Store cosine similarities for each client
    cosine_sims_A = {key: {f'Client {i+1}': [] for i in range(cfg['num_clients'])} for key in compressed_keys}
    cosine_sims_B = {key: {f'Client {i+1}': [] for i in range(cfg['num_clients'])} for key in compressed_keys}
    
    for rnd in range(cfg['num_rounds']):
        A_dicts = [{} for _ in range(cfg['num_clients'])]
        
        # Local training and decomposition
        for i in range(cfg['num_clients']):
            train_client(local_models[i], client_loaders[i], device, epochs=cfg['epochs_per_round'])
            state_dict = local_models[i].state_dict()
            for key in compressed_keys:
                weight = state_dict[key]
                A, B, row_means, original_shape = decompose_weight(weight, k)
                local_Bs[i][key] = B
                local_row_means[i][key] = row_means
                A_dicts[i][key] = A
        
        # Compute client-specific mean cosine similarities
        for key in compressed_keys:
            A_matrices = [A_dicts[i][key] for i in range(cfg['num_clients'])]
            B_matrices = [local_Bs[i][key] for i in range(cfg['num_clients'])]
            for i in range(cfg['num_clients']):
                mean_cos_sim_A = compute_client_cosine_similarity(A_matrices, i)
                mean_cos_sim_B = compute_client_cosine_similarity(B_matrices, i)
                cosine_sims_A[key][f'Client {i+1}'].append(mean_cos_sim_A)
                cosine_sims_B[key][f'Client {i+1}'].append(mean_cos_sim_B)
        
        # Aggregate A
        avg_A_dict = {}
        for key in compressed_keys:
            A_tensors = [A_dicts[i][key] for i in range(cfg['num_clients'])]
            avg_A = torch.stack(A_tensors, dim=0).mean(dim=0)
            avg_A_dict[key] = avg_A
        
        # Update local models with aggregated A
        for i in range(cfg['num_clients']):
            state_dict = local_models[i].state_dict()
            for key in compressed_keys:
                A_global = avg_A_dict[key]
                B_local = local_Bs[i][key]
                row_means = local_row_means[i][key]
                weight = reconstruct_weight(A_global, B_local, row_means, state_dict[key].shape)
                state_dict[key] = weight
            local_models[i].load_state_dict(state_dict)
        
        # Evaluate: Average local model accuracies
        acc = np.mean([evaluate(local_models[i], test_loader, device) for i in range(cfg['num_clients'])])
        acc_per_round.append(acc)
        print(f"Fed Round {rnd+1}/{cfg['num_rounds']} - Avg Local Test Accuracy: {acc:.4f}")

    return {
    "accuracy": acc_per_round,
    "cosine_sims_A": cosine_sims_A,
    "cosine_sims_B": cosine_sims_B
}