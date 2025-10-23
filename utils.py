# dataset loaders, training, evaluation, cosine similarity
import os
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np, random
import matplotlib.pyplot as plt
import numpy as np
import sys

def _safe_pyplot():
    # Import locally to avoid outer-scope shadowing
    import matplotlib.pyplot as _plt
    fig_attr = getattr(_plt, "figure", None)
    # If figure isn't callable, something shadowed pyplot
    if not callable(fig_attr):
        src = getattr(_plt, "__file__", "<no __file__>")
        raise RuntimeError(
            f"matplotlib.pyplot.figure is not callable. "
            f"You're likely shadowing pyplot. "
            f'pyplot came from: {src}; type(figure)={type(fig_attr)}'
        )
    return _plt

def add_label_noise(dataset, flip_prob=0.1):
    noisy_targets = []
    for label in dataset.targets:
        if random.random() < flip_prob:
            noisy_targets.append(1 - label)
        else:
            noisy_targets.append(label)
    dataset.targets = noisy_targets

def get_cifar10_airplane_automobile_loaders_noniid(cfg):
    num_clients = cfg['num_clients']
    client_splits = cfg['client_splits']
    batch_size = cfg['batch_size']
    label_noise = cfg['label_noise']
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10 = datasets.CIFAR10(root=cfg['dataset_root'], train=True, download=True, transform=transform)
    idx_air = [i for i, t in enumerate(cifar10.targets) if t == 0]
    idx_auto = [i for i, t in enumerate(cifar10.targets) if t == 1]
    np.random.shuffle(idx_air); np.random.shuffle(idx_auto)
    min_samples = min(len(idx_air), len(idx_auto)) // num_clients
    cur_air = cur_auto = 0; client_indices = []
    for air_prop, auto_prop in client_splits:
        num_air = int(min_samples * air_prop)
        num_auto = min_samples - num_air
        cidx = idx_air[cur_air:cur_air+num_air] + idx_auto[cur_auto:cur_auto+num_auto]
        cur_air += num_air; cur_auto += num_auto
        client_indices.append(cidx)

    class SubCIFAR10(Dataset):
        def __init__(self, cifar10, indices):
            self.data = cifar10.data[indices]
            self.targets = [0 if cifar10.targets[i] == 0 else 1 for i in indices]
            self.transform = cifar10.transform
        def __len__(self): return len(self.targets)
        def __getitem__(self, idx):
            x = self.data[idx]; y = self.targets[idx]
            if self.transform: x = self.transform(x)
            return x, y

    clients = [SubCIFAR10(cifar10, idxs) for idxs in client_indices]
    for ds in clients: add_label_noise(ds, flip_prob=label_noise)
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True) for ds in clients]

    testset = datasets.CIFAR10(root=cfg['dataset_root'], train=False, download=True, transform=transform)
    idx_test = [i for i, t in enumerate(testset.targets) if t in [0,1]]
    test_data = SubCIFAR10(testset, idx_test)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    return client_loaders, test_loader

def train_client(model, loader, device, epochs=1, lr=0.01, momentum=0.9):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item(); total += y.size(0)
    return correct / total

def compute_client_cosine_similarity(matrices, client_idx):
    similarities = []
    target = matrices[client_idx].flatten()
    for i, m in enumerate(matrices):
        if i != client_idx:
            other = m.flatten()
            cos_sim = F.cosine_similarity(target.unsqueeze(0), other.unsqueeze(0), dim=1).item()
            similarities.append(cos_sim)
    return np.mean(similarities) if similarities else 0.0

def reshape_weight(weight, to_2d=True):
    original_shape = weight.shape
    if len(original_shape) == 4:
        if to_2d:
            return weight.view(original_shape[0], -1), original_shape
        else:
            return weight, original_shape
    elif len(original_shape) == 2:
        return weight, original_shape
    else:
        raise ValueError("Unsupported weight shape")
    

def decompose_weight(weight, k):
    original_shape = weight.shape
    if len(original_shape) == 4:  # Convolutional layer
        weight_2d = weight.view(original_shape[0], -1)
    elif len(original_shape) == 2:  # Linear layer
        weight_2d = weight
    else:
        raise ValueError("Unsupported weight shape")
    # Mean subtraction
    row_means = weight_2d.mean(dim=1, keepdim=True)
    weight_centered = weight_2d - row_means
    # Compute SVD
    U, S, Vh = torch.linalg.svd(weight_centered, full_matrices=False)
    # Compute A and B
    sqrt_S = torch.sqrt(S)
    A = U[:, :k] @ torch.diag(sqrt_S[:k])
    B = torch.diag(sqrt_S[:k]) @ Vh[:k, :]
    return A, B, row_means, original_shape


def reconstruct_weight(A, B, row_means, original_shape):
    weight_2d = A @ B
    weight_2d = weight_2d + row_means
    if len(original_shape) == 4:
        out_channels, in_channels, kh, kw = original_shape
        weight = weight_2d.view(out_channels, in_channels, kh, kw)
    elif len(original_shape) == 2:
        weight = weight_2d
    else:
        raise ValueError("Unsupported original shape")
    return weight

def save_results(acc_dict, save_dir):
    """Save accuracy and other arrays as .npy files."""
    os.makedirs(save_dir, exist_ok=True)
    for name, arr in acc_dict.items():
        np.save(os.path.join(save_dir, f"{name}_acc.npy"), np.array(arr))
    print(f"Results saved in: {save_dir}")

def save_cosine_sims(cosine_sims: dict, save_dir: str, as_csv: bool = True):
    """
    cosine_sims format:
      {"conv1.weight": {"Client 1": [...], "Client 2": [...], ...}, "conv2.weight": {...}}
    Saves:
      - per-layer .npz with keys "<client>"
      - optional per-layer CSVs
      - a metadata JSON summarizing keys
    """
    os.makedirs(save_dir, exist_ok=True)
    meta = {}
    for layer, client_dict in cosine_sims.items():
        layer_safe = layer.replace(".", "_")
        # NPZ (one file per layer)
        arrays = {client: np.asarray(series) for client, series in client_dict.items()}
        npz_path = os.path.join(save_dir, f"{layer_safe}.npz")
        np.savez(npz_path, **arrays)
        meta[layer] = sorted(list(client_dict.keys()))
    # write a small manifest
    with open(os.path.join(save_dir, "_manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Cosine similarities saved in: {save_dir}")


def plot_accuracy(acc_dict, title, save_path=None):
    plt = _safe_pyplot()
    plt.figure(figsize=(8, 5))
    for label, acc in acc_dict.items():
        plt.plot(range(1, len(acc) + 1), acc, marker='^', label=label)
    plt.xlabel("Round")
    plt.ylabel("Average Local Test Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.show()

def plot_cosine_similarities(cosine_sims_dict, key_name, save_dir=None):
    """cosine_sims_dict: {'conv1.weight': {'Client 1': [...], 'Client 2': [...]}}"""
    plt = _safe_pyplot()
    for key, clients_dict in cosine_sims_dict.items():
        plt.figure(figsize=(8, 5))
        for client, values in clients_dict.items():
            plt.plot(range(1, len(values) + 1), values, label=client)
        plt.xlabel("Round")
        plt.ylabel("Mean Cosine Similarity")
        plt.title(f"{key_name} - {key}")
        plt.legend()
        plt.grid(True)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{key.replace('.', '_')}.png"), dpi=300)
            print(f"Saved cosine plot: {os.path.join(save_dir, key.replace('.', '_') + '.png')}")
        plt.show()


