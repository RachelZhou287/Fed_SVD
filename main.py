import yaml
import argparse
from algorithms import global_svd, local_b_svd
from utils import plot_accuracy, plot_cosine_similarities, save_results



if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    
    print("== Running Global SVD Aggregation ==")
    res_global = global_svd.federated_learning(cfg)
    
    print("== Running Local-B SVD Aggregation ==")
    res_localb = local_b_svd.federated_learning(cfg)

    # === Save and plot ===
    acc_dict = {
        "Global SVD": res_global["accuracy"],
        "Local-B SVD": res_localb["accuracy"],
    }
    save_results(acc_dict, "./results")
    plot_accuracy(acc_dict, "Federated Learning Accuracy Comparison", "./results/plots/accuracy_comparison.png")

    plot_cosine_similarities(res_global["cosine_sims"], "Global SVD Cosine Similarity", "./results/plots/global_svd")
    plot_cosine_similarities(res_localb["cosine_sims_A"], "Local-B SVD (A matrices)", "./results/plots/localb_A")
    plot_cosine_similarities(res_localb["cosine_sims_B"], "Local-B SVD (B matrices)", "./results/plots/localb_B")
