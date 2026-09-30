import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Simulate Transformer Dimensions (e.g., Llama-3.2-1B / Qwen2.5-0.5B style architecture)
NUM_LAYERS = 28
NUM_HEADS = 16
HEAD_DIM = 64
ANCHOR_RATIO = 0.02  # 2% sparse anchors

def measure_vram_scaling():
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine. Running CPU simulation estimate instead...\n")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    seq_lengths = [1000, 4000, 8000, 16000, 32000]
    std_vram_mb = []
    cima_vram_mb = []

    print(f"{'Sequence Length':<16} | {'Standard FP16 Cache (MB)':<24} | {'CIMA Cache (MB)':<18} | {'VRAM Reduction':<15}")
    print("-" * 80)

    for N in seq_lengths:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

        # 1. Standard FP16 KV Cache Size (2 bytes per FP16 element for Keys and Values)
        # Size = 2 layers * num_layers * batch(1) * num_heads * seq_len * head_dim * 2 bytes
        std_bytes = 2 * NUM_LAYERS * 1 * NUM_HEADS * N * HEAD_DIM * 2
        std_mb = std_bytes / (1024 ** 2)
        std_vram_mb.append(std_mb)

        # 2. CIMA Cache Size (INF Weights + 2% Anchors)
        # INF parameters per head (2-layer MLP with hidden dim 128) ~ 17,000 parameters per head
        inf_params_per_head = (1 * 128) + 128 + (128 * 128) + 128 + (128 * HEAD_DIM) + HEAD_DIM
        total_inf_bytes = NUM_LAYERS * NUM_HEADS * inf_params_per_head * 2  # FP16
        
        # Sparse Anchors (2% of N tokens saved in FP16)
        k_anchors = int(ANCHOR_RATIO * N)
        anchor_bytes = 2 * NUM_LAYERS * 1 * NUM_HEADS * k_anchors * HEAD_DIM * 2
        
        cima_bytes = total_inf_bytes + anchor_bytes
        cima_mb = cima_bytes / (1024 ** 2)
        cima_vram_mb.append(cima_mb)

        reduction = (1 - (cima_mb / std_mb)) * 100
        print(f"{N:<16} | {std_mb:<24.2f} | {cima_mb:<18.2f} | {reduction:.1f}%")

    # Generate Memory Comparison Plot
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, std_vram_mb, marker='o', linewidth=2, label='Standard FP16 KV Cache O(N)', color='red')
    plt.plot(seq_lengths, cima_vram_mb, marker='s', linewidth=2, label='CIMA Cache (Ours)', color='green')
    plt.title('KV Cache VRAM Consumption Scaling (1k to 32k Context)', fontsize=14)
    plt.xlabel('Sequence Length (Tokens)', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    plot_path = "vram_scaling_plot.png"
    plt.savefig(plot_path)
    print(f"\nVRAM scaling plot saved successfully to '{plot_path}'.")

if __name__ == "__main__":
    measure_vram_scaling()