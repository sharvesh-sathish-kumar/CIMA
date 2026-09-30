import torch
import torch.nn as nn
import torch.optim as optim

class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(
                    -torch.sqrt(torch.tensor(6.0 / self.linear.in_features)) / self.omega_0,
                    torch.sqrt(torch.tensor(6.0 / self.linear.in_features)) / self.omega_0,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class ImplicitNeuralField(nn.Module):
    def __init__(self, d_model: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(1, hidden_dim, is_first=True),
            SineLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)


def evaluate_needle_retrieval(seq_len: int = 8000, needle_depth: float = 0.5, anchor_ratio: float = 0.02):
    """
    Evaluates if CIMA's residual anchors successfully capture a high-entropy 'needle' 
    hidden inside a smooth 'haystack' sequence.
    """
    torch.manual_seed(42)
    d_model = 64
    needle_idx = int(seq_len * needle_depth)

    # 1. Create Haystack (Background) + High-Entropy Needle
    t_steps = torch.linspace(-1, 1, seq_len).unsqueeze(-1)
    haystack_keys = torch.sin(t_steps * 3.0) @ torch.randn(1, d_model)
    
    # Inject unique, high-variance "needle" key at specified depth
    needle_key = torch.randn(1, d_model) * 5.0  # High magnitude state
    true_keys = haystack_keys.clone()
    true_keys[needle_idx] = needle_key.squeeze(0)

    # 2. Fit Implicit Neural Field over sequence
    inf_model = ImplicitNeuralField(d_model=d_model, hidden_dim=128)
    optimizer = optim.Adam(inf_model.parameters(), lr=1e-3)

    for _ in range(100):
        optimizer.zero_grad()
        pred_keys = inf_model(t_steps)
        loss = torch.mean((true_keys - pred_keys) ** 2)
        loss.backward()
        optimizer.step()

    # 3. Residual Anchor Selection (Top k% highest errors)
    with torch.no_grad():
        fitted_keys = inf_model(t_steps)
        errors = torch.norm(true_keys - fitted_keys, dim=-1)
        
        k_anchors = int(anchor_ratio * seq_len)
        top_error_indices = torch.topk(errors, k=k_anchors).indices

        # Check if the needle position was caught by the anchor mechanism
        needle_captured = needle_idx in top_error_indices.tolist()

        # Reconstruct Keys
        reconstructed_keys = fitted_keys.clone()
        for idx in top_error_indices:
            reconstructed_keys[idx] = true_keys[idx]

    # 4. Query Retrieval Check: Search specifically for the Needle Key
    query = needle_key  # Query matching the needle state
    scale = d_model ** 0.5

    exact_attn = torch.softmax((query @ true_keys.T) / scale, dim=-1)
    cima_attn = torch.softmax((query @ reconstructed_keys.T) / scale, dim=-1)

    exact_retrieved_idx = torch.argmax(exact_attn).item()
    cima_retrieved_idx = torch.argmax(cima_attn).item()

    retrieval_success = (exact_retrieved_idx == needle_idx) and (cima_retrieved_idx == needle_idx)

    return needle_captured, retrieval_success


if __name__ == "__main__":
    print("--- Running Phase 1: Needle-In-A-Haystack (NIAH) Benchmark ---")
    depths = [0.10, 0.25, 0.50, 0.75, 0.90]
    seq_lengths = [4000, 8000, 16000]

    for N in seq_lengths:
        print(f"\nContext Length N = {N} tokens:")
        for depth in depths:
            captured, retrieved = evaluate_needle_retrieval(seq_len=N, needle_depth=depth, anchor_ratio=0.02)
            print(f"  Needle Depth {int(depth*100):02d}% | Anchor Captured Needle: {captured} | Correct Key Retrieved: {retrieved}")