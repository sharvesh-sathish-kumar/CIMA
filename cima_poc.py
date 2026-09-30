import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# 1. SIREN (Implicit Neural Field)
# ---------------------------------------------------------------------------
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
        # t expects normalized time coordinates in [-1, 1]
        return self.net(t)


# ---------------------------------------------------------------------------
# 2. Main Proof-of-Concept Execution
# ---------------------------------------------------------------------------
def run_cima_poc():
    torch.manual_seed(42)
    seq_len = 4000
    d_model = 64

    # 1. Synthetic Data Generation: Sequence of 4,000 synthetic Key vectors
    # Smooth signal + random token features to simulate transformer manifold
    t_steps = torch.linspace(-1, 1, seq_len).unsqueeze(-1)
    true_keys = torch.sin(t_steps * 5.0) @ torch.randn(1, d_model) + 0.1 * torch.randn(seq_len, d_model)

    # 2. Implement INF
    inf_model = ImplicitNeuralField(d_model=d_model, hidden_dim=128)
    optimizer = optim.Adam(inf_model.parameters(), lr=1e-3)

    # 3. Fit Function
    print("Fitting Implicit Neural Field (INF)...")
    for step in range(100):
        optimizer.zero_grad()
        pred_keys = inf_model(t_steps)
        loss = torch.mean((true_keys - pred_keys) ** 2)
        loss.backward()
        optimizer.step()
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1:03d} | MSE Loss: {loss.item():.6f}")

    # 4. Error Calculation & Sparse Residual Anchor Selection
    with torch.no_grad():
        fitted_keys = inf_model(t_steps)
        errors = torch.norm(true_keys - fitted_keys, dim=-1)

        # Select top 2% highest-error positions
        k_anchors = int(0.02 * seq_len)
        top_error_indices = torch.topk(errors, k=k_anchors).indices

        # Populate Sparse Anchor Map
        anchors = {idx.item(): true_keys[idx].clone() for idx in top_error_indices}

    print(f"\nStored {len(anchors)} anchors ({len(anchors)/seq_len:.1%} sparse memory budget).")

    # Reconstruct keys: Start from continuous MLP output and insert exact anchors
    reconstructed_keys = fitted_keys.clone()
    for idx, anchor_vec in anchors.items():
        reconstructed_keys[idx] = anchor_vec

    # 5. Query Reconstruction & Standard Attention Match Check
    query = torch.randn(1, d_model)
    scale = d_model**0.5

    # Standard Dot-Product Attention over True vs. CIMA Keys
    exact_attn_logits = (query @ true_keys.T) / scale
    cima_attn_logits = (query @ reconstructed_keys.T) / scale

    exact_attn_weights = torch.softmax(exact_attn_logits, dim=-1)
    cima_attn_weights = torch.softmax(cima_attn_logits, dim=-1)

    cos_sim = torch.cosine_similarity(exact_attn_weights, cima_attn_weights, dim=-1).item()
    max_diff = torch.max(torch.abs(exact_attn_weights - cima_attn_weights)).item()

    print("\n--- Validation Results ---")
    print(f"Attention Output Cosine Similarity: {cos_sim:.6f}")
    print(f"Max Absolute Error in Softmax Probabilities: {max_diff:.6f}")


if __name__ == "__main__":
    run_cima_poc()