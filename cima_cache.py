import torch
from transformers.cache_utils import DynamicCache
from typing import List, Tuple, Optional
from cima_poc import ImplicitNeuralField

class CIMA_Cache(DynamicCache):
    def __init__(self, d_model: int = 64, anchor_ratio: float = 0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = d_model
        self.anchor_ratio = anchor_ratio
        
        # Metadata storage for INF models and sparse residual anchors per layer
        self.inf_models: List[Optional[ImplicitNeuralField]] = []
        self.sparse_anchors: List[dict] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Delegates key/value updates to DynamicCache, retaining standard attention compatibility.
        """
        # Maintain metadata structures for custom layer index tracking
        while len(self.inf_models) <= layer_idx:
            self.inf_models.append(None)
            self.sparse_anchors.append({})

        # DynamicCache handles tensor appending and layer list sizing
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def compress_layer_keys(self, layer_idx: int, fit_steps: int = 50):
        """
        Fits the Implicit Neural Field (SIREN) on keys stored in layer_idx
        and populates the sparse residual anchor map.
        """
        keys = self.key_cache[layer_idx]  # Shape: [batch, heads, seq_len, head_dim]
        batch, heads, seq_len, head_dim = keys.shape
        
        flat_keys = keys.permute(2, 0, 1, 3).reshape(seq_len, -1)
        t_steps = torch.linspace(-1, 1, seq_len, device=keys.device).unsqueeze(-1)

        inf_model = ImplicitNeuralField(d_model=flat_keys.shape[-1], hidden_dim=128).to(keys.device)
        optimizer = torch.optim.Adam(inf_model.parameters(), lr=1e-3)

        for _ in range(fit_steps):
            optimizer.zero_grad()
            pred_keys = inf_model(t_steps)
            loss = torch.mean((flat_keys - pred_keys) ** 2)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            reconstructed = inf_model(t_steps)
            errors = torch.norm(flat_keys - reconstructed, dim=-1)
            k_anchors = max(1, int(self.anchor_ratio * seq_len))
            anchor_indices = torch.topk(errors, k=k_anchors).indices

            anchors = {idx.item(): flat_keys[idx].clone() for idx in anchor_indices}

        self.inf_models[layer_idx] = inf_model
        self.sparse_anchors[layer_idx] = anchors