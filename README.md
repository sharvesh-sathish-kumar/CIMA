CIMA: Continuous Implicit Manifold AttentionShrink your Transformer KV cache by 97.4% at 32k context without losing critical facts.CIMA replaces heavy, uncompressed key-value memory tensors with continuous Implicit Neural Fields (SIRENs) and a tiny 2% residual anchor map for exact retrieval.The Problem: Long-Context VRAM ExplosionRunning 32k+ context on local GPUs usually hits a hard memory wall. Standard FP16 KV caching grows linearly ($O(N)$):1,000 tokens: ~109 MB VRAM32,000 tokens: ~3.5 GB VRAM (just for key-value history on a small model)Existing fixes like INT4 quantization introduce heavy noise, and token eviction methods (like H2O) risk permanently dropping rare, critical tokens.How CIMA Fixes ItInstead of saving every single key vector in VRAM, CIMA fits a continuous coordinate function $f_\theta(t)$ over the sequence keys using Sinusoidal Representation Networks (SIRENs).To prevent spatial smoothing from losing needle-in-a-haystack tokens:CIMA checks reconstruction error across all tokens: $E_i = \Vert{}k_i - f_\theta(t_i)\Vert{}_2$It pinpoints the top 2% highest-error states and saves them in a Sparse Residual Anchor Map ($\mathcal{A}$).On query lookup, key vectors are dynamically reconstructed from the continuous function, while high-entropy "needles" pull exact values from the anchor map.Real BenchmarksTested on a 28-layer architecture ($d_{\text{head}}=64$, 16 heads):Context Length (N)Standard FP16 CacheCIMA CacheVRAM SavedNIAH Accuracy1,000109.38 MB23.57 MB78.5%100%4,000437.50 MB30.13 MB93.1%100%8,000875.00 MB38.88 MB95.6%100%16,0001,750.00 MB56.38 MB96.8%100%32,0003,500.00 MB91.38 MB97.4%100%QuickstartCIMA subclasses Hugging Face's DynamicCache, so it plugs right into standard transformers pipelines.Installationgit clone https://github.com/your-username/CIMA.git
cd CIMA
pip install torch transformers
Running with Qwen2.5import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cima_cache import CIMA_Cache

model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

prompt = "The key to developing long-context neural network architectures lies in"
inputs = tokenizer(prompt, return_tensors="pt")

# Initialize CIMA Cache
cima_cache = CIMA_Cache(
    d_model=model.config.hidden_size,
    anchor_ratio=0.02
)

# Run generation using CIMA
outputs = model.generate(
    **inputs,
    max_new_tokens=30,
    past_key_values=cima_cache,
    use_cache=True,
    do_sample=False
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
CitationIf you use CIMA or build on this architecture, feel free to cite the preprint:@article{sharvesh2024cima,
  title={Continuous Implicit Manifold Attention (CIMA): Sub-Linear KV Cache Memory Scaling via Neural Fields and Residual Anchors},
  author={Sharvesh},
  journal={Zenodo Preprint},
  doi={10.5281/zenodo.23049831},
  year={2024}
}
LicenseMIT
