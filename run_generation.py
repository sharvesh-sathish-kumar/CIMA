import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cima_cache import CIMA_Cache

def run_cima_generation():
    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"Loading model: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.float32, 
        device_map="cpu"
    )

    prompt = "The key to developing long-context neural network architectures lies in"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Instantiate CIMA Cache as a DynamicCache subclass
    cima_cache = CIMA_Cache(
        d_model=model.config.hidden_size, 
        anchor_ratio=0.02
    )

    print("\nGenerating text using CIMA_Cache...")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        past_key_values=cima_cache,
        use_cache=True,
        do_sample=False
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- Generation Output ---")
    print(generated_text)
    print(f"\nTotal Tokens Cached: {cima_cache.get_seq_length()}")

if __name__ == "__main__":
    run_cima_generation()