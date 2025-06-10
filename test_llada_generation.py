#!/usr/bin/env python3
"""Test script to debug LLaDA generation issues."""

import torch
from models.config import LLaDAConfig
from models.llada.model import LLaDAModel
import sys

def test_llada_generation():
    # Create config with proper mask_token_id
    config = LLaDAConfig(
        block_size=512,
        vocab_size=50304,
        n_layer=4,
        n_head=8,
        n_embd=512,
        num_experts=4,
        k=2,
        mask_token_id=50303  # Set to vocab_size - 1
    )
    
    print(f"Config: vocab_size={config.vocab_size}, mask_token_id={config.mask_token_id}")
    
    # Create model
    device = 'cpu'  # Force CPU due to CUDA compatibility issues
    model = LLaDAModel(config).to(device)
    model.eval()
    
    # Create a test prompt
    batch_size = 1
    prompt_length = 10
    prompt = torch.randint(0, config.vocab_size - 1, (batch_size, prompt_length), device=device)
    
    print(f"\nTest prompt shape: {prompt.shape}")
    print(f"Prompt tokens: {prompt[0].tolist()}")
    
    # Test BD3-LM generation (default generate method)
    print("\n=== Testing BD3-LM Generation ===")
    try:
        with torch.no_grad():
            generated, _ = model.generate(prompt, gen_length=20)
            print(f"Generated shape: {generated.shape}")
            print(f"Generated tokens: {generated[0].tolist()}")
            
            # Check if generation actually produced new tokens
            prompt_part = generated[0, :prompt_length]
            generated_part = generated[0, prompt_length:]
            print(f"\nPrompt part matches: {torch.equal(prompt[0], prompt_part)}")
            print(f"Generated part: {generated_part.tolist()}")
            
            # Check if generated tokens are all the same (indicating a problem)
            unique_generated = torch.unique(generated_part)
            print(f"Unique generated tokens: {unique_generated.tolist()}")
            if len(unique_generated) == 1:
                print("WARNING: All generated tokens are the same!")
    except Exception as e:
        print(f"BD3-LM generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test original LLaDA generation
    print("\n=== Testing Original LLaDA Generation ===")
    try:
        with torch.no_grad():
            generated, _ = model.generate_original_llada(
                prompt=prompt, 
                steps=16,
                gen_length=20,
                block_length=20,
                temperature=0.8
            )
            print(f"Generated shape: {generated.shape}")
            print(f"Generated tokens: {generated[0].tolist()}")
            
            # Check generation quality
            generated_part = generated[0, prompt_length:]
            unique_generated = torch.unique(generated_part)
            print(f"Unique generated tokens: {unique_generated.tolist()}")
    except Exception as e:
        print(f"Original LLaDA generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llada_generation()