import torch
import sys
sys.path.append('.')
from llada.llada import LLaDAConfig, LLaDAModel, generate_example_wrapper

def main():
    # Create a small test model
    config = LLaDAConfig(n_layer=2, n_head=4, n_embd=128)
    model = LLaDAModel(config)
    model = model.to('cpu')
    
    # Simple test prompt
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    
    # Try to load a tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        print('Loaded tokenizer successfully')
    except:
        print('Could not load tokenizer')
        tokenizer = None
    
    # Test generation
    print('Testing generation...')
    generate_example_wrapper(model, prompt, tokenizer=tokenizer, steps=2, gen_length=5)
    print('Test completed!')

if __name__ == '__main__':
    main() 