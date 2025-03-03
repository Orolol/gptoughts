import argparse
import torch
from llada import LLaDAConfig, LLaDAModel
from tqdm import tqdm
import time

def load_model(checkpoint_path, device):
    """Load a trained LLaDA model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # If config not in checkpoint, use default config (this should be avoided)
        print("Warning: No config found in checkpoint, using default config")
        config = LLaDAConfig()
    
    model = LLaDAModel(config)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Try loading the checkpoint directly (if it only contains model weights)
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, config

def get_tokenizer(tokenizer_path):
    """Load tokenizer (placeholder, implement based on your tokenizer)"""
    # This is just a placeholder function - you would need to implement 
    # based on your actual tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return tokenizer
    except:
        print(f"Warning: Could not load tokenizer from {tokenizer_path}")
        # Create a dummy tokenizer for demonstration purposes
        class DummyTokenizer:
            def encode(self, text):
                return [ord(c) % 50000 for c in text]
                
            def decode(self, ids):
                return ''.join([chr((id % 26) + 97) for id in ids])
                
        return DummyTokenizer()

def generate_text(model, tokenizer, prompt_text, args):
    """Generate text using the LLaDA model"""
    device = args.device
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt_text)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # Set up generation parameters
    steps = args.steps if args.steps else len(prompt_ids)
    gen_length = args.length if args.length else len(prompt_ids)
    
    # Ensure block length divides generation length
    if gen_length % args.block_length != 0:
        gen_length = (gen_length // args.block_length + 1) * args.block_length
        print(f"Adjusted generation length to {gen_length} to be divisible by block length")
    
    # Adjust steps if needed
    if steps % (gen_length // args.block_length) != 0:
        steps = (steps // (gen_length // args.block_length) + 1) * (gen_length // args.block_length)
        print(f"Adjusted steps to {steps} to be divisible by number of blocks")
    
    print(f"Generating {gen_length} tokens with {steps} demasking steps...")
    
    # Generate text using LLaDA diffusion process
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            remasking=args.remasking
        )
    
    generation_time = time.time() - start_time
    tokens_per_second = gen_length / generation_time
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    return {
        'prompt': prompt_text,
        'generated_text': generated_text,
        'generation_time': generation_time,
        'tokens_per_second': tokens_per_second
    }

def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained LLaDA model")
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='Path or name of tokenizer')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, default='Once upon a time', help='Text prompt to start generation')
    parser.add_argument('--length', type=int, default=128, help='Number of tokens to generate')
    parser.add_argument('--steps', type=int, default=None, help='Number of demasking steps (default: same as length)')
    parser.add_argument('--block_length', type=int, default=32, help='Block length for semi-autoregressive generation')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for sampling (0 for deterministic)')
    parser.add_argument('--remasking', type=str, default='low_confidence', 
                       choices=['low_confidence', 'random'], help='Remasking strategy')
    
    # Output options
    parser.add_argument('--output', type=str, default=None, help='Output file path (optional)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run generation on')
    
    args = parser.parse_args()
    
    # Load model
    device = torch.device(args.device)
    model, config = load_model(args.checkpoint, device)
    
    # Load tokenizer
    tokenizer = get_tokenizer(args.tokenizer)
    
    # Generate text
    result = generate_text(model, tokenizer, args.prompt, args)
    
    # Print results
    print("\n" + "="*50)
    print(f"Prompt: {result['prompt']}")
    print("-"*50)
    print(f"Generated text:\n{result['generated_text']}")
    print("-"*50)
    print(f"Generation time: {result['generation_time']:.2f}s")
    print(f"Speed: {result['tokens_per_second']:.2f} tokens/second")
    print("="*50 + "\n")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {result['prompt']}\n\n")
            f.write(f"Generated text:\n{result['generated_text']}\n\n")
            f.write(f"Generation time: {result['generation_time']:.2f}s\n")
            f.write(f"Speed: {result['tokens_per_second']:.2f} tokens/second\n")
        print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main() 