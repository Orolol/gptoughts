#!/usr/bin/env python3
"""
Script de test des optimisations RTX 5090
Teste différentes configurations pour trouver les paramètres optimaux
"""

import torch
import time
import gc
from optimization.rtx5090_optimizations import RTX5090Optimizer, setup_5090_training
from train.lightning_module import LLMLightningModule
import argparse

def benchmark_config(model_type: str, size: str, batch_size: int, seq_len: int):
    """Benchmark une configuration spécifique"""
    
    print(f"\n🔬 Testing: {model_type} ({size}) - Batch: {batch_size}, Seq: {seq_len}")
    print("-" * 60)
    
    # Configuration
    args = argparse.Namespace(
        model_type=model_type,
        size=size,
        batch_size=batch_size,
        block_size=seq_len,
        vocab_size=128000,
        dropout=0.0,
        bias=False,
        use_fp8=True,
        learning_rate=4e-4,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        warmup_iters=1000,
        lr_decay_iters=10000,
        min_lr=4e-5,
        grad_clip=1.0,
        decay_lr=True,
        compile=False,  # Désactivé pour le benchmark
        optimize_attention=True,
        preallocate_memory=False,
        gradient_accumulation_steps=1,
    )
    
    # Setup optimizations
    RTX5090Optimizer.setup_cuda_optimizations()
    
    try:
        # Créer le module
        module = LLMLightningModule(args)
        module = module.cuda()
        module.train()
        
        # Créer des données factices
        dummy_batch = {
            'input_ids': torch.randint(0, args.vocab_size, (batch_size, seq_len)).cuda(),
            'labels': torch.randint(0, args.vocab_size, (batch_size, seq_len)).cuda()
        }
        
        # Warmup
        print("Warming up...")
        for _ in range(3):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = module(dummy_batch['input_ids'], targets=dummy_batch['labels'])
                loss = outputs['loss']
                if loss is not None:
                    loss.backward()
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        num_steps = 10
        for step in range(num_steps):
            step_start = time.time()
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = module(dummy_batch['input_ids'], targets=dummy_batch['labels'])
                loss = outputs['loss']
                if loss is not None:
                    loss.backward()
            
            torch.cuda.synchronize()
            step_time = time.time() - step_start
            
            if step == 0:
                print(f"First step: {step_time:.3f}s")
        
        total_time = time.time() - start_time
        avg_time = total_time / num_steps
        
        # Statistiques mémoire
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        current_memory = torch.cuda.memory_allocated() / 1e9
        
        # Calcul des tokens/sec
        tokens_per_batch = batch_size * seq_len
        tokens_per_sec = tokens_per_batch / avg_time
        
        print(f"\n📊 Results:")
        print(f"  Average step time: {avg_time:.3f}s")
        print(f"  Tokens/sec: {tokens_per_sec:,.0f}")
        print(f"  Peak memory: {peak_memory:.2f} GB")
        print(f"  Current memory: {current_memory:.2f} GB")
        print(f"  Memory efficiency: {(current_memory/32)*100:.1f}%")
        
        # Nettoyer
        del module
        del dummy_batch
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'avg_time': avg_time,
            'tokens_per_sec': tokens_per_sec,
            'peak_memory': peak_memory,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return {
            'avg_time': float('inf'),
            'tokens_per_sec': 0,
            'peak_memory': 0,
            'success': False,
            'error': str(e)
        }

def find_optimal_batch_size(model_type: str, size: str, seq_len: int):
    """Trouve la taille de batch optimale pour une configuration"""
    
    print(f"\n🎯 Finding optimal batch size for {model_type} ({size}) with seq_len={seq_len}")
    
    batch_sizes = [8, 16, 24, 32, 48, 64, 96, 128]
    best_config = None
    best_tokens_per_sec = 0
    
    for batch_size in batch_sizes:
        result = benchmark_config(model_type, size, batch_size, seq_len)
        
        if result['success'] and result['tokens_per_sec'] > best_tokens_per_sec:
            best_tokens_per_sec = result['tokens_per_sec']
            best_config = {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'tokens_per_sec': result['tokens_per_sec'],
                'memory': result['peak_memory']
            }
        
        if not result['success']:
            print(f"  Batch size {batch_size} failed - likely OOM")
            break
    
    if best_config:
        print(f"\n✅ Optimal configuration found:")
        print(f"  Batch size: {best_config['batch_size']}")
        print(f"  Tokens/sec: {best_config['tokens_per_sec']:,.0f}")
        print(f"  Memory used: {best_config['memory']:.2f} GB")
    
    return best_config

def main():
    print("🚀 RTX 5090 Optimization Benchmark")
    print("=" * 60)
    
    # Vérifier GPU
    if not torch.cuda.is_available():
        print("❌ No CUDA device available")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.1f} GB")
    
    # Test configurations
    test_configs = [
        # (model_type, size, seq_lengths_to_test)
        ("mla", "small", [2048, 4096, 8192]),
        ("mla", "medium", [2048, 4096]),
        ("mla", "large", [1024, 2048, 4096]),
        ("nsa", "small", [2048, 4096]),
        ("hrm", "small", [2048, 4096]),
    ]
    
    results = []
    
    for model_type, size, seq_lengths in test_configs:
        for seq_len in seq_lengths:
            print(f"\n{'='*60}")
            config = find_optimal_batch_size(model_type, size, seq_len)
            if config:
                config['model_type'] = model_type
                config['size'] = size
                results.append(config)
    
    # Résumé
    print(f"\n{'='*60}")
    print("📈 SUMMARY - Best Configurations:")
    print(f"{'='*60}")
    
    for r in sorted(results, key=lambda x: x['tokens_per_sec'], reverse=True):
        print(f"\n{r['model_type']} ({r['size']}):")
        print(f"  Batch: {r['batch_size']}, Seq: {r['seq_len']}")
        print(f"  → {r['tokens_per_sec']:,.0f} tokens/sec")
        print(f"  → {r['memory']:.1f} GB memory")
    
    # Sauvegarder les résultats
    import json
    with open('rtx5090_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to rtx5090_benchmark_results.json")

if __name__ == "__main__":
    main()