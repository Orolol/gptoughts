import os
import torch

class TrainingConfig:
    def __init__(self):
        # Access token
        self.access_token = os.getenv('HF_TOKEN')
        
        # Détection automatique du nombre de GPUs
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.device = 'cuda' if self.n_gpus > 0 else 'cpu'
        
        # I/O
        self.out_dir = 'out'
        self.eval_interval = 1000
        self.log_interval = 1
        self.generate_interval = 100
        self.eval_iters = 100
        self.eval_only = False
        self.always_save_checkpoint = True
        self.init_from = 'scratch'
        
        # Wandb logging
        self.wandb_log = False
        self.wandb_project = 'owt'
        self.wandb_run_name = 'gpt2'
        
        # Data
        self.dataset = 'openwebtext'
        self.data_dir = 'data/openwebtext'
        
        # Training
        if self.device == 'cuda':
            self.batch_size = 8 * max(1, self.n_gpus)  # Scale with number of GPUs
            self.block_size = 64
            
            # Encoder config
            self.encoder_n_layer = 4
            self.encoder_n_head = 8
            self.encoder_n_embd = 768
            self.encoder_ratio_kv = 8
            
            # Decoder config
            self.decoder_n_layer = 12
            self.decoder_n_head = 12
            self.decoder_n_embd = 768
            self.decoder_ratio_kv = 8
            
            # Memory optimizations
            self.gradient_accumulation_steps = 1
            self.dtype = 'bfloat16'
            
            # Enable optimized memory management
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.batch_size = 2
            self.block_size = 64
            
            # Minimal configs for CPU
            self.encoder_n_layer = 1
            self.encoder_n_head = 1
            self.encoder_n_embd = 2
            self.encoder_ratio_kv = 1
            
            self.decoder_n_layer = 1
            self.decoder_n_head = 1
            self.decoder_n_embd = 2
            self.decoder_ratio_kv = 1
            
            self.gradient_accumulation_steps = 1
            self.dtype = 'float32'
        
        # Optimizer settings
        self.learning_rate = 3e-4
        self.max_iters = 600000
        self.weight_decay = 0.1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0
        
        # Learning rate decay
        self.decay_lr = True
        self.dropout = 0.0
        self.bias = False
        
        # Learning rate warmup and decay settings
        self.warmup_iters = 2000  # how many steps to warm up for
        self.lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
        self.min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        
        # DDP settings
        self.backend = 'nccl'  # 'nccl', 'gloo', etc.
        self.compile = False  # use PyTorch 2.0 to compile the model to be faster

    def get_dtype(self):
        if self.dtype == 'bfloat16':
            return torch.bfloat16
        elif self.dtype == 'float16':
            return torch.float16
        return torch.float32

    def __str__(self):
        return (
            f"Training Configuration:\n"
            f"  Device: {self.device} ({self.n_gpus} GPUs available)\n"
            f"  Batch Size: {self.batch_size}\n"
            f"  Learning Rate: {self.learning_rate}\n"
            f"  Data Type: {self.dtype}\n"
            f"  Model Size (Decoder): {self.decoder_n_embd}d, {self.decoder_n_layer}l, {self.decoder_n_head}h"
        )
