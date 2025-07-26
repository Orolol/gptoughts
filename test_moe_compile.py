"""Test MOE-MLA model compilation with gradient checkpointing"""

import torch
import torch.nn as nn
from models.models.moe_mla_model import create_moe_mla_model

# Test compilation
print("Testing MOE-MLA model compilation with gradient checkpointing...")

# Create a small model with gradient checkpointing
model = create_moe_mla_model(
    size='small',
    n_layer=4,
    num_experts=4,
    experts_per_token=2,
    block_size=512,
    use_gradient_checkpointing=True,
    use_fp8=False,
    use_dyt=True
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
print(f"Gradient checkpointing enabled: {model.config.use_gradient_checkpointing}")

# Test without compilation first
print("\n1. Testing forward pass WITHOUT compilation...")
batch_size = 2
seq_len = 128
input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
targets = input_ids.clone()

# Forward and backward pass
model.train()
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    logits, loss = model(input_ids, targets)
    loss.backward()

print(f"✓ Forward/backward pass successful without compilation")
print(f"  Loss: {loss.item():.4f}")

# Clear gradients
model.zero_grad()
torch.cuda.empty_cache()

# Now test with compilation
print("\n2. Testing model compilation...")
try:
    # Compile the model
    compiled_model = torch.compile(model, mode="default")
    print("✓ Model compiled successfully")
    
    # Test forward pass with compiled model
    print("\n3. Testing forward pass WITH compilation and gradient checkpointing...")
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    targets = input_ids.clone()
    
    compiled_model.train()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits, loss = compiled_model(input_ids, targets)
        loss.backward()
    
    print(f"✓ Forward/backward pass successful with compilation!")
    print(f"  Loss: {loss.item():.4f}")
    
    # Test multiple iterations
    print("\n4. Testing multiple iterations with compiled model...")
    losses = []
    for i in range(5):
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)
        targets = input_ids.clone()
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, loss = compiled_model(input_ids, targets)
            loss.backward()
        
        losses.append(loss.item())
        compiled_model.zero_grad()
        
        if i == 0:
            print(f"  Iteration {i+1}: Loss = {loss.item():.4f}")
    
    print(f"✓ All {len(losses)} iterations completed successfully")
    print(f"  Average loss: {sum(losses)/len(losses):.4f}")
    
    print("\n✅ SUCCESS: MOE-MLA model now works with torch.compile and gradient checkpointing!")
    
except Exception as e:
    print(f"\n❌ ERROR during compilation or execution: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nTest completed!")