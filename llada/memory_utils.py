import gc
import torch
import os
import threading
import time
from typing import Optional, Dict, List
from contextlib import contextmanager

class TensorPool:
    """A pool of reusable tensors to avoid memory allocation/deallocation overhead"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.pools = {}  # Pools for different shapes and dtypes
        self.active_tensors = set()  # Track tensors currently in use
        self.lock = threading.Lock()
        self.enabled = torch.cuda.is_available()
    
    def get(self, shape, dtype=torch.float):
        """Get a tensor from the pool or create a new one"""
        if not self.enabled:
            return torch.zeros(shape, dtype=dtype, device=self.device)
            
        with self.lock:
            key = (shape, dtype)
            if key not in self.pools:
                self.pools[key] = []
                
            if self.pools[key]:
                tensor = self.pools[key].pop()
                tensor.zero_()
            else:
                tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                
            self.active_tensors.add(tensor)
            return tensor
    
    def release(self, tensor):
        """Return a tensor to the pool"""
        if not self.enabled or tensor is None:
            return
            
        with self.lock:
            if tensor in self.active_tensors:
                key = (tensor.shape, tensor.dtype)
                self.pools[key].append(tensor)
                self.active_tensors.remove(tensor)
    
    def clear(self):
        """Clear all pools"""
        with self.lock:
            self.pools.clear()
            self.active_tensors.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Create a global tensor pool for reuse
tensor_pool = TensorPool()

class MemoryTracker:
    """Track memory usage during model training"""
    
    def __init__(self, report_interval: int = 100, enabled: bool = True):
        self.report_interval = report_interval
        self.enabled = enabled and torch.cuda.is_available()
        self.step = 0
        self.memory_stats = []
        self.peak_memory = 0
        self.tracking_thread = None
        self.stop_event = threading.Event()
    
    def start_continuous_tracking(self, interval_seconds: float = 1.0):
        """Start a background thread to continuously track memory usage"""
        if not self.enabled:
            return
            
        def tracking_worker():
            while not self.stop_event.is_set():
                self.update_stats()
                time.sleep(interval_seconds)
        
        self.stop_event.clear()
        self.tracking_thread = threading.Thread(target=tracking_worker, daemon=True)
        self.tracking_thread.start()
    
    def stop_continuous_tracking(self):
        """Stop the background memory tracking thread"""
        if self.tracking_thread is not None:
            self.stop_event.set()
            self.tracking_thread.join(timeout=2.0)
            self.tracking_thread = None
    
    def update_stats(self):
        """Update memory usage statistics"""
        if not self.enabled:
            return
            
        try:
            # Get current memory allocation
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            self.peak_memory = max(self.peak_memory, allocated)
            
            # Store stats
            self.memory_stats.append({
                'step': self.step,
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'peak_mb': self.peak_memory
            })
            
            # Keep only recent stats to avoid memory buildup
            if len(self.memory_stats) > 1000:
                self.memory_stats = self.memory_stats[-500:]
        except:
            # Silently fail if CUDA is not available
            pass
    
    def step_callback(self):
        """Call this method after each training step"""
        self.step += 1
        
        if self.enabled and self.step % self.report_interval == 0:
            self.update_stats()
            self.print_report()
    
    def print_report(self):
        """Print a memory usage report"""
        if not self.enabled or not self.memory_stats:
            return
            
        # Get the latest stats
        latest = self.memory_stats[-1]
        
        print(f"\n===== CUDA Memory Report (Step {self.step}) =====")
        print(f"Allocated: {latest['allocated_mb']:.1f} MB")
        print(f"Reserved:  {latest['reserved_mb']:.1f} MB")
        print(f"Peak:      {latest['peak_mb']:.1f} MB")
        
        # Calculate fragmentation
        if latest['reserved_mb'] > 0:
            fragmentation = (latest['reserved_mb'] - latest['allocated_mb']) / latest['reserved_mb'] * 100
            print(f"Estimated fragmentation: {fragmentation:.1f}%")
        
        # Calculate the change over last 10 steps
        if len(self.memory_stats) > 10:
            prev = self.memory_stats[-11]  # 10 steps ago
            change = latest['allocated_mb'] - prev['allocated_mb']
            print(f"Change over last 10 steps: {change:+.1f} MB")
        
        print("=============================================")
    
    def get_summary(self) -> Dict:
        """Get a summary of memory statistics"""
        if not self.enabled or not self.memory_stats:
            return {'peak_mb': 0, 'current_mb': 0, 'reserved_mb': 0}
            
        latest = self.memory_stats[-1]
        return {
            'peak_mb': self.peak_memory,
            'current_mb': latest['allocated_mb'],
            'reserved_mb': latest['reserved_mb']
        }

@contextmanager
def optimized_memory_context():
    """Context manager for optimized memory usage during critical operations"""
    # Cache the original state
    gc_enabled = gc.isenabled()
    
    try:
        # Disable Python garbage collection during the operation
        gc.disable()
        
        # Reserve CUDA memory before the operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Try to trigger garbage collection of CUDA tensors
            try:
                torch.cuda.synchronize()
            except:
                pass
        
        # Execute the wrapped code
        yield
    finally:
        # Restore garbage collection state
        if gc_enabled:
            gc.enable()
        
        # Clean up CUDA memory if needed
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except:
                pass

def free_memory(threshold_mb: Optional[int] = None):
    """Free GPU memory if usage is above the threshold"""
    if not torch.cuda.is_available():
        return False
        
    # Get current memory usage
    allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
    
    # Check if we're above threshold (if specified)
    if threshold_mb is not None and allocated < threshold_mb:
        return False
    
    # Trigger a more aggressive memory cleanup
    for _ in range(3):  # Multiple passes can help with fragmentation
        # First clear CUDA cache
        torch.cuda.empty_cache()
        
        # Force CUDA synchronization to complete any pending operations
        torch.cuda.synchronize()
        
        # Run Python's garbage collection
        gc.collect()
        
        # Try allocating and freeing a small tensor to help with defragmentation
        try:
            temp = torch.zeros(1024, 1024, device='cuda')
            del temp
        except:
            pass
    
    # Return True if we freed memory
    new_allocated = torch.cuda.memory_allocated() / (1024**2)
    return new_allocated < allocated

def set_memory_fraction(fraction: float = 0.85):
    """Set GPU memory fraction while avoiding setting it too aggressively"""
    if not torch.cuda.is_available():
        return
        
    # Set a reasonable fraction of GPU memory
    try:
        torch.cuda.set_per_process_memory_fraction(fraction)
    except:
        # If setting memory fraction fails, try alternative API
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:128,expandable_segments:True"

def setup_optimal_memory_config():
    """Setup optimal memory configuration for training"""
    # Tune garbage collection - less frequent but more thorough
    gc.set_threshold(1200, 10, 10)  # Make GC less aggressive but more thorough
    
    # Set PyTorch memory allocation strategy
    if torch.cuda.is_available():
        # Set environment variables for better memory allocation
        # Use smaller split size to reduce fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True,garbage_collection_threshold:0.6,roundup_power2:True"
        
        # Clear any cached memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Try to defragment memory with multiple passes
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated()
            
            # Only try to defragment if substantial memory is available
            if allocated < total * 0.7:
                # Do multiple passes of allocation/deallocation to help memory defrag
                for size_factor in [0.5, 0.3, 0.2]:
                    block_size = int((total - allocated) * size_factor)
                    if block_size > 0:
                        try:
                            temp = torch.empty(block_size // 8, device='cuda', dtype=torch.float16)
                            del temp
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        except RuntimeError:
                            # If we hit OOM, stop the defrag attempts
                            break
                
                # Force a final synchronization and garbage collection
                torch.cuda.synchronize()
                gc.collect()
        except Exception as e:
            print(f"Memory defragmentation failed: {e}")
