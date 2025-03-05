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
    """Free GPU memory if usage is above the threshold with optimized defragmentation"""
    if not torch.cuda.is_available():
        return False
    
    # Get current memory usage
    try:
        allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
        reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
        
        # Calculate fragmentation
        fragmentation = 0
        if reserved > 0:
            fragmentation = (reserved - allocated) / reserved * 100
        
        # Check if we're above threshold (if specified)
        if threshold_mb is not None and allocated < threshold_mb and fragmentation < 20:
            return False
        
        # Create a separate stream for memory operations
        memory_stream = torch.cuda.Stream()
        with torch.cuda.stream(memory_stream):
            # First clear CUDA cache
            torch.cuda.empty_cache()
            
            # Force CUDA synchronization
            torch.cuda.synchronize()
            
            # Run Python's garbage collection
            gc.collect()
            
            # Advanced defragmentation technique for high fragmentation
            if fragmentation > 30:  # Only do this for high fragmentation
                # Calculate size to use for defrag
                defrag_size = min(int(reserved * 0.3), 512 * 1024 * 1024) // 4  # Limit to 512MB
                
                try:
                    # Try to allocate a tensor that forces defragmentation
                    tensors = []
                    
                    # Allocate in smaller chunks for better defrag
                    chunk_size = defrag_size // 8
                    for _ in range(8):
                        tensors.append(torch.zeros(chunk_size, dtype=torch.half, device='cuda'))
                    
                    # Free tensors explicitly
                    for t in tensors:
                        del t
                    tensors = []
                except RuntimeError:
                    # If we fail, it's likely because we're low on memory
                    pass
        
        # Wait for memory operations to complete
        torch.cuda.current_stream().wait_stream(memory_stream)
        
        # Get updated memory stats
        new_allocated = torch.cuda.memory_allocated() / (1024**2)
        new_reserved = torch.cuda.memory_reserved() / (1024**2)
        
        # Calculate if we've effectively reduced fragmentation
        if new_reserved > 0:
            new_fragmentation = (new_reserved - new_allocated) / new_reserved * 100
            return new_fragmentation < fragmentation or new_allocated < allocated
        
        return new_allocated < allocated
    except Exception as e:
        # Fallback to basic method if something fails
        print(f"Memory optimization error: {e}, using fallback method")
        torch.cuda.empty_cache()
        gc.collect()
        return True

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
    """Setup optimal memory configuration for training with advanced techniques"""
    # Get GPU information for adaptive configuration
    gpu_name = ""
    gpu_memory_gb = 0
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Tune garbage collection based on GPU memory
    if gpu_memory_gb >= 24:  # High-end GPUs (e.g., A100, H100, RTX 4090)
        # For large memory GPUs, make GC less aggressive
        gc.set_threshold(2000, 15, 15)
    elif gpu_memory_gb >= 12:  # Mid-range GPUs (e.g., RTX 3080, A5000)
        gc.set_threshold(1000, 12, 12)
    else:  # Low-memory GPUs
        # More aggressive GC for smaller GPUs
        gc.set_threshold(700, 10, 10)
    
    # Set PyTorch memory allocation strategy
    if torch.cuda.is_available():
        # Detect GPU architecture for optimal settings
        cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
        
        # Set environment variables based on GPU type and memory size
        if "rtx" in gpu_name or "titan" in gpu_name or "quadro" in gpu_name:
            # Gaming/Prosumer GPUs (NVIDIA GeForce RTX, TITAN, Quadro)
            if gpu_memory_gb > 20:  # High-end consumer GPU (>20GB)
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True,garbage_collection_threshold:0.8,roundup_power2:True"
            else:  # Mid-range or lower consumer GPU
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True,garbage_collection_threshold:0.6,roundup_power2:True"
        elif "a100" in gpu_name or "h100" in gpu_name or "a10" in gpu_name:
            # Data center GPUs
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True,garbage_collection_threshold:0.8,roundup_power2:True"
        else:
            # Default for other GPUs
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True,garbage_collection_threshold:0.7,roundup_power2:True"
        
        # Clear any cached memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Try to defragment memory with multiple passes in a separate stream
        try:
            # Use a dedicated stream for defragmentation
            defrag_stream = torch.cuda.Stream()
            with torch.cuda.stream(defrag_stream):
                total = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated()
                
                # Calculate optimal defrag chunk sizes based on GPU memory
                max_chunk_mb = int(min(256, gpu_memory_gb * 32))  # Limit to 256MB per chunk
                num_chunks = 4 if gpu_memory_gb < 12 else 8  # Use fewer chunks on smaller GPUs
                
                # Only try to defragment if substantial memory is available
                if allocated < total * 0.6:
                    # Allocate progressively larger chunks for better defragmentation
                    available_mb = (total - allocated) / (1024 * 1024)
                    chunk_sizes_mb = [
                        min(max_chunk_mb, int(available_mb * factor / num_chunks))
                        for factor in [0.1, 0.2, 0.3, 0.4]
                    ]
                    
                    for chunk_mb in chunk_sizes_mb:
                        if chunk_mb <= 0:
                            continue
                            
                        try:
                            # Convert MB to bytes and elements
                            chunk_bytes = chunk_mb * 1024 * 1024
                            
                            # Use float16 for memory efficiency
                            elements = chunk_bytes // 2  # 2 bytes per float16
                            
                            tensors = []
                            # Allocate multiple smaller tensors for better defrag
                            mini_chunks = min(num_chunks, max(2, int(chunk_mb / 32)))
                            mini_elements = elements // mini_chunks
                            
                            for _ in range(mini_chunks):
                                if mini_elements > 0:
                                    tensors.append(torch.zeros(mini_elements, dtype=torch.float16, device='cuda'))
                            
                            # Free immediately to hold cache but release tensors
                            for t in tensors:
                                del t
                            tensors = []
                            
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        except RuntimeError as e:
                            # If we hit OOM, log and stop the defrag attempts
                            print(f"Memory optimization limited: {e}")
                            break
            
            # Wait for defrag operations to complete
            torch.cuda.current_stream().wait_stream(defrag_stream)
            
            # Force a final synchronization and garbage collection
            torch.cuda.synchronize()
            gc.collect()
            
            # Print memory stats after optimization
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            
            print(f"Memory optimization complete. Allocated: {allocated_gb:.2f}GB, Reserved: {reserved_gb:.2f}GB")
            
        except Exception as e:
            print(f"Memory optimization could not complete: {e}")
