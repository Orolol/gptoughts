#!/bin/bash
# Script to run the minimal debug example for MLA model

# Set environment variables for debugging
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=1 
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Clear previous output file
rm -f debug_output.log

# Run the debug script with extensive logging
python run_mla_debug.py | tee debug_output.log

echo "Debug output saved to debug_output.log"