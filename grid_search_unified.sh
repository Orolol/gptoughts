#!/bin/bash

# Unified Grid Search Script for GPToughts
# This single script handles all grid search scenarios with different modes

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [MODE] [OPTIONS]

MODES:
  quick       - Quick test of optimizers with fixed parameters (default)
  optimizers  - Test optimizers, grad clip, and FP8
  galore      - Test GaLore variants with ranks 128 and 256
  custom      - Use custom parameter grids

OPTIONS:
  --model-type TYPE      Model type (default: mla)
  --model-size SIZE      Model size (default: small)
  --max-steps STEPS      Maximum training steps (default: 1000)
  --output-dir DIR       Output directory (default: grid_search_results)
  --parallel N           Number of parallel experiments (default: 1)

EXAMPLES:
  $0 quick                    # Quick optimizer comparison
  $0 optimizers               # Full optimizer grid search
  $0 galore                   # GaLore-specific testing
  $0 custom --config my.yaml  # Custom configuration

EOF
    exit 1
}

# Default values
MODE=${1:-quick}
MODEL_TYPE="mla"
MODEL_SIZE="small"
MAX_STEPS=1000
BASE_OUTPUT_DIR="grid_search_results"
PARALLEL=1

# Fixed parameters for consistency
BATCH_SIZE=8
LEARNING_RATE=5e-5
BLOCK_SIZE=1024

# Parse command line arguments
shift # Remove MODE from arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --output-dir)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Create timestamped results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${BASE_OUTPUT_DIR}/${MODE}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Initialize results file
RESULTS_CSV="${RESULTS_DIR}/results.csv"
RESULTS_LOG="${RESULTS_DIR}/experiment.log"

# Define parameter grids based on mode
case $MODE in
    quick)
        # Quick comparison of optimizer families
        OPTIMIZERS=("adamw" "lion" "apollo-mini" "galore-8bit")
        GRAD_CLIPS=(0 1.0)  # Include 0 for adamw
        USE_FP8_OPTIONS=(1)
        GALORE_RANKS=(128)
        GALORE_SCALES=(0.25)
        MAX_STEPS=500  # Override for quick test
        ;;
    
    optimizers)
        # Full optimizer grid search
        OPTIMIZERS=("apollo" "apollo-mini" "galore-8bit")
        GRAD_CLIPS=(0 0.5 1.0 2.0)  # Include 0 for adamw
        USE_FP8_OPTIONS=(0 1)
        GALORE_RANKS=(128 256)
        GALORE_SCALES=(0.25)
        ;;
    
    galore)
        # GaLore-specific testing
        OPTIMIZERS=("galore" "galore-8bit")
        GRAD_CLIPS=(0.5 1.0 2.0)
        USE_FP8_OPTIONS=(0 1)
        GALORE_RANKS=(128 256)
        GALORE_SCALES=(0.1 0.25 0.5)
        ;;
    
    custom)
        echo "Custom mode requires manual parameter definition in the script"
        exit 1
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        show_usage
        ;;
esac

# Print configuration
echo "=============================================" | tee "$RESULTS_LOG"
echo "Grid Search Configuration" | tee -a "$RESULTS_LOG"
echo "=============================================" | tee -a "$RESULTS_LOG"
echo "Mode: $MODE" | tee -a "$RESULTS_LOG"
echo "Model: $MODEL_TYPE ($MODEL_SIZE)" | tee -a "$RESULTS_LOG"
echo "Max steps: $MAX_STEPS" | tee -a "$RESULTS_LOG"
echo "Optimizers: ${OPTIMIZERS[@]}" | tee -a "$RESULTS_LOG"
echo "Grad clips: ${GRAD_CLIPS[@]}" | tee -a "$RESULTS_LOG"
echo "FP8 options: ${USE_FP8_OPTIONS[@]}" | tee -a "$RESULTS_LOG"
if [[ " ${OPTIMIZERS[@]} " =~ " galore" ]]; then
    echo "GaLore ranks: ${GALORE_RANKS[@]}" | tee -a "$RESULTS_LOG"
    echo "GaLore scales: ${GALORE_SCALES[@]}" | tee -a "$RESULTS_LOG"
fi
echo "Results directory: $RESULTS_DIR" | tee -a "$RESULTS_LOG"
echo "=============================================" | tee -a "$RESULTS_LOG"

# Initialize CSV with headers
echo "experiment_id,optimizer,grad_clip,use_fp8,galore_rank,galore_scale,final_loss,val_loss,time_seconds,status" > "$RESULTS_CSV"

# Function to extract metrics from logs
extract_metrics() {
    local log_file=$1
    local metric_name=${2:-"loss"}
    
    if [ -f "$log_file" ]; then
        # Try different patterns to extract metrics
        local value=$(grep -oE "${metric_name}[:\s]+[0-9]+\.[0-9]+" "$log_file" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | tail -1)
        echo "${value:-N/A}"
    else
        echo "N/A"
    fi
}

# Function to run a single experiment
run_experiment() {
    local optimizer=$1
    local grad_clip=$2
    local use_fp8=$3
    local galore_rank=${4:-"N/A"}
    local galore_scale=${5:-"N/A"}
    
    # Create experiment ID
    local exp_id="${optimizer}_gc${grad_clip}_fp8${use_fp8}"
    if [[ "$optimizer" == "galore"* ]]; then
        exp_id="${exp_id}_r${galore_rank}_s${galore_scale}"
    fi
    
    local output_dir="${RESULTS_DIR}/${exp_id}"
    local log_file="${output_dir}.log"
    
    echo "" | tee -a "$RESULTS_LOG"
    echo "Starting experiment: $exp_id" | tee -a "$RESULTS_LOG"
    echo "  Optimizer: $optimizer" | tee -a "$RESULTS_LOG"
    if [[ "$grad_clip" == "0" ]] || [[ "$grad_clip" == "0.0" ]]; then
        echo "  Grad clip: 0 (disabled)" | tee -a "$RESULTS_LOG"
    else
        echo "  Grad clip: $grad_clip" | tee -a "$RESULTS_LOG"
    fi
    echo "  FP8: $use_fp8" | tee -a "$RESULTS_LOG"
    if [[ "$optimizer" == "galore"* ]]; then
        echo "  GaLore rank: $galore_rank" | tee -a "$RESULTS_LOG"
        echo "  GaLore scale: $galore_scale" | tee -a "$RESULTS_LOG"
    fi
    
    # Build command
    local cmd="python run_train.py"
    cmd="$cmd --model_type $MODEL_TYPE"
    cmd="$cmd --size $MODEL_SIZE"
    cmd="$cmd --batch_size $BATCH_SIZE"
    cmd="$cmd --block_size $BLOCK_SIZE"
    cmd="$cmd --learning_rate $LEARNING_RATE"
    cmd="$cmd --optimizer_type $optimizer"
    
    # Add grad_clip parameter
    cmd="$cmd --grad_clip $grad_clip"
    
    cmd="$cmd --max_iters $MAX_STEPS"
    cmd="$cmd --output_dir $output_dir"
    cmd="$cmd --eval_interval_steps 100"
    cmd="$cmd --log_interval_steps 10"
    cmd="$cmd --warmup_iters 100"
    cmd="$cmd --precision bf16-mixed"
    cmd="$cmd --optimize_attention"
    cmd="$cmd --preallocate_memory"
    cmd="$cmd --use_dyt"
    cmd="$cmd --weight_decay 0.1"
    cmd="$cmd --compile"
    cmd="$cmd --keep_checkpoints 0"  # Disable checkpointing for grid search
    
    # Add FP8 if enabled
    if [ "$use_fp8" = "1" ]; then
        cmd="$cmd --use_fp8"
    fi
    
    # Add GaLore-specific parameters
    if [[ "$optimizer" == "galore"* ]]; then
        cmd="$cmd --galore_rank $galore_rank"
        cmd="$cmd --galore_update_proj_gap 200"
        cmd="$cmd --galore_scale $galore_scale"
        cmd="$cmd --galore_proj_type std"
    fi
    
    # Run the experiment
    local start_time=$(date +%s)
    local status="completed"
    
    # Execute training
    if $cmd > "$log_file" 2>&1; then
        status="completed"
    else
        status="failed"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Extract metrics - look for the last occurrence of train/loss and val/loss
    local final_loss=$(grep -oE "train/loss=[0-9]+\.[0-9]+" "$log_file" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    local val_loss=$(grep -oE "val/loss=[0-9]+\.[0-9]+" "$log_file" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
    
    # Save result to CSV
    echo "$exp_id,$optimizer,$grad_clip,$use_fp8,$galore_rank,$galore_scale,$final_loss,$val_loss,$duration,$status" >> "$RESULTS_CSV"
    
    echo "  Status: $status (${duration}s)" | tee -a "$RESULTS_LOG"
    echo "  Final loss: $final_loss" | tee -a "$RESULTS_LOG"
    
    return 0
}

# Function to run experiments in parallel
run_parallel() {
    local jobs=()
    local job_count=0
    
    for optimizer in "${OPTIMIZERS[@]}"; do
        for grad_clip in "${GRAD_CLIPS[@]}"; do
            # Skip non-zero grad_clip values for adamw
            if [[ "$optimizer" == "adamw" ]] && [[ "$grad_clip" != "0" ]] && [[ "$grad_clip" != "0.0" ]]; then
                continue
            fi
            
            for use_fp8 in "${USE_FP8_OPTIONS[@]}"; do
                if [[ "$optimizer" == "galore"* ]]; then
                    # For GaLore, iterate through ranks and scales
                    for galore_rank in "${GALORE_RANKS[@]}"; do
                        for galore_scale in "${GALORE_SCALES[@]}"; do
                            # Run in background if parallel mode
                            if [ "$PARALLEL" -gt 1 ]; then
                                run_experiment "$optimizer" "$grad_clip" "$use_fp8" "$galore_rank" "$galore_scale" &
                                jobs+=($!)
                                job_count=$((job_count + 1))
                                
                                # Wait if we've reached the parallel limit
                                if [ ${#jobs[@]} -ge "$PARALLEL" ]; then
                                    wait "${jobs[0]}"
                                    jobs=("${jobs[@]:1}")
                                fi
                            else
                                run_experiment "$optimizer" "$grad_clip" "$use_fp8" "$galore_rank" "$galore_scale"
                            fi
                        done
                    done
                else
                    # For non-GaLore optimizers
                    if [ "$PARALLEL" -gt 1 ]; then
                        run_experiment "$optimizer" "$grad_clip" "$use_fp8" &
                        jobs+=($!)
                        job_count=$((job_count + 1))
                        
                        if [ ${#jobs[@]} -ge "$PARALLEL" ]; then
                            wait "${jobs[0]}"
                            jobs=("${jobs[@]:1}")
                        fi
                    else
                        run_experiment "$optimizer" "$grad_clip" "$use_fp8"
                    fi
                fi
            done
        done
    done
    
    # Wait for remaining jobs
    if [ "$PARALLEL" -gt 1 ]; then
        wait
    fi
    
    echo "" | tee -a "$RESULTS_LOG"
    echo "All experiments completed!" | tee -a "$RESULTS_LOG"
    echo "Total experiments: $job_count" | tee -a "$RESULTS_LOG"
}

# Count total experiments
count_experiments() {
    local count=0
    for optimizer in "${OPTIMIZERS[@]}"; do
        if [[ "$optimizer" == "galore"* ]]; then
            count=$((count + ${#GRAD_CLIPS[@]} * ${#USE_FP8_OPTIONS[@]} * ${#GALORE_RANKS[@]} * ${#GALORE_SCALES[@]}))
        elif [[ "$optimizer" == "adamw" ]]; then
            # AdamW only runs with grad_clip=0
            local adamw_count=0
            for grad_clip in "${GRAD_CLIPS[@]}"; do
                if [[ "$grad_clip" == "0" ]] || [[ "$grad_clip" == "0.0" ]]; then
                    adamw_count=$((adamw_count + 1))
                fi
            done
            count=$((count + adamw_count * ${#USE_FP8_OPTIONS[@]}))
        else
            count=$((count + ${#GRAD_CLIPS[@]} * ${#USE_FP8_OPTIONS[@]}))
        fi
    done
    echo $count
}

# Run experiments
TOTAL_EXPERIMENTS=$(count_experiments)
echo "" | tee -a "$RESULTS_LOG"
echo "Starting $TOTAL_EXPERIMENTS experiments..." | tee -a "$RESULTS_LOG"

run_parallel

# Generate summary report
echo "" | tee -a "$RESULTS_LOG"
echo "=============================================" | tee -a "$RESULTS_LOG"
echo "GRID SEARCH COMPLETE" | tee -a "$RESULTS_LOG"
echo "=============================================" | tee -a "$RESULTS_LOG"

# Show top results
echo "" | tee -a "$RESULTS_LOG"
echo "Top 5 configurations by loss:" | tee -a "$RESULTS_LOG"
echo "------------------------------" | tee -a "$RESULTS_LOG"
(head -n 1 "$RESULTS_CSV" && tail -n +2 "$RESULTS_CSV" | grep "completed" | sort -t',' -k7 -g | head -5) | column -t -s',' | tee -a "$RESULTS_LOG"

# Create analysis script
cat > "${RESULTS_DIR}/analyze.py" << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import sys
import os

# Load results
df = pd.read_csv('results.csv')
df_complete = df[df['status'] == 'completed'].copy()

if len(df_complete) == 0:
    print("No completed experiments found!")
    sys.exit(1)

# Convert numeric columns
numeric_cols = ['grad_clip', 'use_fp8', 'time_seconds']
for col in numeric_cols:
    if col in df_complete.columns:
        df_complete[col] = pd.to_numeric(df_complete[col], errors='coerce')

# Handle loss columns
for loss_col in ['final_loss', 'val_loss']:
    if loss_col in df_complete.columns:
        df_complete[loss_col] = df_complete[loss_col].replace('N/A', pd.NA)
        df_complete[loss_col] = pd.to_numeric(df_complete[loss_col], errors='coerce')

# Best configuration
print("\n=== BEST CONFIGURATION ===")
if 'final_loss' in df_complete.columns and df_complete['final_loss'].notna().any():
    best_idx = df_complete['final_loss'].idxmin()
    best = df_complete.loc[best_idx]
    print(f"Experiment: {best['experiment_id']}")
    print(f"Optimizer: {best['optimizer']}")
    print(f"Grad Clip: {best['grad_clip']}")
    print(f"FP8: {'Enabled' if best['use_fp8'] else 'Disabled'}")
    if best['optimizer'].startswith('galore'):
        print(f"GaLore Rank: {best['galore_rank']}")
        print(f"GaLore Scale: {best['galore_scale']}")
    print(f"Final Loss: {best['final_loss']:.6f}")
    if pd.notna(best.get('val_loss')):
        print(f"Val Loss: {best['val_loss']:.6f}")
    print(f"Time: {best['time_seconds']}s")

# Optimizer comparison
print("\n=== OPTIMIZER COMPARISON ===")
opt_stats = df_complete.groupby('optimizer').agg({
    'final_loss': ['mean', 'std', 'min', 'count'],
    'time_seconds': 'mean'
}).round(4)
print(opt_stats)

# Gradient clipping analysis
print("\n=== GRADIENT CLIPPING IMPACT ===")
gc_stats = df_complete.groupby('grad_clip').agg({
    'final_loss': ['mean', 'std', 'min']
}).round(4)
print(gc_stats)

# FP8 impact
print("\n=== FP8 IMPACT ===")
if df_complete['use_fp8'].nunique() > 1:
    fp8_stats = df_complete.groupby('use_fp8').agg({
        'final_loss': ['mean', 'std', 'min'],
        'time_seconds': 'mean'
    }).round(4)
    fp8_stats.index = ['FP8 Disabled', 'FP8 Enabled']
    print(fp8_stats)

# GaLore analysis if applicable
galore_df = df_complete[df_complete['optimizer'].str.startswith('galore')]
if len(galore_df) > 0:
    print("\n=== GALORE ANALYSIS ===")
    
    # Rank comparison
    if 'galore_rank' in galore_df.columns and galore_df['galore_rank'].notna().any():
        print("\nRank Comparison:")
        rank_stats = galore_df.groupby('galore_rank').agg({
            'final_loss': ['mean', 'min'],
            'time_seconds': 'mean'
        }).round(4)
        print(rank_stats)
    
    # Scale comparison
    if 'galore_scale' in galore_df.columns and galore_df['galore_scale'].notna().any():
        print("\nScale Comparison:")
        scale_stats = galore_df.groupby('galore_scale').agg({
            'final_loss': ['mean', 'min']
        }).round(4)
        print(scale_stats)
    
    # 8-bit vs standard
    if galore_df['optimizer'].nunique() > 1:
        print("\n8-bit vs Standard GaLore:")
        galore_type_stats = galore_df.groupby('optimizer').agg({
            'final_loss': ['mean', 'min'],
            'time_seconds': 'mean'
        }).round(4)
        print(galore_type_stats)

# Save detailed analysis
df_complete.to_csv('analysis_detailed.csv', index=False)
print(f"\nDetailed results saved to analysis_detailed.csv")

# Top configurations
print("\n=== TOP 10 CONFIGURATIONS ===")
top10 = df_complete.nsmallest(10, 'final_loss')[['experiment_id', 'optimizer', 'grad_clip', 'use_fp8', 'final_loss']]
if 'galore_rank' in df_complete.columns:
    galore_configs = df_complete[df_complete['optimizer'].str.startswith('galore')].nsmallest(10, 'final_loss')
    if len(galore_configs) > 0:
        top10 = df_complete.nsmallest(10, 'final_loss')[['experiment_id', 'optimizer', 'grad_clip', 'use_fp8', 'galore_rank', 'galore_scale', 'final_loss']]
print(top10.to_string(index=False))

# Summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total experiments: {len(df)}")
print(f"Completed: {len(df_complete)}")
print(f"Failed: {len(df[df['status'] == 'failed'])}")
print(f"Success rate: {len(df_complete)/len(df)*100:.1f}%")
print(f"Average experiment time: {df_complete['time_seconds'].mean():.1f}s")
print(f"Total time: {df_complete['time_seconds'].sum()/3600:.1f} hours")
EOF

chmod +x "${RESULTS_DIR}/analyze.py"

echo "" | tee -a "$RESULTS_LOG"
echo "Results saved to: $RESULTS_DIR" | tee -a "$RESULTS_LOG"
echo "" | tee -a "$RESULTS_LOG"
echo "To analyze results in detail, run:" | tee -a "$RESULTS_LOG"
echo "  cd $RESULTS_DIR && python analyze.py" | tee -a "$RESULTS_LOG"

# Make the script executable
chmod +x "$0"