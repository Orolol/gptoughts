#!/usr/bin/env python3
"""
Script to update grid search results by extracting loss values from log files
"""

import os
import re
import csv
import sys
from pathlib import Path

def extract_metrics_from_log(log_file):
    """Extract final train/loss and val/loss from a log file"""
    train_loss = "N/A"
    val_loss = "N/A"
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Find all occurrences of train/loss=X.XXX
        train_losses = re.findall(r'train/loss=(\d+\.\d+)', content)
        if train_losses:
            train_loss = train_losses[-1]  # Get the last one
            
        # Find all occurrences of val/loss=X.XXX
        val_losses = re.findall(r'val/loss=(\d+\.\d+)', content)
        if val_losses:
            val_loss = val_losses[-1]  # Get the last one
            
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        
    return train_loss, val_loss

def update_results_csv(results_dir):
    """Update the results.csv file with extracted metrics"""
    csv_path = os.path.join(results_dir, 'results.csv')
    
    if not os.path.exists(csv_path):
        print(f"No results.csv found in {results_dir}")
        return
        
    # Read existing results
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    # Update each row with metrics from log files
    updated_count = 0
    for row in rows:
        exp_id = row['experiment_id']
        log_file = os.path.join(results_dir, f"{exp_id}.log")
        
        if os.path.exists(log_file):
            train_loss, val_loss = extract_metrics_from_log(log_file)
            
            # Update if we found values
            if train_loss != "N/A" or val_loss != "N/A":
                row['final_loss'] = train_loss
                row['val_loss'] = val_loss
                updated_count += 1
                print(f"Updated {exp_id}: train_loss={train_loss}, val_loss={val_loss}")
            else:
                print(f"No loss values found in {exp_id}.log")
        else:
            print(f"Log file not found: {log_file}")
    
    # Write updated results back
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nUpdated {updated_count} entries in {csv_path}")
    
    # Show summary of results
    print("\nTop 5 configurations by train loss:")
    print("-" * 80)
    
    # Filter and sort by train loss
    valid_rows = [r for r in rows if r['final_loss'] != 'N/A' and r['status'] == 'completed']
    if valid_rows:
        sorted_rows = sorted(valid_rows, key=lambda x: float(x['final_loss']))
        
        print(f"{'Optimizer':<15} {'Grad Clip':<10} {'FP8':<5} {'Train Loss':<12} {'Val Loss':<12}")
        print("-" * 80)
        for row in sorted_rows[:5]:
            print(f"{row['optimizer']:<15} {row['grad_clip']:<10} {row['use_fp8']:<5} "
                  f"{row['final_loss']:<12} {row['val_loss']:<12}")
    else:
        print("No valid results found")

def main():
    if len(sys.argv) < 2:
        # Try to find the most recent grid search results
        base_dir = "grid_search_results"
        if os.path.exists(base_dir):
            dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if dirs:
                # Sort by modification time
                dirs.sort(key=lambda d: os.path.getmtime(os.path.join(base_dir, d)), reverse=True)
                results_dir = os.path.join(base_dir, dirs[0])
                print(f"Using most recent results directory: {results_dir}")
            else:
                print("No results directories found")
                sys.exit(1)
        else:
            print("Usage: python update_grid_search_results.py [results_directory]")
            sys.exit(1)
    else:
        results_dir = sys.argv[1]
    
    if not os.path.isdir(results_dir):
        print(f"Directory not found: {results_dir}")
        sys.exit(1)
    
    update_results_csv(results_dir)

if __name__ == "__main__":
    main()