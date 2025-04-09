#!/usr/bin/env python
"""
Simplified Budget Calculator for Weather Transformer
Uses calibration-based estimates rather than theoretical calculations
"""

import os
import sys
import argparse
import yaml
import math

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Weather Transformer Budget Calculator - Convert between time and epochs"
    )
    
    # Basic configuration
    parser.add_argument("--config", type=str, default="config/configs.yaml",
                      help="Path to configuration file")
    parser.add_argument("--config_name", type=str, default="base",
                      help="Configuration name within YAML file")
    parser.add_argument("--num_nodes", type=int, default=1,
                      help="Number of nodes to use")
    parser.add_argument("--devices_per_node", type=int, default=8,
                      help="Number of devices per node")
    
    # Sub-commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Estimate epochs for a time budget using calibration data
    time_parser = subparsers.add_parser("time_to_epochs", help="Estimate epochs for a time budget")
    time_parser.add_argument("--hours", type=float, required=True, help="Time budget in hours")
    time_parser.add_argument("--minutes_per_epoch", type=float, help="Minutes per epoch (from calibration)")
    
    # Estimate time for a number of epochs using calibration data
    epochs_parser = subparsers.add_parser("epochs_to_time", help="Estimate time for specified epochs")
    epochs_parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    epochs_parser.add_argument("--minutes_per_epoch", type=float, help="Minutes per epoch (from calibration)")
    
    # Generate a time budget table for planning
    table_parser = subparsers.add_parser("budget_table", help="Generate budget table for planning")
    table_parser.add_argument("--min_nodes", type=int, default=1, help="Minimum number of nodes")
    table_parser.add_argument("--max_nodes", type=int, default=32, help="Maximum number of nodes")
    table_parser.add_argument("--time_budgets", type=str, default="0.5,1,2,4,8,24",
                            help="Comma-separated list of time budgets in hours")
    table_parser.add_argument("--minutes_per_epoch", type=float, required=True, 
                            help="Minutes per epoch on a single node (from calibration)")
    
    # Settings for SLURM jobs
    slurm_parser = subparsers.add_parser("slurm_settings", help="Calculate settings for SLURM job")
    slurm_parser.add_argument("--hours", type=float, required=True, help="Wall-clock time budget in hours")
    slurm_parser.add_argument("--nodes", type=int, required=True, help="Number of nodes to use")
    slurm_parser.add_argument("--minutes_per_epoch", type=float, required=True, 
                            help="Minutes per epoch on a single node (from calibration)")
    slurm_parser.add_argument("--buffer_factor", type=float, default=1.2, 
                            help="Safety buffer factor (default: 1.2)")
    
    return parser.parse_args()

def load_config(config_file, config_name):
    """Load configuration from a YAML file"""
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    if config_name not in config_data:
        print(f"Error: Configuration '{config_name}' not found in {config_file}")
        print(f"Available configurations: {', '.join(config_data.keys())}")
        sys.exit(1)
    
    return config_data[config_name]

def get_default_minutes_per_epoch(config, num_nodes=1, devices_per_node=8):
    """
    Get a very rough estimate of minutes per epoch based on model size.
    This is just a fallback when no calibration data is available.
    """
    # Extract model size parameters
    model_config = config.get('model', {})
    embed_dim = model_config.get('embed_dim', 384)
    num_layers = model_config.get('num_layers', 12)
    
    # Extract image dimensions
    img_height, img_width = model_config.get('img_size', [360, 720])
    if isinstance(model_config.get('img_size'), list):
        img_height, img_width = model_config.get('img_size')
    
    # Extract batch size
    training_config = config.get('training', {})
    batch_size = training_config.get('batch_size', 32)
    
    # Very rough estimate - adjust constants based on actual observations
    # This is just a heuristic that considers model size, image dimensions, and parallelism
    base_minutes = (embed_dim/384) * (num_layers/12) * (img_height/360) * (img_width/720) * (batch_size/32) * 5
    
    # Scale by number of GPUs (assuming good scaling efficiency)
    total_gpus = num_nodes * devices_per_node
    minutes_per_epoch = base_minutes / total_gpus
    
    return minutes_per_epoch

def format_time(hours):
    """Format time in a human-readable format."""
    if hours < 1:
        return f"{hours*60:.1f}m"
    elif hours < 24:
        return f"{hours:.1f}h"
    else:
        days = hours / 24
        return f"{days:.1f}d"

def main():
    """Main function"""
    args = parse_args()
    
    # Load configuration if needed
    config = None
    if args.config and args.config_name:
        config = load_config(args.config, args.config_name)
    
    # Handle different commands
    if args.command == "time_to_epochs":
        # Get minutes per epoch from args or estimate from config
        minutes_per_epoch = args.minutes_per_epoch
        if minutes_per_epoch is None and config is not None:
            minutes_per_epoch = get_default_minutes_per_epoch(config, args.num_nodes, args.devices_per_node)
            print(f"No calibration data provided. Using rough estimate: {minutes_per_epoch:.2f} minutes per epoch")
        
        # Calculate epochs that can fit in the time budget
        hours = args.hours
        minutes = hours * 60
        epochs = int(minutes / minutes_per_epoch)
        
        print(f"\nWith a time budget of {hours} hours ({minutes:.1f} minutes):")
        print(f"- Estimated epochs: {epochs}")
        print(f"- Based on {minutes_per_epoch:.2f} minutes per epoch")
        print(f"- Using {args.num_nodes} nodes with {args.devices_per_node} GPUs each")
        
    elif args.command == "epochs_to_time":
        # Get minutes per epoch from args or estimate from config
        minutes_per_epoch = args.minutes_per_epoch
        if minutes_per_epoch is None and config is not None:
            minutes_per_epoch = get_default_minutes_per_epoch(config, args.num_nodes, args.devices_per_node)
            print(f"No calibration data provided. Using rough estimate: {minutes_per_epoch:.2f} minutes per epoch")
        
        # Calculate time needed for the specified epochs
        epochs = args.epochs
        minutes = epochs * minutes_per_epoch
        hours = minutes / 60
        
        print(f"\nTraining for {epochs} epochs:")
        print(f"- Estimated time: {hours:.2f} hours ({minutes:.1f} minutes)")
        print(f"- Based on {minutes_per_epoch:.2f} minutes per epoch")
        print(f"- Using {args.num_nodes} nodes with {args.devices_per_node} GPUs each")
        
    elif args.command == "budget_table":
        # Parse time budgets
        time_budgets = [float(t) for t in args.time_budgets.split(',')]
        
        # Print table header
        print("\n| Nodes | GPUs | ", end="")
        for hours in time_budgets:
            print(f"{format_time(hours)} | ", end="")
        print("")
        
        # Print separator
        print("|" + "-"*7 + "|" + "-"*6 + "|", end="")
        for _ in time_budgets:
            print("-"*7 + "|", end="")
        print("")
        
        # Generate rows - assuming linear scaling with number of nodes
        for nodes in range(args.min_nodes, args.max_nodes + 1):
            if nodes == 1 or nodes == 2 or nodes == 4 or nodes == 8 or nodes == 16 or nodes == 32 or nodes == 64:
                total_gpus = nodes * args.devices_per_node
                
                # Estimate minutes per epoch with this node count
                # Assuming linear scaling with node count (which is optimistic)
                minutes_per_epoch = args.minutes_per_epoch / nodes
                
                # Print row start
                print(f"| {nodes:<5d} | {total_gpus:<4d} | ", end="")
                
                # Calculate for each time budget
                for hours in time_budgets:
                    minutes = hours * 60
                    epochs = int(minutes / minutes_per_epoch)
                    print(f"{epochs:<5d} | ", end="")
                print("")
    
    elif args.command == "slurm_settings":
        # Calculate number of epochs that can fit in the time budget
        minutes_per_epoch_on_nodes = args.minutes_per_epoch / args.nodes
        hours = args.hours
        minutes = hours * 60
        epochs = int(minutes / minutes_per_epoch_on_nodes)
        
        # Calculate SLURM time limit with buffer
        buffered_minutes = minutes_per_epoch_on_nodes * epochs * args.buffer_factor
        buffered_hours = math.ceil(buffered_minutes / 60)
        
        print(f"\nSLURM Job Settings with {args.nodes} nodes:")
        print(f"- Wall-clock budget: {hours} hours")
        print(f"- Minutes per epoch on {args.nodes} nodes: {minutes_per_epoch_on_nodes:.2f}")
        print(f"- Recommended epochs: {epochs}")
        print(f"- SLURM time limit with {args.buffer_factor}x buffer: {buffered_hours} hours")
        print(f"\nSLURM command example:")
        print(f"  sbatch -N {args.nodes} -t {buffered_hours:02d}:00:00 submit_frontier.sh configs.yaml --config_name base --time_budget {hours}")
    
    else:
        print("Error: No command specified. Use --help for usage information.")

if __name__ == "__main__":
    main() 