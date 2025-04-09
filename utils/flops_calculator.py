#!/usr/bin/env python
"""
FLOPs Calculator - Utilities for working with DeepSpeed's FLOPs profiler results
"""

import math
import yaml
import argparse
from typing import Dict, Any, Union, Optional
import json
import logging

def get_deepspeed_flops_config(enabled=True, profile_step=5, detailed=False):
    """
    Generate a DeepSpeed FLOPS profiler configuration.
    
    Args:
        enabled: Whether to enable the FLOPS profiler
        profile_step: Which step to profile
        detailed: Whether to generate detailed module-level profile
        
    Returns:
        A dictionary with DeepSpeed FLOPS profiler configuration
    """
    return {
        "flops_profiler": {
            "enabled": enabled,
            "profile_step": profile_step,
            "module_depth": -1,
            "top_modules": 3,
            "detailed": detailed,
            "output_file": None
        }
    }

def calculate_epochs_for_flops_budget(flops_per_epoch: float, flops_budget: float) -> int:
    """
    Calculate the number of epochs that can be trained within a given FLOPs budget.
    
    Args:
        flops_per_epoch: FLOPs per epoch (measured by DeepSpeed)
        flops_budget: Available FLOPs budget
        
    Returns:
        Number of epochs that can be completed within the FLOPs budget
    """
    if flops_per_epoch <= 0:
        return 0
        
    return max(1, int(flops_budget / flops_per_epoch))

def calculate_epochs_for_time_budget(minutes_per_epoch: float, time_budget_hours: float) -> int:
    """
    Calculate the number of epochs that can be trained within a given time budget.
    
    Args:
        minutes_per_epoch: Minutes per epoch (measured during training)
        time_budget_hours: Available time in hours
        
    Returns:
        Number of epochs that can be completed within the time budget
    """
    if minutes_per_epoch <= 0:
        return 0
        
    minutes_available = time_budget_hours * 60
    return max(1, int(minutes_available / minutes_per_epoch))

def format_flops(flops: float) -> str:
    """Format FLOPs in a human-readable format."""
    if flops == 0:
        return "0 FLOPS"
        
    units = ['', 'K', 'M', 'G', 'T', 'P', 'E']
    
    for unit in units:
        if flops < 1000:
            return f"{flops:.2f} {unit}FLOPS"
        flops /= 1000
    
    return f"{flops:.2f} ZFLOPS"

def analyze_flops_profile(profile_file):
    """Analyze a DeepSpeed FLOPs profile file."""
    try:
        with open(profile_file, 'r') as f:
            data = json.load(f)
            
        # Extract key metrics
        total_flops = data.get('total_flops', 0)
        model_flops_per_seq = data.get('model_flops_per_seq', 0)
        
        return {
            'total_flops': total_flops,
            'model_flops_per_seq': model_flops_per_seq,
            'model_tflops': total_flops / 1e12,
        }
    except Exception as e:
        logging.error(f"Error processing FLOPs profile: {e}")
        return None

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze DeepSpeed FLOPs profile')
    parser.add_argument('--profile_file', type=str, required=True,
                        help='Path to DeepSpeed FLOPs profile JSON file')
    args = parser.parse_args()
    
    logging.info(f"Processing DeepSpeed FLOPs profile from: {args.profile_file}")
    logging.info("This is a placeholder - in actual use, we rely on DeepSpeed's built-in profiler")
    
    results = analyze_flops_profile(args.profile_file)
    if results:
        for key, value in results.items():
            if isinstance(value, float):
                logging.info(f"{key}: {value:.2f}")
            else:
                logging.info(f"{key}: {value}")

if __name__ == "__main__":
    main() 