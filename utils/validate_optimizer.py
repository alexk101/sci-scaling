#!/usr/bin/env python
"""
Utility to validate and fix DeepSpeed optimizer setup to prevent
KeyError: 0 and other related issues.
"""

import logging
import torch
import sys
from typing import Dict, Any, Optional, List

logger = logging.getLogger("validate_optimizer")

def validate_and_fix_deepspeed_optimizer(optimizer) -> bool:
    """Validates and fixes DeepSpeed optimizer internal state to prevent common errors.
    
    Args:
        optimizer: DeepSpeed optimizer instance
        
    Returns:
        bool: True if validation passed or fixes were applied successfully
    """
    # Check if this is a DeepSpeed optimizer
    if not hasattr(optimizer, "averaged_gradients") or not hasattr(optimizer, "params_in_partition"):
        logger.info("Not a DeepSpeed ZeRO optimizer or necessary attributes not found")
        return True
    
    # Check the types of the key attributes
    avg_grad_type = type(optimizer.averaged_gradients)
    params_type = type(optimizer.params_in_partition)
    
    logger.info(f"DeepSpeed optimizer state check:")
    logger.info(f"  averaged_gradients type: {avg_grad_type}")
    logger.info(f"  params_in_partition type: {params_type}")
    
    # If both are dictionaries, we can proceed with key validation
    if isinstance(optimizer.averaged_gradients, dict) and isinstance(optimizer.params_in_partition, dict):
        avg_grad_keys = list(optimizer.averaged_gradients.keys())
        params_keys = list(optimizer.params_in_partition.keys())
        
        logger.info(f"  averaged_gradients keys: {avg_grad_keys}")
        logger.info(f"  params_in_partition keys: {params_keys}")
        
        # Check for key mismatches
        if set(avg_grad_keys) != set(params_keys):
            logger.warning("Mismatch between averaged_gradients and params_in_partition keys")
            
            # Create a merged set of keys
            all_keys = set(avg_grad_keys).union(set(params_keys))
            
            # Ensure all keys exist in both dictionaries
            for key in all_keys:
                if key not in optimizer.averaged_gradients:
                    logger.info(f"Adding missing key {key} to averaged_gradients")
                    # Get corresponding parameter
                    if key in optimizer.params_in_partition:
                        param = optimizer.params_in_partition[key]
                        # Create zero gradient with same shape as param
                        optimizer.averaged_gradients[key] = torch.zeros_like(param)
                
                if key not in optimizer.params_in_partition:
                    logger.info(f"Adding missing key {key} to params_in_partition")
                    # We shouldn't get here, but if we do, we'll need to create a placeholder
                    if key in optimizer.averaged_gradients:
                        grad = optimizer.averaged_gradients[key]
                        # Create empty param with same shape as grad
                        optimizer.params_in_partition[key] = torch.zeros_like(grad)
        
        # Check for key 0 specifically (common issue)
        if 0 not in optimizer.averaged_gradients and 0 in optimizer.params_in_partition:
            logger.info("Key 0 missing from averaged_gradients - adding it")
            optimizer.averaged_gradients[0] = torch.zeros_like(optimizer.params_in_partition[0])
        
        if 0 not in optimizer.params_in_partition and 0 in optimizer.averaged_gradients:
            logger.info("Key 0 missing from params_in_partition - adding it")
            optimizer.params_in_partition[0] = torch.zeros_like(optimizer.averaged_gradients[0])
        
        # Verify fix was successful
        fixed_avg_grad_keys = list(optimizer.averaged_gradients.keys())
        fixed_params_keys = list(optimizer.params_in_partition.keys())
        
        if set(fixed_avg_grad_keys) != set(fixed_params_keys):
            logger.error("Failed to fix key mismatch in DeepSpeed optimizer dictionaries")
            return False
        
        if 0 not in optimizer.averaged_gradients or 0 not in optimizer.params_in_partition:
            logger.error("Failed to ensure key 0 is in both dictionaries")
            return False
        
        logger.info("DeepSpeed optimizer dictionary validation passed!")
        return True
    
    # Handle the case where one or both are lists
    elif isinstance(optimizer.averaged_gradients, list) and isinstance(optimizer.params_in_partition, list):
        logger.info("DeepSpeed optimizer is using lists for both parameters - this is an expected configuration")
        
        # For list-based storage, check lengths match
        if len(optimizer.averaged_gradients) != len(optimizer.params_in_partition):
            logger.warning(f"Length mismatch: averaged_gradients has {len(optimizer.averaged_gradients)} elements, " 
                          f"but params_in_partition has {len(optimizer.params_in_partition)} elements")
            # Try to fix by extending the shorter list
            if len(optimizer.averaged_gradients) < len(optimizer.params_in_partition):
                logger.info("Extending averaged_gradients list to match params_in_partition length")
                while len(optimizer.averaged_gradients) < len(optimizer.params_in_partition):
                    # Add zero tensors of same shape
                    idx = len(optimizer.averaged_gradients)
                    if idx < len(optimizer.params_in_partition):
                        optimizer.averaged_gradients.append(torch.zeros_like(optimizer.params_in_partition[idx]))
            elif len(optimizer.params_in_partition) < len(optimizer.averaged_gradients):
                logger.info("Extending params_in_partition list to match averaged_gradients length")
                while len(optimizer.params_in_partition) < len(optimizer.averaged_gradients):
                    # Add zero tensors of same shape
                    idx = len(optimizer.params_in_partition)
                    if idx < len(optimizer.averaged_gradients):
                        optimizer.params_in_partition.append(torch.zeros_like(optimizer.averaged_gradients[idx]))
        
        # Log the fixed state
        logger.info(f"List lengths: averaged_gradients={len(optimizer.averaged_gradients)}, "
                    f"params_in_partition={len(optimizer.params_in_partition)}")
        
        # Check that index 0 exists in both lists (if they have elements)
        if len(optimizer.averaged_gradients) > 0 and len(optimizer.params_in_partition) > 0:
            logger.info("Both lists have elements, which should be sufficient for the optimizer")
            return True
        else:
            logger.error("One or both lists are empty, which will cause issues")
            return False
    
    # Handle the MIXED case where one is a dict and one is a list (this is the problematic case)
    elif isinstance(optimizer.averaged_gradients, dict) and isinstance(optimizer.params_in_partition, list):
        logger.warning("MIXED STRUCTURE: averaged_gradients is a dict but params_in_partition is a list")
        
        # Convert to consistent format - in single-task multi-GPU setup, list format is generally safer
        try:
            # Check if the dict is empty or small
            if len(optimizer.averaged_gradients) == 0 or len(optimizer.averaged_gradients) < len(optimizer.params_in_partition):
                logger.info("averaged_gradients dict is empty or smaller than params_in_partition list - converting to list")
                
                # Convert averaged_gradients to a list matching params_in_partition
                new_averaged_gradients = []
                for i, param in enumerate(optimizer.params_in_partition):
                    new_averaged_gradients.append(torch.zeros_like(param))
                
                # Replace the dict with our new list
                optimizer.averaged_gradients = new_averaged_gradients
                logger.info(f"Converted averaged_gradients to list with {len(optimizer.averaged_gradients)} elements")
                
                return True
            else:
                # Dict has elements, convert params_in_partition to dict
                logger.info("averaged_gradients has keys, converting params_in_partition to dict to match")
                
                # Convert params_in_partition list to dict
                new_params = {}
                for i, param in enumerate(optimizer.params_in_partition):
                    new_params[i] = param
                
                # Replace list with dict
                optimizer.params_in_partition = new_params
                logger.info(f"Converted params_in_partition to dict with keys: {list(optimizer.params_in_partition.keys())}")
                
                # Ensure all keys from averaged_gradients exist in params_in_partition
                for key in optimizer.averaged_gradients.keys():
                    if key not in optimizer.params_in_partition:
                        logger.info(f"Adding missing key {key} to params_in_partition")
                        optimizer.params_in_partition[key] = torch.zeros_like(optimizer.averaged_gradients[key])
                
                # Ensure key 0 exists in both
                if 0 not in optimizer.averaged_gradients and 0 in optimizer.params_in_partition:
                    logger.info("Key 0 missing from averaged_gradients - adding it")
                    optimizer.averaged_gradients[0] = torch.zeros_like(optimizer.params_in_partition[0])
                
                logger.info("Successfully converted mixed structure to consistent dict structure")
                return True
        except Exception as e:
            logger.error(f"Error while converting mixed structure: {str(e)}")
            # Emergency fallback: reset averaged_gradients to empty list and sync with params_in_partition
            try:
                logger.info("Attempting emergency structure reset")
                optimizer.averaged_gradients = []
                for param in optimizer.params_in_partition:
                    optimizer.averaged_gradients.append(torch.zeros_like(param))
                logger.info("Emergency reset successful")
                return True
            except:
                logger.error("Emergency reset failed")
                return False
    
    # Handle the case where params_in_partition is a dict but averaged_gradients is a list
    elif isinstance(optimizer.params_in_partition, dict) and isinstance(optimizer.averaged_gradients, list):
        logger.warning("MIXED STRUCTURE: params_in_partition is a dict but averaged_gradients is a list")
        
        try:
            # Convert averaged_gradients list to dict
            new_averaged_gradients = {}
            for i, grad in enumerate(optimizer.averaged_gradients):
                new_averaged_gradients[i] = grad
            
            # Replace list with dict
            optimizer.averaged_gradients = new_averaged_gradients
            logger.info(f"Converted averaged_gradients to dict with keys: {list(optimizer.averaged_gradients.keys())}")
            
            # Ensure all keys from params_in_partition exist in averaged_gradients
            for key in optimizer.params_in_partition.keys():
                if key not in optimizer.averaged_gradients:
                    logger.info(f"Adding missing key {key} to averaged_gradients")
                    optimizer.averaged_gradients[key] = torch.zeros_like(optimizer.params_in_partition[key])
            
            # Ensure key 0 exists in both
            if 0 not in optimizer.params_in_partition and 0 in optimizer.averaged_gradients:
                logger.info("Key 0 missing from params_in_partition - adding it")
                optimizer.params_in_partition[0] = torch.zeros_like(optimizer.averaged_gradients[0])
            
            logger.info("Successfully converted mixed structure to consistent dict structure")
            return True
        except Exception as e:
            logger.error(f"Error while converting mixed structure: {str(e)}")
            # Emergency fallback: try to convert params_in_partition to list
            try:
                logger.info("Attempting emergency structure reset")
                new_params = []
                for i in range(len(optimizer.averaged_gradients)):
                    if i in optimizer.params_in_partition:
                        new_params.append(optimizer.params_in_partition[i])
                    else:
                        new_params.append(torch.zeros_like(optimizer.averaged_gradients[i]))
                optimizer.params_in_partition = new_params
                logger.info("Emergency reset successful")
                return True
            except:
                logger.error("Emergency reset failed")
                return False
    
    # Unknown structure
    else:
        logger.error(f"Unknown structure: averaged_gradients is {avg_grad_type}, params_in_partition is {params_type}")
        return False

def log_optimizer_state(optimizer):
    """Logs detailed information about the optimizer state.
    
    Args:
        optimizer: PyTorch or DeepSpeed optimizer
    """
    logger.info(f"Optimizer type: {type(optimizer)}")
    
    # For any PyTorch optimizer
    if hasattr(optimizer, "param_groups"):
        logger.info(f"Number of parameter groups: {len(optimizer.param_groups)}")
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"Group {i}:")
            logger.info(f"  Learning rate: {group.get('lr', 'N/A')}")
            logger.info(f"  Weight decay: {group.get('weight_decay', 'N/A')}")
            logger.info(f"  Number of parameters: {len(group.get('params', []))}")
    
    # DeepSpeed specific
    if hasattr(optimizer, "optimizer"):
        logger.info("DeepSpeed optimizer wrapper detected")
        inner_opt = optimizer.optimizer
        logger.info(f"Inner optimizer type: {type(inner_opt)}")
    
    # ZeRO specific attributes - handle different types safely
    if hasattr(optimizer, "averaged_gradients"):
        logger.info(f"averaged_gradients type: {type(optimizer.averaged_gradients)}")
        if isinstance(optimizer.averaged_gradients, dict):
            logger.info(f"averaged_gradients keys: {list(optimizer.averaged_gradients.keys())}")
        else:
            logger.info(f"averaged_gradients is not a dictionary but a {type(optimizer.averaged_gradients)}")
            if hasattr(optimizer.averaged_gradients, "__len__"):
                logger.info(f"averaged_gradients length: {len(optimizer.averaged_gradients)}")
    
    if hasattr(optimizer, "params_in_partition"):
        logger.info(f"params_in_partition type: {type(optimizer.params_in_partition)}")
        if isinstance(optimizer.params_in_partition, dict):
            logger.info(f"params_in_partition keys: {list(optimizer.params_in_partition.keys())}")
        elif isinstance(optimizer.params_in_partition, list):
            logger.info(f"params_in_partition is a list with {len(optimizer.params_in_partition)} elements")
        else:
            logger.info(f"params_in_partition is {type(optimizer.params_in_partition)}")

def check_gradient_flow(model, verbose=False):
    """Checks if gradients are flowing properly through the model.
    
    Args:
        model: PyTorch model
        verbose: Whether to print details for each parameter
        
    Returns:
        bool: True if gradients are flowing properly
    """
    all_grads_ok = True
    total_params = 0
    params_with_grad = 0
    params_without_grad = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        
        if param.requires_grad:
            has_grad = param.grad is not None
            
            if has_grad:
                params_with_grad += 1
                grad_norm = param.grad.data.norm(2).item()
                if verbose:
                    logger.info(f"{name}: grad_norm = {grad_norm}")
            else:
                params_without_grad += 1
                all_grads_ok = False
                if verbose:
                    logger.info(f"{name}: No gradient!")
    
    logger.info(f"Gradient flow summary:")
    logger.info(f"  Total parameters: {total_params}")
    logger.info(f"  Parameters with gradients: {params_with_grad}")
    logger.info(f"  Parameters without gradients: {params_without_grad}")
    
    return all_grads_ok

def setup_logger(log_dir="logs", log_level=logging.INFO):
    """Sets up the validate_optimizer logger with proper file handling.
    
    Args:
        log_dir: Directory for logs
        log_level: Logging level
    """
    global logger
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set up proper logging
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_dir is provided
    if log_dir:
        import os
        os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(f"{log_dir}/optimizer_validation.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    logger.info(f"Logger setup complete for validate_optimizer") 