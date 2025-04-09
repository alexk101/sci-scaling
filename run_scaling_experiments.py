import yaml
import subprocess
import argparse
from typing import List
from pathlib import Path
import copy


def create_scaled_yaml_config(
    base_config_path: str,
    base_config_name: str,
    scale_type: str,
    scale_factor: float,
    output_dir: str = "experiments"
) -> str:
    """Create a new YAML configuration by scaling a base configuration.
    
    Args:
        base_config_path: Path to the base YAML configuration file
        base_config_name: Name of the base configuration to scale
        scale_type: Type of scaling to apply
        scale_factor: Scaling factor to apply
        output_dir: Directory to save the new configuration
        
    Returns:
        Path to the new configuration file
    """
    # Load original configurations from the YAML file
    with open(base_config_path, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    # Get the base configuration to scale
    if base_config_name not in all_configs:
        raise ValueError(f"Base configuration '{base_config_name}' not found in {base_config_path}")
    
    base_config = all_configs[base_config_name]
    
    # Create a deep copy of the base configuration
    scaled_config = copy.deepcopy(base_config)
    
    # Generate a name for the scaled configuration
    experiment_name = f"{scale_type}_scale_{scale_factor}"
    
    # Update configuration name and description
    scaled_config["name"] = experiment_name
    scaled_config["description"] = f"Scaling experiment for {scale_type} with factor {scale_factor}"
    
    # Apply scaling based on the scale type
    if scale_type == "model":
        scaled_config["num_layers"] = int(base_config.get("num_layers", 12) * scale_factor)
        scaled_config["num_heads"] = int(base_config.get("num_heads", 8) * scale_factor)
        
    elif scale_type == "embedding":
        scaled_config["embed_dim"] = int(base_config.get("embed_dim", 384) * scale_factor)
        
    elif scale_type == "sequence":
        scaled_config["sequence_length"] = int(base_config.get("sequence_length", 4) * scale_factor)
        
    elif scale_type == "resolution":
        img_size = base_config.get("img_size", [360, 720])
        scaled_config["img_size"] = [int(dim * scale_factor) for dim in img_size]
        
    elif scale_type == "batch":
        scaled_config["batch_size"] = int(base_config.get("batch_size", 32) * scale_factor)
        # Scale learning rate with batch size (square root scaling)
        scaled_config["learning_rate"] = base_config.get("learning_rate", 1e-4) * (scale_factor ** 0.5)
        
    elif scale_type == "data":
        # Adjust data subset size
        scaled_config["data_subset"] = min(1.0, base_config.get("data_subset", 1.0) * scale_factor)
        
    elif scale_type == "precision":
        # Map scale factor to precision: 0=fp32, 1=fp16, 2=bf16
        precision_map = {0: "fp32", 1: "fp16", 2: "bf16"}
        precision_idx = int(scale_factor)
        if precision_idx not in precision_map:
            raise ValueError(f"precision scale factor must be 0, 1, or 2, got {scale_factor}")
        scaled_config["mixed_precision"] = precision_map[precision_idx]
        
    elif scale_type == "node":
        scaled_config["num_nodes"] = int(scale_factor)
        
    else:
        raise ValueError(f"Unknown scale type: {scale_type}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the scaled configuration to a new YAML file
    config_file = output_path / f"{experiment_name}.yaml"
    
    # Create a new YAML with just this configuration
    with open(config_file, "w") as f:
        yaml.dump({experiment_name: scaled_config}, f)
    
    return str(config_file)


def run_experiment(
    config_path: str,
    config_name: str,
    strategy: str = "deepspeed",
    precision: str = None,
    devices: int = None,
    num_nodes: int = None
) -> None:
    """Run a single experiment.
    
    Args:
        config_path: Path to the configuration file
        config_name: Name of the configuration to use
        strategy: Training strategy to use
        precision: Precision to use (overrides configuration)
        devices: Number of devices to use (overrides configuration)
        num_nodes: Number of nodes to use (overrides configuration)
    """
    cmd = ["python", "train.py", "--config", config_path, "--config_name", config_name]
    
    if strategy:
        cmd.extend(["--strategy", strategy])
    
    if precision:
        cmd.extend(["--precision", precision])
    
    if devices:
        cmd.extend(["--devices", str(devices)])
    
    if num_nodes:
        cmd.extend(["--num_nodes", str(num_nodes)])
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_scaling_experiments(
    base_config_path: str,
    base_config_name: str,
    scale_types: List[str],
    scale_factors: List[float],
    strategy: str = "deepspeed",
    num_nodes: int = 1,
    devices_per_node: int = 8
) -> None:
    """Run multiple scaling experiments.
    
    Args:
        base_config_path: Path to the base configuration file
        base_config_name: Name of the base configuration to scale
        scale_types: List of scaling types to apply
        scale_factors: List of scaling factors to apply
        strategy: Training strategy to use
        num_nodes: Number of nodes to use
        devices_per_node: Number of devices per node
    """
    # Run experiments for each scale type and factor
    for scale_type in scale_types:
        for scale_factor in scale_factors:
            experiment_name = f"{scale_type}_scale_{scale_factor}"
            print(f"Running experiment: {experiment_name}")
            
            # Create experiment config
            config_path = create_scaled_yaml_config(
                base_config_path=base_config_path,
                base_config_name=base_config_name,
                scale_type=scale_type,
                scale_factor=scale_factor
            )
            
            # Calculate total devices
            total_devices = num_nodes * devices_per_node
            
            # Run the experiment
            run_experiment(
                config_path=config_path,
                config_name=experiment_name,
                strategy=strategy,
                devices=total_devices,
                num_nodes=num_nodes
            )


def main():
    """Parse command line arguments and run experiments."""
    parser = argparse.ArgumentParser(description="Run scaling experiments")
    
    # Configuration
    parser.add_argument("--base_config", type=str, default="config/configs.yaml",
                        help="Path to the base configuration file")
    parser.add_argument("--base_config_name", type=str, default="base",
                        help="Name of the base configuration to scale")
    
    # Scaling parameters
    parser.add_argument("--scale_types", type=str, nargs="+", 
                        default=["model", "embedding", "sequence", "batch"],
                        help="Types of scaling to apply")
    parser.add_argument("--scale_factors", type=float, nargs="+", 
                        default=[0.5, 1.0, 2.0, 4.0],
                        help="Scaling factors to apply")
    
    # Training parameters
    parser.add_argument("--strategy", type=str, default="deepspeed",
                        help="Training strategy to use")
    parser.add_argument("--num_nodes", type=int, default=1,
                        help="Number of nodes to use")
    parser.add_argument("--devices_per_node", type=int, default=8,
                        help="Number of devices per node")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run scaling experiments
    run_scaling_experiments(
        base_config_path=args.base_config,
        base_config_name=args.base_config_name,
        scale_types=args.scale_types,
        scale_factors=args.scale_factors,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        devices_per_node=args.devices_per_node
    )


if __name__ == "__main__":
    main()
