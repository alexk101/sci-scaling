import torch
import torch.backends
import argparse
import logging
from pathlib import Path
from utils.synthetic_data import create_test_data
from utils.config_utils import load_config, PrecisionType, LoggingBackend
from train import WeatherTrainer
from utils.logging import init_run_directory

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a Weather Transformer model locally")
    parser.add_argument("--config", type=str, default="config/configs.yaml",
                        help="Path to the YAML configuration file")
    parser.add_argument("--config_name", type=str, default="local",
                        help="Name of the configuration to use from the YAML file (default: local)")
    parser.add_argument("--enable_flops_profiler", action="store_true", help="Enable FLOPs profiling")
    parser.add_argument("--run_name", type=str, default=None, 
                        help="Unique name for this run. If not provided, a timestamp will be used.")
    
    args = parser.parse_args()
    
    # Initialize the run directory and configure logging
    run_dir = init_run_directory("logs")
    
    logging.info(f"Run ID: {run_dir.name}")
    
    # Check for external flash attention
    try:
        from flash_attn import flash_attn_func
        logging.info("External Flash Attention package is available")
        external_flash_available = True
    except ImportError:
        logging.info("External Flash Attention package not available")
        external_flash_available = False

    # Determine available hardware
    if torch.cuda.is_available():
        accelerator = "cuda"
        device_count = torch.cuda.device_count()
        strategy = "ddp" if device_count > 1 else "auto"
        logging.info(f"Using CUDA with {device_count} GPU(s)")
        
        # Check for PyTorch native Flash Attention
        pytorch_flash_available = (torch.__version__ >= "2.0.0" and 
                                  hasattr(torch.backends, 'cuda') and 
                                  hasattr(torch.backends.cuda, 'sdp_kernel'))
        
        if pytorch_flash_available:
            logging.info("PyTorch native Flash Attention is available (PyTorch 2.0+)")
            # Enable PyTorch native Flash Attention if external package is not available
            if not external_flash_available:
                logging.info("Enabling PyTorch native Flash Attention for CUDA")
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_math_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
        
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        accelerator = "mps"
        device_count = 1
        strategy = "auto"  # MPS doesn't support DDP
        logging.info("Using MPS (Metal Performance Shader) device")
        logging.info("Flash Attention not available for MPS, using standard attention")
        external_flash_available = False
        pytorch_flash_available = False
    else:
        accelerator = "cpu"
        device_count = 1
        strategy = "auto"
        logging.info("No GPU available, using CPU")
        logging.info("Flash Attention not available for CPU, using standard attention")
        external_flash_available = False
        pytorch_flash_available = False
    
    # Create synthetic data only if it doesn't exist
    data_dir = Path("data")
    if not (data_dir / "train").exists() or not (data_dir / "valid").exists():
        logging.info("Synthetic data not found. Generating...")
        data_dir = create_test_data()
    else:
        logging.info(f"Using existing data in {data_dir}")
    
    # Load configuration from YAML file
    logging.info(f"Loading configuration '{args.config_name}' from {args.config}")
    try:
        config = load_config(args.config, args.config_name)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return
    
    # Update config with local paths and run directory
    config.training.train_data_path = str(data_dir / "train")
    config.training.valid_data_path = str(data_dir / "valid")
    config.training.test_data_path = str(data_dir / "test")
    config.training.means_path = str(data_dir / "stats" / "global_means.npy")
    config.training.stds_path = str(data_dir / "stats" / "global_stds.npy")
    config.training.log_dir = str(run_dir)  # Use the run directory for logs
    
    # Apply model-specific configuration settings
    if accelerator == "cuda" and hasattr(torch.backends.cuda, 'matmul'):
        # Apply TF32 setting from config
        torch.backends.cuda.matmul.allow_tf32 = config.model.allow_tf32
        logging.info(f"Setting CUDA TF32 to: {config.model.allow_tf32}")
    
    # Create trainer
    trainer = WeatherTrainer(
        config=config,
        strategy=strategy,
        precision=config.model.precision,
        devices=device_count,
        num_nodes=1,
        logging_backend=LoggingBackend.TENSORBOARD,
        max_checkpoints=config.training.max_checkpoints,
        enable_flops_profiler=args.enable_flops_profiler
    )
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main() 