from typing import Any, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import yaml
from enum import Enum

class PrecisionType(str, Enum):
    FP32 = "32-true"
    FP16 = "16-true"
    BF16 = "bf16-true"
    BF16MIXED = "bf16-mixed"
    FP16MIXED = "16-mixed"

class LoggingBackend(str, Enum):
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    BOTH = "both"
    NONE = "none"

@dataclass
class ModelConfig:
    # Required parameters (no defaults)
    input_channels: int
    output_channels: int
    embed_dim: int
    num_layers: int
    num_heads: int
    img_size: Tuple[int, int]
    
    # Optional parameters (with defaults)
    qkv_bias: bool = True
    dropout: float = 0.1
    attention_dropout: float = 0.1
    stochastic_depth: float = 0.1
    use_flash_attention: bool = True
    patch_size: int = 8
    precision: PrecisionType = PrecisionType.BF16
    use_checkpointing: bool = False
    allow_tf32: bool = False

    def __post_init__(self):
        """Validate and convert types after initialization."""
        # Convert numeric values to their proper types
        self.input_channels = int(self.input_channels)
        self.output_channels = int(self.output_channels)
        self.embed_dim = int(self.embed_dim)
        self.num_layers = int(self.num_layers)
        self.num_heads = int(self.num_heads)
        self.patch_size = int(self.patch_size)
        self.dropout = float(self.dropout)
        self.attention_dropout = float(self.attention_dropout)
        self.stochastic_depth = float(self.stochastic_depth)
        
        # Convert img_size to tuple of ints if it's a list
        if isinstance(self.img_size, list):
            self.img_size = tuple(int(x) for x in self.img_size)

@dataclass
class TrainingConfig:
    # Required parameters (no defaults)
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    data_dir: str
    checkpoint_dir: str
    log_dir: str
    train_data_path: str
    valid_data_path: str
    test_data_path: str
    means_path: str
    stds_path: str
    
    # Optional parameters (with defaults)
    num_workers: int = 4
    save_every: int = 1
    flops_budget: float = 1e18
    time_budget_hours: float = 24
    normalize: bool = True
    data_subset: float = 1.0
    temporal_coverage: str = "1y"
    log_frequency: int = 100
    save_frequency: int = 10
    max_checkpoints: int = 3
    use_wandb: bool = True
    wandb_project: str = "weather-transformer"
    logging_backend: LoggingBackend = LoggingBackend.TENSORBOARD
    sequence_length: int = 4
    temporal_resolution: str = "6h"
    dt: int = 1
    warmup_steps: int = 1000
    gradient_clipping: float = 1.0

    def __post_init__(self):
        """Validate and convert types after initialization."""
        # Convert numeric values to their proper types
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.num_workers = int(self.num_workers)
        self.save_every = int(self.save_every)
        self.flops_budget = float(self.flops_budget)
        self.time_budget_hours = float(self.time_budget_hours)
        self.data_subset = float(self.data_subset)
        self.log_frequency = int(self.log_frequency)
        self.save_frequency = int(self.save_frequency)
        self.max_checkpoints = int(self.max_checkpoints)
        self.sequence_length = int(self.sequence_length)
        self.dt = int(self.dt)
        self.warmup_steps = int(self.warmup_steps)
        self.gradient_clipping = float(self.gradient_clipping)

@dataclass
class ExperimentConfig:
    # Required parameters (no defaults)
    name: str
    description: str
    
    # Optional parameters (with defaults)
    num_gpus: int = 1
    num_nodes: int = 1
    memory_optimization_level: int = 1
    deepspeed_config: str = "config/ds_config.yaml"

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    experiment: ExperimentConfig

def validate_config(config_dict: Dict[str, Any]) -> None:
    """Validate the configuration dictionary."""
    required_sections = ['model', 'training', 'experiment']
    for section in required_sections:
        if section not in config_dict:
            raise ValueError(f"Missing required section '{section}' in configuration")
    
    # Validate model section
    model = config_dict['model']
    required_model_params = [
        'embed_dim', 'num_layers', 'num_heads', 'input_channels', 
        'output_channels', 'img_size'
    ]
    for param in required_model_params:
        if param not in model:
            raise ValueError(f"Missing required model parameter '{param}'")
    
    # Validate training section
    training = config_dict['training']
    required_training_params = [
        'batch_size', 'learning_rate', 'weight_decay', 'epochs',
        'train_data_path', 'valid_data_path', 'test_data_path'
    ]
    for param in required_training_params:
        if param not in training:
            raise ValueError(f"Missing required training parameter '{param}'")
    
    # Validate experiment section
    experiment = config_dict['experiment']
    required_experiment_params = ['name', 'description']
    for param in required_experiment_params:
        if param not in experiment:
            raise ValueError(f"Missing required experiment parameter '{param}'")

def load_config(config_path: Union[str, Path], config_name: Optional[str] = None) -> Config:
    """
    Load and validate configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        config_name: Name of the specific configuration to load (if None, returns the first one)
        
    Returns:
        Validated Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    if config_name is None:
        # If no specific config requested, return the first one found
        for name, config in all_configs.items():
            if isinstance(config, dict):
                config_dict = config
                break
        else:
            raise ValueError(f"No valid configuration found in {config_path}")
    else:
        if config_name not in all_configs:
            raise ValueError(f"Configuration '{config_name}' not found in {config_path}")
        config_dict = all_configs[config_name]
    
    # Handle YAML inheritance (merge key <<:)
    def resolve_inheritance(config: Dict[str, Any]) -> Dict[str, Any]:
        # Check for merge key
        if '<<' in config:
            base_configs = config.pop('<<')
            if not isinstance(base_configs, list):
                base_configs = [base_configs]
            
            # Merge base configurations
            merged = {}
            for base in base_configs:
                if isinstance(base, str) and base.startswith('*'):
                    base_name = base[1:]
                    if base_name in all_configs:
                        base_config = resolve_inheritance(all_configs[base_name])
                        if isinstance(base_config, dict):
                            merged.update(base_config)
            
            # Update with current config (local config overrides base config)
            if isinstance(config, dict):
                merged.update(config)
                return merged
        
        # Recursively resolve inheritance in nested dictionaries
        result = {}
        for k, v in config.items():
            if isinstance(v, dict):
                result[k] = resolve_inheritance(v)
            else:
                result[k] = v
        return result
    
    # Resolve inheritance
    config_dict = resolve_inheritance(config_dict)
    
    # Validate the configuration
    validate_config(config_dict)
    
    # Convert to strongly typed objects
    try:
        model_config = ModelConfig(**config_dict['model'])
        training_config = TrainingConfig(**config_dict['training'])
        experiment_config = ExperimentConfig(**config_dict['experiment'])
        
        return Config(
            model=model_config,
            training=training_config,
            experiment=experiment_config
        )
    except Exception as e:
        raise ValueError(f"Error converting configuration to typed objects: {str(e)}") 