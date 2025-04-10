import numpy as np
import torch
import torch.nn as nn
import time
import psutil
from typing import Dict, List, Optional
from sklearn.metrics import r2_score, explained_variance_score
from dataclasses import dataclass
from collections import defaultdict
import logging


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""

    gpu_utilization: float
    gpu_memory_used: float
    gpu_memory_total: float
    cpu_percent: float
    memory_percent: float
    disk_io_read: float
    disk_io_write: float
    network_io_sent: float
    network_io_recv: float


@dataclass
class PerformanceMetrics:
    """Performance metrics."""

    forward_time: float
    backward_time: float
    optimizer_time: float
    total_time: float
    samples_per_second: float
    flops: int
    macs: int
    params: int


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    mse: float
    mae: float
    rmse: float
    r2: float
    explained_variance: float
    loss: float
    attention_maps: Optional[List[torch.Tensor]] = None
    gradients: Optional[Dict[str, torch.Tensor]] = None


class MetricsTracker:
    """Tracks and computes various metrics during training."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.resource_metrics = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        self.model_metrics = defaultdict(list)
        self.start_time = None
        self.last_disk_io = psutil.disk_io_counters()
        self.last_net_io = psutil.net_io_counters()

        # Detect GPU type for hardware-specific metrics
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if any(gpu in device_name.lower() for gpu in ['amd', 'radeon', 'instinct']):
                logging.info(f"Detected AMD GPU: {device_name}")
                self.gpu_type = "amd"
            else:
                logging.info(f"Detected NVIDIA GPU: {device_name}")
                self.gpu_type = "nvidia"
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logging.info("Detected Apple MPS device")
            self.gpu_type = "mps"
        else:
            logging.info("No GPU detected, using CPU metrics only")
            self.gpu_type = "cpu"

        # Initialize specialized metrics tracking based on GPU type
        if self.gpu_type == "amd":
            try:
                from utils.rocm_profiler import RocmProfiler
                self.rocm_profiler = RocmProfiler()
                logging.info("Initialized ROCm metrics tracking")
            except ImportError as e:
                self.rocm_profiler = None
                logging.warning(f"Warning: ROCm profiler not available. {e}")
        
        elif self.gpu_type == "nvidia":
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                logging.info("Initialized NVIDIA NVML metrics tracking")
            except (ImportError, AttributeError):
                self.nvml_handle = None
                logging.warning("Warning: NVML not available. Install pynvml for advanced NVIDIA GPU metrics.")
        
        elif self.gpu_type == "mps":
            try:
                import platform
                self.apple_platform = platform.platform()
                logging.info("MPS metrics tracking limited on Apple Silicon")
            except Exception as e:
                self.apple_platform = None
                logging.warning(f"Warning: Failed to initialize MPS metrics: {e}")

    def reset(self):
        """Reset all metric collections. Used between validation epochs."""
        self.resource_metrics = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        self.model_metrics = defaultdict(list)
        self.start_time = None

    def compute(self) -> Dict[str, float]:
        """Compute and return all metrics. Used for validation reporting."""
        # Get all metrics as dictionary
        metrics = self.get_average_metrics()
        
        # Convert from resource/model prefixed format to flat dictionary
        result = {}
        for key, value in metrics.items():
            # Extract the actual metric name by removing prefixes
            if '/' in key:
                simple_key = key.split('/')[-1]
                result[simple_key] = value
            else:
                result[key] = value
                
        return result

    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics using platform-specific tools."""
        metrics = {
            "gpu_utilization": 0.0,
            "gpu_memory_used": 0.0,
            "gpu_memory_total": 0.0,
            "gpu_temperature": 0.0,
            "gpu_power_usage": 0.0,
        }

        # Common metrics for any GPU platform using PyTorch
        if torch.cuda.is_available():
            # These work for both NVIDIA and AMD GPUs via PyTorch
            metrics["gpu_memory_used"] = torch.cuda.memory_allocated() / 1024**3  # GB
            metrics["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        # Platform-specific metrics
        if self.gpu_type == "nvidia" and hasattr(self, 'nvml_handle') and self.nvml_handle:
            try:
                import pynvml
                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                metrics["gpu_utilization"] = util.gpu
                
                # Get temperature
                temp = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics["gpu_temperature"] = temp
                
                # Get power usage
                power = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0  # Convert from mW to W
                metrics["gpu_power_usage"] = power
            except Exception as e:
                logging.warning(f"Warning: Failed to get NVIDIA metrics: {e}")
        
        elif self.gpu_type == "amd" and hasattr(self, 'rocm_profiler') and self.rocm_profiler:
            metrics["gpu_utilization"] = self.rocm_profiler.getUtilization(0)
            metrics["gpu_temperature"] = self.rocm_profiler.getTemp(0)
            metrics["gpu_power_usage"] = self.rocm_profiler.getPower(0)
        
        # For MPS (Apple Silicon), we unfortunately don't have access to detailed metrics
        # through Python APIs yet

        return metrics

    def start_batch(self):
        """Start timing a batch."""
        self.start_time = time.time()

    def end_batch(self, batch_size: int):
        """End timing a batch and compute metrics."""
        if self.start_time is None:
            return

        total_time = time.time() - self.start_time
        self.performance_metrics["total_time"].append(total_time)
        self.performance_metrics["samples_per_second"].append(batch_size / total_time)

    def record_resource_metrics(self):
        """Record current resource utilization."""
        # GPU metrics
        gpu_metrics = self._get_gpu_metrics()
        for key, value in gpu_metrics.items():
            self.resource_metrics[key].append(value)

        # CPU and memory metrics
        self.resource_metrics["cpu_percent"].append(psutil.cpu_percent())
        self.resource_metrics["memory_percent"].append(psutil.virtual_memory().percent)

        # Disk I/O metrics
        current_disk_io = psutil.disk_io_counters()
        if self.last_disk_io and current_disk_io:
            self.resource_metrics["disk_io_read"].append(
                (current_disk_io.read_bytes - self.last_disk_io.read_bytes)
                / 1024
                / 1024  # MB
            )
            self.resource_metrics["disk_io_write"].append(
                (current_disk_io.write_bytes - self.last_disk_io.write_bytes)
                / 1024
                / 1024  # MB
            )
            self.last_disk_io = current_disk_io

        # Network I/O metrics
        current_net_io = psutil.net_io_counters()
        if self.last_net_io and current_net_io:
            self.resource_metrics["network_io_sent"].append(
                (current_net_io.bytes_sent - self.last_net_io.bytes_sent)
                / 1024
                / 1024  # MB
            )
            self.resource_metrics["network_io_recv"].append(
                (current_net_io.bytes_recv - self.last_net_io.bytes_recv)
                / 1024
                / 1024  # MB
            )
            self.last_net_io = current_net_io

    def record_performance_metrics(
        self, forward_time: float, backward_time: float, optimizer_time: float
    ):
        """Record performance timing metrics."""
        self.performance_metrics["forward_time"].append(forward_time)
        self.performance_metrics["backward_time"].append(backward_time)
        self.performance_metrics["optimizer_time"].append(optimizer_time)

    def record_model_metrics(
        self, output: torch.Tensor, target: torch.Tensor, loss: float
    ):
        """Record model performance metrics."""
        with torch.no_grad():
            # Convert to numpy for sklearn metrics
            output_np = output.cpu().numpy()
            target_np = target.cpu().numpy()

            # Compute metrics
            mse = nn.MSELoss()(output, target).item()
            mae = nn.L1Loss()(output, target).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()

            # Reshape for sklearn metrics
            output_flat = output_np.reshape(-1, output_np.shape[-1])
            target_flat = target_np.reshape(-1, target_np.shape[-1])

            r2 = r2_score(target_flat, output_flat)
            explained_variance = explained_variance_score(target_flat, output_flat)

            # Record metrics
            self.model_metrics["mse"].append(mse)
            self.model_metrics["mae"].append(mae)
            self.model_metrics["rmse"].append(rmse)
            self.model_metrics["r2"].append(r2)
            self.model_metrics["explained_variance"].append(explained_variance)
            self.model_metrics["loss"].append(loss)

    def record_attention_maps(self, attention_maps: List[torch.Tensor]):
        """Record attention maps for analysis."""
        self.model_metrics["attention_maps"].append(attention_maps)

    def record_gradients(self, gradients: Dict[str, torch.Tensor]):
        """Record gradient statistics."""
        self.model_metrics["gradients"] = gradients

    def get_average_metrics(self) -> Dict[str, float]:
        """Get average of all recorded metrics."""
        metrics = {}

        # Resource metrics
        for key, values in self.resource_metrics.items():
            if values:
                metrics[f"resource/{key}"] = np.mean(values)

        # Performance metrics
        for key, values in self.performance_metrics.items():
            if values:
                metrics[f"performance/{key}"] = np.mean(values)

        # Model metrics
        for key, values in self.model_metrics.items():
            if key not in ["attention_maps", "gradients"] and values:
                metrics[f"model/{key}"] = np.mean(values)

        # Add platform info as a string - but skip these for numeric logging contexts
        # They can be accessed as properties but shouldn't be logged as metrics
        metrics["system/platform_name"] = self.gpu_type

        # Filter out non-numeric values to prevent logging errors
        filtered_metrics = {}
        for key, value in metrics.items():
            try:
                # Skip string values and convert to float to ensure numeric
                if isinstance(value, (int, float, bool, np.number)):
                    filtered_metrics[key] = float(value)
                # Skip strings, None values, etc.
            except (ValueError, TypeError):
                # If conversion fails, skip this metric
                pass

        return filtered_metrics

    def get_attention_analysis(self) -> Dict[str, float]:
        """Analyze attention patterns."""
        if not self.model_metrics["attention_maps"]:
            return {}

        attention_maps = self.model_metrics["attention_maps"][
            -1
        ]  # Use last recorded maps
        analysis = {}

        for i, attn_map in enumerate(attention_maps):
            # Compute attention statistics
            attn_np = attn_map.cpu().numpy()
            analysis[f"attention_{i}/mean"] = np.mean(attn_np)
            analysis[f"attention_{i}/std"] = np.std(attn_np)
            analysis[f"attention_{i}/max"] = np.max(attn_np)
            analysis[f"attention_{i}/min"] = np.min(attn_np)

            # Compute sparsity
            analysis[f"attention_{i}/sparsity"] = np.mean(attn_np < 0.01)

        return analysis

    def get_gradient_analysis(self) -> Dict[str, float]:
        """Analyze gradient statistics."""
        if not self.model_metrics["gradients"]:
            return {}

        gradients = self.model_metrics["gradients"]
        analysis = {}

        for name, grad in gradients.items():
            if grad is not None:
                grad_np = grad.cpu().numpy()
                analysis[f"gradient_{name}/mean"] = np.mean(grad_np)
                analysis[f"gradient_{name}/std"] = np.std(grad_np)
                analysis[f"gradient_{name}/max"] = np.max(grad_np)
                analysis[f"gradient_{name}/min"] = np.min(grad_np)
                analysis[f"gradient_{name}/norm"] = np.linalg.norm(grad_np)

        return analysis

    def update(self, output: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a new batch of predictions and targets.
        This is the main method called during validation to update metrics.
        
        Args:
            output: Model predictions tensor [batch, channels, height, width]
            target: Ground truth tensor [batch, channels, height, width]
        """
        with torch.no_grad():
            # Use the existing compute_metrics function for consistency
            metrics = compute_metrics(output, target)
            
            # Record metrics
            for key, value in metrics.items():
                self.model_metrics[key].append(value)
            
            # Record resource metrics if needed
            self.record_resource_metrics()


def compute_metrics(output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute basic metrics for a batch."""
    with torch.no_grad():
        mse = nn.MSELoss()(output, target).item()
        mae = nn.L1Loss()(output, target).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()

        # Convert to numpy for sklearn metrics
        output_np = output.cpu().numpy()
        target_np = target.cpu().numpy()

        # Reshape for sklearn metrics
        output_flat = output_np.reshape(-1, output_np.shape[-1])
        target_flat = target_np.reshape(-1, target_np.shape[-1])

        r2 = r2_score(target_flat, output_flat)
        explained_variance = explained_variance_score(target_flat, output_flat)

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "explained_variance": explained_variance,
        }


# Weather-specific metrics
@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    """Convert latitude index to degrees."""
    return 90.0 - j * 180.0 / float(num_lat - 1)


@torch.jit.script
def latitude_weighting_factor(
    j: torch.Tensor, num_lat: int, s: torch.Tensor
) -> torch.Tensor:
    """Compute latitude weighting factor."""
    return num_lat * torch.cos(3.1416 / 180.0 * lat(j, num_lat)) / s


@torch.jit.script
def weighted_rmse_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute latitude-weighted RMSE for each channel."""
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416 / 180.0 * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target) ** 2.0, dim=(-1, -2)))
    return result


@torch.jit.script
def weighted_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute average latitude-weighted RMSE across channels."""
    result = weighted_rmse_channels(pred, target)
    return torch.mean(result, dim=0)


@torch.jit.script
def weighted_acc_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute latitude-weighted anomaly correlation coefficient for each channel."""
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416 / 180.0 * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1, -2)) / torch.sqrt(
        torch.sum(weight * pred * pred, dim=(-1, -2))
        * torch.sum(weight * target * target, dim=(-1, -2))
    )
    return result


@torch.jit.script
def weighted_acc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute average latitude-weighted anomaly correlation coefficient across channels."""
    result = weighted_acc_channels(pred, target)
    return torch.mean(result, dim=0)


def time_communication(comm, device):
    """Time communication operations."""
    # Create dummy tensors for testing
    dummy_tensor = torch.ones(1000000, device=device)  # 4MB tensor

    # Time all-reduce (used in gradient synchronization)
    torch.cuda.synchronize()
    start = time.time()
    if comm.is_initialized("dp"):
        torch.distributed.all_reduce(dummy_tensor, group=comm.get_group("dp"))
    torch.cuda.synchronize()
    all_reduce_time = time.time() - start

    # Time broadcast (used in parameter synchronization)
    torch.cuda.synchronize()
    start = time.time()
    if comm.is_initialized("tp"):
        try:
            torch.distributed.broadcast(dummy_tensor, 0, group=comm.get_group("tp"))
        except ValueError:
            pass
    torch.cuda.synchronize()
    broadcast_time = time.time() - start

    return {
        "all_reduce_time_ms": all_reduce_time * 1000,
        "broadcast_time_ms": broadcast_time * 1000,
    }


def backward_with_comm_timing(loss, optimizer):
    """Time backward pass and communication operations."""
    # Start timing
    torch.cuda.synchronize()
    backward_start = time.time()

    # Do backward pass
    loss.backward()

    # Synchronize before communication
    torch.cuda.synchronize()
    compute_time = time.time() - backward_start

    # Time the all-reduce operations (if using DDP)
    comm_start = time.time()
    optimizer.step()
    torch.cuda.synchronize()
    comm_time = time.time() - comm_start

    return {
        "backward_compute_time": compute_time,
        "comm_time": comm_time,
        "comm_ratio": comm_time / (compute_time + comm_time),
    }
