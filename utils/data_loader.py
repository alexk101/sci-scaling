import logging
import glob
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Tuple, Dict, Any, Union, List
from pathlib import Path


def worker_init_fn(worker_id):
    """Initialize worker with unique seed."""
    np.random.seed(torch.utils.data.get_worker_info().seed % (2**32 - 1))


def get_data_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    distributed: bool = False,
    train: bool = True,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    data_parallel_shards: Optional[int] = None,
    data_parallel_shard_id: Optional[int] = None,
    drop_last: bool = True,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """
    Create a DataLoader with appropriate sampler for distributed training.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size for each process
        num_workers: Number of workers for data loading
        distributed: Whether to use distributed training
        train: Whether this is for training (affects shuffling)
        world_size: Total number of processes (for custom distribution)
        rank: Current process rank (for custom distribution)
        data_parallel_shards: Number of data-parallel shards for model parallelism
        data_parallel_shard_id: Current data-parallel shard ID
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        Tuple of (dataloader, sampler). Sampler is returned to allow for epoch-based reshuffling.
    """
    sampler = None
    
    if distributed:
        # For model parallelism scenarios
        if data_parallel_shards is not None and data_parallel_shard_id is not None:
            sampler = DistributedSampler(
                dataset,
                num_replicas=data_parallel_shards,
                rank=data_parallel_shard_id,
                shuffle=train
            )
        # For standard distributed training
        else:
            # Custom world_size and rank can be provided
            kwargs = {}
            if world_size is not None:
                kwargs['num_replicas'] = world_size
            if rank is not None:
                kwargs['rank'] = rank
                
            sampler = DistributedSampler(
                dataset,
                shuffle=train,
                **kwargs
            )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(sampler is None and train),  # Only shuffle if no sampler and in training mode
        sampler=sampler,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader, sampler


class WeatherDataset(Dataset):
    """Dataset for weather data.

    Args:
        data_path: Path to data directory
        dt: Time difference between input and target
        img_size: Tuple of (height, width) of input image
        input_channels: Number of input channels
        output_channels: Number of output channels
        normalize: Whether to normalize the data
        means_path: Path to pre-computed means (required if normalize=True)
        stds_path: Path to pre-computed standard deviations (required if normalize=True)
        limit_samples: Optional limit on number of samples
        train: Whether this is training data
    """

    def __init__(
        self,
        data_path: str,
        dt: int = 1,
        img_size: Tuple[int, int] = (360, 720),
        input_channels: int = 20,
        output_channels: int = 20,
        normalize: bool = True,
        means_path: Optional[str] = None,
        stds_path: Optional[str] = None,
        limit_samples: Optional[int] = None,
        train: bool = True,
    ):
        self.data_path = Path(data_path)
        self.dt = dt
        self.img_size = img_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.normalize = normalize
        self.train = train

        # Load normalization stats
        if normalize:
            # Check that paths are provided when normalization is enabled
            if not means_path or not stds_path:
                raise ValueError(
                    "When normalize=True, both means_path and stds_path must be provided. "
                    "These should be specified in the configuration."
                )
            try:
                self.means = np.load(means_path)[0][:input_channels]
                self.stds = np.load(stds_path)[0][:input_channels]
                logging.info(f"Loaded normalization stats from {means_path} and {stds_path}")
            except Exception as e:
                logging.error(f"Error loading normalization stats: {e}")
                raise ValueError(
                    f"Failed to load normalization stats from {means_path} and {stds_path}: {e}"
                )
        else:
            self.means = None
            self.stds = None

        # Get list of data files
        try:
            self.files = sorted(glob.glob(str(self.data_path / "*.h5")))
            if not self.files:
                raise ValueError(f"No HDF5 files found in {data_path}")
        except Exception as e:
            logging.error(f"Error finding data files: {e}")
            raise

        # Get dataset statistics
        try:
            with h5py.File(self.files[0], "r") as f:
                self.samples_per_file = f["fields"].shape[0]
                self.total_samples = len(self.files) * self.samples_per_file
                
                # Check if image size is smaller than data dimensions (cropping)
                full_height, full_width = f["fields"].shape[2:4]
                requested_height, requested_width = self.img_size
                
                if requested_height < full_height or requested_width < full_width:
                    logging.info(f"Cropping data: requested size {requested_height}x{requested_width} is smaller than "
                                f"actual data dimensions {full_height}x{full_width}")
        except Exception as e:
            logging.error(f"Error reading dataset statistics: {e}")
            raise

        # Apply sample limit if specified
        if limit_samples is not None:
            self.total_samples = min(self.total_samples, limit_samples)

        # Initialize file handles and datasets
        self.file_handles = [None] * len(self.files)
        self.datasets = [None] * len(self.files)
        
        # Preload the first file
        if len(self.files) > 0:
            self._load_file(0)

        loader_type = "training" if self.train else "validation"
        logging.info(f"Initialized {loader_type} dataset with {self.total_samples} samples")

    def _normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize the data using pre-computed means and standard deviations."""
        if self.normalize and self.means is not None and self.stds is not None:
            data = data - torch.from_numpy(self.means).view(-1, 1, 1)
            data = data / torch.from_numpy(self.stds).view(-1, 1, 1)
        return data

    def _get_file_and_index(self, idx: int) -> Tuple[int, int]:
        """Convert global index to file index and local index."""
        file_idx = idx // self.samples_per_file
        local_idx = idx % self.samples_per_file
        return file_idx, local_idx

    def _load_file(self, file_idx: int):
        """Load a file and its dataset."""
        try:
            if self.file_handles[file_idx] is None:
                self.file_handles[file_idx] = h5py.File(self.files[file_idx], "r")
                self.datasets[file_idx] = self.file_handles[file_idx]["fields"]
        except Exception as e:
            logging.error(f"Error loading file {self.files[file_idx]}: {e}")
            raise

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data sample.

        Args:
            idx: Index of the sample to get

        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        file_idx, local_idx = self._get_file_and_index(idx)

        # Ensure we don't exceed file boundaries
        local_idx = int(local_idx % (self.samples_per_file - self.dt))
        if local_idx < self.dt:
            local_idx += self.dt

        # Make sure the file is loaded before accessing 
        if self.datasets[file_idx] is None:
            self._load_file(file_idx)

        # Load input and target data
        try:
            # Cast slicing indices to integers to avoid type issues
            in_ch = int(self.input_channels)
            out_ch = int(self.output_channels)
            h = int(self.img_size[0])
            w = int(self.img_size[1])
            
            input_data = self.datasets[file_idx][
                local_idx, :in_ch, :h, :w
            ]
            target_data = self.datasets[file_idx][
                local_idx + self.dt,
                :out_ch,
                :h,
                :w,
            ]
        except Exception as e:
            logging.error(f"Error reading data at index {idx}: {e}")
            logging.error(f"Dataset shape: {self.datasets[file_idx].shape}")
            logging.error(f"Attempted indices: local_idx={local_idx}, dt={self.dt}, channels={in_ch}, h={h}, w={w}")
            raise

        # Convert to torch tensors and normalize
        input_tensor = torch.from_numpy(input_data).float()
        target_tensor = torch.from_numpy(target_data).float()

        input_tensor = self._normalize(input_tensor)
        target_tensor = self._normalize(target_tensor)

        return input_tensor, target_tensor

    def __del__(self):
        """Clean up resources when the dataset is deleted."""
        try:
            # Close all open file handles
            for handle in self.file_handles:
                if handle is not None:
                    handle.close()
        except Exception as e:
            logging.error(f"Error in dataset cleanup: {e}")
