import numpy as np
import h5py
import os
from pathlib import Path
from typing import Tuple, Union
import logging

def generate_synthetic_weather_file(
    output_path: str,
    num_samples: int = 50,
    channels: int = 20,
    height: int = 64,
    width: int = 128,
) -> str:
    """Generate a synthetic weather data file for a single year.
    
    Args:
        output_path: Path to save the HDF5 file
        num_samples: Number of samples (time steps) to generate
        channels: Number of weather variables
        height: Height of the spatial grid
        width: Width of the spatial grid
        
    Returns:
        Path to the created file
    """
    # Create a base pattern with some spatial and temporal variation
    x = np.linspace(0, 2*np.pi, width)
    y = np.linspace(0, 2*np.pi, height)
    
    X, Y = np.meshgrid(x, y)
    
    # Generate temporal data with a flat time series format [time, channels, height, width]
    all_data = np.zeros((num_samples, channels, height, width), dtype=np.float32)
    
    for time_idx in range(num_samples):
        for channel_idx in range(channels):
            # Create a unique pattern for each channel
            phase_c = 2 * np.pi * channel_idx / channels
            phase_t = 2 * np.pi * time_idx / num_samples
            
            # Generate pattern with spatial and temporal variation
            pattern = np.sin(X + phase_c) * np.cos(Y + phase_c) * np.sin(phase_t + phase_c)
            
            # Add some random noise
            noise = np.random.normal(0, 0.1, (height, width))
            
            # Store in the data array
            all_data[time_idx, channel_idx] = pattern + noise
    
    # Save data to HDF5 file
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('fields', data=all_data)
    
    return output_path

def generate_synthetic_weather_data(
    output_dir: Union[str, Path],
    num_years: int = 3,
    samples_per_year: int = 50,
    channels: int = 20,
    height: int = 64,
    width: int = 128,
) -> Tuple[Path, Path]:
    """Generate synthetic weather data for testing, organized in folders like real data.
    
    Args:
        output_dir: Base directory to save the data
        num_years: Number of years (files) to generate per split
        samples_per_year: Number of samples per year
        channels: Number of weather variables
        height: Height of the spatial grid
        width: Width of the spatial grid
        
    Returns:
        Tuple of (train_dir, valid_dir) as Path objects
    """
    # Create output directories using pathlib Path
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    valid_dir = output_path / "valid"
    stats_dir = output_path / "stats"
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Log the data generation process
    logging.info(f"Generating synthetic data with dimensions: [time={samples_per_year}, channels={channels}, height={height}, width={width}]")
    
    # Generate training data files
    train_files = []
    for year in range(2010, 2010 + num_years):
        output_path = train_dir / f"weather_{year}.h5"
        generate_synthetic_weather_file(
            output_path=str(output_path),  # h5py expects string paths
            num_samples=samples_per_year,
            channels=channels,
            height=height,
            width=width,
        )
        train_files.append(output_path)
        # Log the creation of files
        logging.info(f"Created training file: {output_path}")
    
    # Generate validation data files
    valid_files = []
    for year in range(2015, 2015 + num_years):
        output_path = valid_dir / f"weather_{year}.h5"
        generate_synthetic_weather_file(
            output_path=str(output_path),  # h5py expects string paths
            num_samples=samples_per_year,
            channels=channels,
            height=height,
            width=width,
        )
        valid_files.append(output_path)
        # Log the creation of files
        logging.info(f"Created validation file: {output_path}")
    
    # Compute statistics from training data (just use the first file for simplicity)
    with h5py.File(str(train_files[0]), 'r') as f:
        train_data = f['fields'][:]
    
    means = np.mean(train_data, axis=(0, 2, 3))
    stds = np.std(train_data, axis=(0, 2, 3))
    
    # Save statistics
    np.save(stats_dir / "global_means.npy", means[np.newaxis, :])
    np.save(stats_dir / "global_stds.npy", stds[np.newaxis, :])
    
    # Log the directory structure
    logging.info(f"Created synthetic data directory structure:")
    logging.info(f"Training data: {train_dir}")
    logging.info(f"Validation data: {valid_dir}")
    logging.info(f"Stats: {output_path / 'stats'}")
    
    return train_dir, valid_dir

def create_test_data(height=64, width=128) -> Path:
    """Create a small test dataset with proper directory structure.
    
    Args:
        height: Height of the spatial grid (default: 64)
        width: Width of the spatial grid (default: 128)
        
    Returns:
        The path to the data directory as a Path object
    """
    data_dir = Path("data")
    train_dir, valid_dir = generate_synthetic_weather_data(
        output_dir=data_dir,
        num_years=3,  # Generate 3 files per split
        samples_per_year=50,  # 50 time steps per year
        channels=20,
        height=height,
        width=width,
    )
    
    logging.info(f"Created synthetic data directory structure:")
    logging.info(f"Training data: {train_dir}")
    logging.info(f"Validation data: {valid_dir}")
    logging.info(f"Stats: {data_dir / 'stats'}")
    
    return data_dir 