import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from io import BytesIO

def create_comparison_figure(y_pred, y_true, channel_idx=0, max_samples=4, channel_name=None, vmin=None, vmax=None, error_max=None):
    """Create a figure comparing prediction, ground truth and error for wandb logging.
    
    Args:
        y_pred: Model prediction tensor [batch, channels, height, width]
        y_true: Ground truth tensor [batch, channels, height, width]
        channel_idx: Index of the channel to visualize
        max_samples: Maximum number of samples to include in the figure
        channel_name: Optional name of the channel for plot titles
        vmin: Optional minimum value for colorscale (if None, will be computed from data)
        vmax: Optional maximum value for colorscale (if None, will be computed from data)
        error_max: Optional maximum value for error colorscale (if None, will be computed from data)
        
    Returns:
        fig: Matplotlib figure object
    """
    # Ensure tensors are on CPU
    y_pred = y_pred.cpu() if hasattr(y_pred, 'cpu') else y_pred
    y_true = y_true.cpu() if hasattr(y_true, 'cpu') else y_true
    
    batch_size = min(y_pred.shape[0], max_samples)
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    
    # Handle case with single sample
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    # Use channel name if provided, otherwise use index
    channel_label = channel_name if channel_name else f"Channel {channel_idx}"
    
    # If vmin/vmax/error_max not provided, compute from all data for consistency
    if vmin is None or vmax is None:
        all_true = y_true[:batch_size, channel_idx].cpu().numpy()
        all_pred = y_pred[:batch_size, channel_idx].cpu().numpy()
        vmin = vmin if vmin is not None else min(np.min(all_true), np.min(all_pred))
        vmax = vmax if vmax is not None else max(np.max(all_true), np.max(all_pred))
    
    if error_max is None:
        all_errors = []
        for i in range(batch_size):
            pred = y_pred[i, channel_idx].cpu().numpy()
            true = y_true[i, channel_idx].cpu().numpy()
            all_errors.append(np.abs(pred - true))
        error_max = np.max(all_errors)
    
    for i in range(batch_size):
        # Get data for the current sample
        pred = y_pred[i, channel_idx].cpu().numpy()
        true = y_true[i, channel_idx].cpu().numpy()
        error = np.abs(pred - true)
        
        # Plot prediction
        im0 = axes[i, 0].imshow(pred, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'Prediction: {channel_label} (Sample {i+1})')
        fig.colorbar(im0, ax=axes[i, 0])
        
        # Plot ground truth
        im1 = axes[i, 1].imshow(true, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'Ground Truth: {channel_label} (Sample {i+1})')
        fig.colorbar(im1, ax=axes[i, 1])
        
        # Plot error
        im2 = axes[i, 2].imshow(error, cmap='hot', vmin=0, vmax=error_max)
        axes[i, 2].set_title(f'Error: {channel_label} (Sample {i+1})')
        fig.colorbar(im2, ax=axes[i, 2])
    
    plt.tight_layout()
    return fig

def fig_to_image(fig):
    """Convert matplotlib figure to image bytes for wandb logging."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_animation(pred_sequence, true_sequence, channel_idx=0, channel_name=None):
    """Create an animation showing temporal evolution of predictions vs ground truth.
    
    Args:
        pred_sequence: Model predictions tensor [sequence_length, channels, height, width]
        true_sequence: Ground truth tensor [sequence_length, channels, height, width]
        channel_idx: Index of the channel to visualize
        channel_name: Optional name of the channel for plot titles
        
    Returns:
        BytesIO object containing the animation in MP4 format
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Use channel name if provided, otherwise use index
    channel_label = channel_name if channel_name else f"Channel {channel_idx}"
    
    # Get data for the specified channel
    pred_data = pred_sequence[:, channel_idx].cpu().numpy()
    true_data = true_sequence[:, channel_idx].cpu().numpy()
    
    # Determine global min/max for consistent colormap
    vmin = min(pred_data.min(), true_data.min())
    vmax = max(pred_data.max(), true_data.max())
    
    # Initial frames
    im1 = ax1.imshow(pred_data[0], cmap='viridis', vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(true_data[0], cmap='viridis', vmin=vmin, vmax=vmax)
    
    # Add titles and colorbar
    ax1.set_title(f'Prediction: {channel_label}')
    ax2.set_title(f'Ground Truth: {channel_label}')
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    
    # Set tight layout
    plt.tight_layout()
    
    # Function to update frame
    def update(frame):
        im1.set_array(pred_data[frame])
        im2.set_array(true_data[frame])
        return im1, im2
    
    # Create animation
    seq_length = min(pred_data.shape[0], true_data.shape[0])
    anim = animation.FuncAnimation(fig, update, frames=seq_length, blit=True)
    
    # Save animation to buffer
    buf = BytesIO()
    anim.save(buf, writer='pillow', fps=2) # Use pillow for GIF output
    buf.seek(0)
    plt.close(fig)
    
    return buf

def create_spatial_error_map(y_pred, y_true, epoch=0, error_max=None):
    """Create a spatial error map visualization.
    
    Args:
        y_pred: Model prediction tensor [batch, channels, height, width]
        y_true: Ground truth tensor [batch, channels, height, width]
        epoch: Current epoch number for title
        error_max: Optional maximum value for error colorscale (if None, will be computed from data)
        
    Returns:
        fig: Matplotlib figure object
    """
    # Ensure tensors are on CPU before computing error
    y_pred_cpu = y_pred.cpu() if hasattr(y_pred, 'cpu') else y_pred
    y_true_cpu = y_true.cpu() if hasattr(y_true, 'cpu') else y_true
    
    error_map = torch.abs(y_pred_cpu - y_true_cpu).mean(dim=1)  # Mean across channels
    
    # Use provided error_max or compute from data
    if error_max is None:
        error_max = error_map.max().item()
    
    error_fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(error_map[0].numpy(), cmap='hot', vmin=0, vmax=error_max)
    ax.set_title(f'Mean Spatial Error (Epoch {epoch})')
    error_fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return error_fig

def create_lat_lon_error_plots(y_pred, y_true, epoch=0, error_max=None):
    """Create latitude and longitude error profile plots.
    
    Args:
        y_pred: Model prediction tensor [batch, channels, height, width]
        y_true: Ground truth tensor [batch, channels, height, width]
        epoch: Current epoch number for title
        error_max: Optional maximum value for y-axis (if None, will default to 1)
        
    Returns:
        fig: Matplotlib figure object
    """
    # Ensure tensors are on CPU
    y_pred = y_pred.cpu() if hasattr(y_pred, 'cpu') else y_pred
    y_true = y_true.cpu() if hasattr(y_true, 'cpu') else y_true
    
    # Calculate mean error along latitude and longitude
    lat_error = torch.abs(y_pred - y_true).mean(dim=(0, 1, 3))  # Mean across batch, channels, longitude
    lon_error = torch.abs(y_pred - y_true).mean(dim=(0, 1, 2))  # Mean across batch, channels, latitude
    
    # Use provided error_max or default to 1
    y_max = error_max if error_max is not None else 1
    
    # Create figure with latitude and longitude error profiles
    lat_lon_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Latitude error profile
    ax1.plot(lat_error.numpy())
    ax1.set_title('Error by Latitude')
    ax1.set_xlabel('Latitude Index')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_ylim(0, y_max)
    ax1.grid(True)
    
    # Longitude error profile
    ax2.plot(lon_error.numpy())
    ax2.set_title('Error by Longitude')
    ax2.set_xlabel('Longitude Index')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_ylim(0, y_max)
    ax2.grid(True)
    
    plt.tight_layout()
    return lat_lon_fig

def create_difference_map(y_pred, y_true, channel_idx=0, channel_name=None, epoch=0, diff_max=None):
    """Create a difference map (prediction - ground truth).
    
    Args:
        y_pred: Model prediction tensor [batch, channels, height, width]
        y_true: Ground truth tensor [batch, channels, height, width]
        channel_idx: Index of the channel to visualize
        channel_name: Optional name of the channel for plot title
        epoch: Current epoch number for title
        diff_max: Optional maximum absolute value for difference colorscale (if None, will be computed from data)
        
    Returns:
        fig: Matplotlib figure object
    """
    # Ensure tensors are on CPU
    y_pred = y_pred.cpu() if hasattr(y_pred, 'cpu') else y_pred
    y_true = y_true.cpu() if hasattr(y_true, 'cpu') else y_true
    
    # Use channel name if provided, otherwise use index
    channel_label = channel_name if channel_name else f"Channel {channel_idx}"
    
    diff_fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    diff = y_pred[0, channel_idx] - y_true[0, channel_idx]  # First batch, specified channel
    
    # Use provided diff_max or compute from data
    if diff_max is None:
        vmax = max(abs(diff.min().item()), abs(diff.max().item()))
    else:
        vmax = diff_max
    vmin = -vmax
    
    im = ax.imshow(diff.numpy(), cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title(f'Prediction - Ground Truth ({channel_label})')
    diff_fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return diff_fig

def create_synthetic_sequence(sample, seq_length):
    """Create a synthetic temporal sequence from a single sample.
    
    Args:
        sample: Single sample tensor [batch, channels, height, width]
        seq_length: Number of time steps to generate
        
    Returns:
        seq: Synthetic sequence tensor [seq_length, channels, height, width]
    """
    seq = torch.zeros(seq_length, sample.shape[1], sample.shape[2], sample.shape[3])
    
    # Create evolving pattern (simple evolving pattern for demo)
    for t in range(seq_length):
        # Scale factor increases over time (0.8 to 1.2)
        scale = 0.8 + 0.4 * (t / max(1, seq_length - 1))
        seq[t] = sample[0] * scale
    
    return seq

def create_spatial_rmse_map(y_pred, y_true, channel_idx=None, epoch=0, rmse_max=None):
    """Create a spatial RMSE error map visualization.
    
    Args:
        y_pred: Model prediction tensor [batch, channels, height, width]
        y_true: Ground truth tensor [batch, channels, height, width]
        channel_idx: Index of the channel to visualize. If None, compute RMSE across all channels.
        epoch: Current epoch number for title
        rmse_max: Optional maximum value for error colorscale (if None, will be computed from data)
        
    Returns:
        fig: Matplotlib figure object
    """
    # Ensure tensors are on CPU before computing error
    y_pred_cpu = y_pred.cpu() if hasattr(y_pred, 'cpu') else y_pred
    y_true_cpu = y_true.cpu() if hasattr(y_true, 'cpu') else y_true
    
    # If channel_idx is provided, compute RMSE for that channel only
    if channel_idx is not None:
        squared_error = (y_pred_cpu[:, channel_idx] - y_true_cpu[:, channel_idx])**2
        rmse_map = torch.sqrt(squared_error.mean(dim=0))  # Mean across batch
        channel_str = f" - Channel {channel_idx}"
    else:
        # Compute RMSE across all channels
        squared_error = (y_pred_cpu - y_true_cpu)**2
        rmse_map = torch.sqrt(squared_error.mean(dim=(0, 1)))  # Mean across batch and channels
        channel_str = ""
    
    # Use provided rmse_max or compute from data
    if rmse_max is None:
        rmse_max = rmse_map.max().item()
    
    # Create figure
    rmse_fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(rmse_map.numpy(), cmap='hot', vmin=0, vmax=rmse_max)
    ax.set_title(f'Spatial RMSE Error{channel_str} (Epoch {epoch})')
    rmse_fig.colorbar(im, ax=ax)
    ax.grid(True)
    plt.tight_layout()
    return rmse_fig 