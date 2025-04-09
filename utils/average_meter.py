class AverageMeter:
    """Computes and stores the average and current value for tracking metrics.
    
    This class is useful for tracking running averages of metrics during training.
    """
    
    def __init__(self):
        """Initialize the AverageMeter."""
        self.reset()
        
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        """Update the meter with a new value.
        
        Args:
            val: The new value to include in the meter
            n: The weight of this new sample (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0 