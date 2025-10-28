import numpy as np
import torch


__all__ = ['MultiMarginalPairedSampler']


class MultiMarginalPairedSampler:
    """Sampler for naturally paired multi-marginal data.
    Shuffles the batch while maintaining the pairing across marginals.
    """
    
    def __init__(self, device='cpu', shuffle=True):
        self.device = device
        self.shuffle = shuffle
    
    def sample_plan(self, z, replace=False):
        """Sample from naturally paired marginals by shuffling the batch.
        
        Args:
            z: Tensor of shape (M, bs, d).
            replace: Ignored. Shuffling (without replacement) is used.
        
        Returns:
            zhat: Tensor of shape (M, bs, d) with shuffled paired samples.
        """
        M, bs, d = z.shape
        
        if self.shuffle:
            # Shuffle indices without replacement using torch.randperm
            # Ensure indices are on the same device as the input tensor
            indices = torch.randperm(bs, device=z.device)
            
            # Index all marginals with the same indices to maintain pairing
            zhat = z[:, indices, :]
        else:
            zhat = z
        
        return zhat