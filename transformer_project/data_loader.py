import torch
import tables
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional

pmt_positions = [
     {'r': 328.00, 'x': 0.00, 'y': -328.00, 'z': 542.85},
     {'r': 246.00, 'x': 0.00, 'y': -246.00, 'z': 542.85},
     {'r': 164.00, 'x': 0.00, 'y': -164.00, 'z': 542.85},
     {'r': 82.00, 'x': 0.00, 'y': -82.00, 'z': 542.85},
     {'r': 0.00, 'x': 0.00, 'y': 0.00, 'z': 542.85},
     {'r': 82.00, 'x': 0.00, 'y': 82.00, 'z': 542.85},
     {'r': 164.00, 'x': 0.00, 'y': 164.00, 'z': 542.85},
     {'r': 246.00, 'x': 0.00, 'y': 246.00, 'z': 542.85},
     {'r': 328.00, 'x': 0.00, 'y': 328.00, 'z': 542.85},
     {'r': 295.65, 'x': 71.01, 'y': -287.00, 'z': 542.85},
     {'r': 216.95, 'x': 71.01, 'y': -205.00, 'z': 542.85},
     {'r': 142.03, 'x': 71.01, 'y': -123.00, 'z': 542.85},
     {'r': 82.00, 'x': 71.01, 'y': -41.00, 'z': 542.85},
     {'r': 82.00, 'x': 71.01, 'y': 41.00, 'z': 542.85},
     {'r': 142.03, 'x': 71.01, 'y': 123.00, 'z': 542.85},
     {'r': 216.95, 'x': 71.01, 'y': 205.00, 'z': 542.85},
     {'r': 295.65, 'x': 71.01, 'y': 287.00, 'z': 542.85},
     {'r': 284.06, 'x': 142.03, 'y': -246.00, 'z': 542.85},
     {'r': 216.95, 'x': 142.03, 'y': -164.00, 'z': 542.85},
     {'r': 164.00, 'x': 142.03, 'y': -82.00, 'z': 542.85},
     {'r': 142.03, 'x': 142.03, 'y': 0.00, 'z': 542.85},
     {'r': 164.00, 'x': 142.03, 'y': 82.00, 'z': 542.85},
     {'r': 216.95, 'x': 142.03, 'y': 164.00, 'z': 542.85},
     {'r': 284.06, 'x': 142.03, 'y': 246.00, 'z': 542.85},
     {'r': 295.65, 'x': 213.04, 'y': -205.00, 'z': 542.85},
     {'r': 246.00, 'x': 213.04, 'y': -123.00, 'z': 542.85},
     {'r': 216.95, 'x': 213.04, 'y': -41.00, 'z': 542.85},
     {'r': 216.95, 'x': 213.04, 'y': 41.00, 'z': 542.85},
     {'r': 246.00, 'x': 213.04, 'y': 123.00, 'z': 542.85},
     {'r': 295.65, 'x': 213.04, 'y': 205.00, 'z': 542.85},
     {'r': 328.00, 'x': 284.06, 'y': -164.00, 'z': 542.85},
     {'r': 295.65, 'x': 284.06, 'y': -82.00, 'z': 542.85},
     {'r': 284.06, 'x': 284.06, 'y': 0.00, 'z': 542.85},
     {'r': 295.65, 'x': 284.06, 'y': 82.00, 'z': 542.85},
     {'r': 328.00, 'x': 284.06, 'y': 164.00, 'z': 542.85},
     {'r': 295.65, 'x': -71.01, 'y': -287.00, 'z': 542.85},
     {'r': 216.95, 'x': -71.01, 'y': -205.00, 'z': 542.85},
     {'r': 142.03, 'x': -71.01, 'y': -123.00, 'z': 542.85},
     {'r': 82.00, 'x': -71.01, 'y': -41.00, 'z': 542.85},
     {'r': 82.00, 'x': -71.01, 'y': 41.00, 'z': 542.85},
     {'r': 142.03, 'x': -71.01, 'y': 123.00, 'z': 542.85},
     {'r': 216.95, 'x': -71.01, 'y': 205.00, 'z': 542.85},
     {'r': 295.65, 'x': -71.01, 'y': 287.00, 'z': 542.85},
     {'r': 284.06, 'x': -142.03, 'y': -246.00, 'z': 542.85},
     {'r': 216.95, 'x': -142.03, 'y': -164.00, 'z': 542.85},
     {'r': 164.00, 'x': -142.03, 'y': -82.00, 'z': 542.85},
     {'r': 142.03, 'x': -142.03, 'y': 0.00, 'z': 542.85},
     {'r': 164.00, 'x': -142.03, 'y': 82.00, 'z': 542.85},
     {'r': 216.95, 'x': -142.03, 'y': 164.00, 'z': 542.85},
     {'r': 284.06, 'x': -142.03, 'y': 246.00, 'z': 542.85},
     {'r': 295.65, 'x': -213.04, 'y': -205.00, 'z': 542.85},
     {'r': 246.00, 'x': -213.04, 'y': -123.00, 'z': 542.85},
     {'r': 216.95, 'x': -213.04, 'y': -41.00, 'z': 542.85},
     {'r': 216.95, 'x': -213.04, 'y': 41.00, 'z': 542.85},
     {'r': 246.00, 'x': -213.04, 'y': 123.00, 'z': 542.85},
     {'r': 295.65, 'x': -213.04, 'y': 205.00, 'z': 542.85},
     {'r': 328.00, 'x': -284.06, 'y': -164.00, 'z': 542.85},
     {'r': 295.65, 'x': -284.06, 'y': -82.00, 'z': 542.85},
     {'r': 284.06, 'x': -284.06, 'y': 0.00, 'z': 542.85},
     {'r': 295.65, 'x': -284.06, 'y': 82.00, 'z': 542.85},
     {'r': 328.00, 'x': -284.06, 'y': 164.00, 'z': 542.85},
     {'r': 328.00, 'x': 0.00, 'y': -328.00, 'z': -542.85},
     {'r': 246.00, 'x': 0.00, 'y': -246.00, 'z': -542.85},
     {'r': 164.00, 'x': 0.00, 'y': -164.00, 'z': -542.85},
     {'r': 82.00, 'x': 0.00, 'y': -82.00, 'z': -542.85},
     {'r': 0.00, 'x': 0.00, 'y': 0.00, 'z': -542.85},
     {'r': 82.00, 'x': 0.00, 'y': 82.00, 'z': -542.85},
     {'r': 164.00, 'x': 0.00, 'y': 164.00, 'z': -542.85},
     {'r': 246.00, 'x': 0.00, 'y': 246.00, 'z': -542.85},
     {'r': 328.00, 'x': 0.00, 'y': 328.00, 'z': -542.85},
     {'r': 295.65, 'x': 71.01, 'y': -287.00, 'z': -542.85},
     {'r': 216.95, 'x': 71.01, 'y': -205.00, 'z': -542.85},
     {'r': 142.03, 'x': 71.01, 'y': -123.00, 'z': -542.85},
     {'r': 82.00, 'x': 71.01, 'y': -41.00, 'z': -542.85},
     {'r': 82.00, 'x': 71.01, 'y': 41.00, 'z': -542.85},
     {'r': 142.03, 'x': 71.01, 'y': 123.00, 'z': -542.85},
     {'r': 216.95, 'x': 71.01, 'y': 205.00, 'z': -542.85},
     {'r': 295.65, 'x': 71.01, 'y': 287.00, 'z': -542.85},
     {'r': 284.06, 'x': 142.03, 'y': -246.00, 'z': -542.85},
     {'r': 216.95, 'x': 142.03, 'y': -164.00, 'z': -542.85},
     {'r': 164.00, 'x': 142.03, 'y': -82.00, 'z': -542.85},
     {'r': 142.03, 'x': 142.03, 'y': 0.00, 'z': -542.85},
     {'r': 164.00, 'x': 142.03, 'y': 82.00, 'z': -542.85},
     {'r': 216.95, 'x': 142.03, 'y': 164.00, 'z': -542.85},
     {'r': 284.06, 'x': 142.03, 'y': 246.00, 'z': -542.85},
     {'r': 295.65, 'x': 213.04, 'y': -205.00, 'z': -542.85},
     {'r': 246.00, 'x': 213.04, 'y': -123.00, 'z': -542.85},
     {'r': 216.95, 'x': 213.04, 'y': -41.00, 'z': -542.85},
     {'r': 216.95, 'x': 213.04, 'y': 41.00, 'z': -542.85},
     {'r': 246.00, 'x': 213.04, 'y': 123.00, 'z': -542.85},
     {'r': 295.65, 'x': 213.04, 'y': 205.00, 'z': -542.85},
     {'r': 328.00, 'x': 284.06, 'y': -164.00, 'z': -542.85},
     {'r': 295.65, 'x': 284.06, 'y': -82.00, 'z': -542.85},
     {'r': 284.06, 'x': 284.06, 'y': 0.00, 'z': -542.85},
     {'r': 295.65, 'x': 284.06, 'y': 82.00, 'z': -542.85},
     {'r': 328.00, 'x': 284.06, 'y': 164.00, 'z': -542.85},
     {'r': 295.65, 'x': -71.01, 'y': -287.00, 'z': -542.85},
     {'r': 216.95, 'x': -71.01, 'y': -205.00, 'z': -542.85},
     {'r': 142.03, 'x': -71.01, 'y': -123.00, 'z': -542.85},
     {'r': 82.00, 'x': -71.01, 'y': -41.00, 'z': -542.85},
     {'r': 82.00, 'x': -71.01, 'y': 41.00, 'z': -542.85},
     {'r': 142.03, 'x': -71.01, 'y': 123.00, 'z': -542.85},
     {'r': 216.95, 'x': -71.01, 'y': 205.00, 'z': -542.85},
     {'r': 295.65, 'x': -71.01, 'y': 287.00, 'z': -542.85},
     {'r': 284.06, 'x': -142.03, 'y': -246.00, 'z': -542.85},
     {'r': 216.95, 'x': -142.03, 'y': -164.00, 'z': -542.85},
     {'r': 164.00, 'x': -142.03, 'y': -82.00, 'z': -542.85},
     {'r': 142.03, 'x': -142.03, 'y': 0.00, 'z': -542.85},
     {'r': 164.00, 'x': -142.03, 'y': 82.00, 'z': -542.85},
     {'r': 216.95, 'x': -142.03, 'y': 164.00, 'z': -542.85},
     {'r': 284.06, 'x': -142.03, 'y': 246.00, 'z': -542.85},
     {'r': 295.65, 'x': -213.04, 'y': -205.00, 'z': -542.85},
     {'r': 246.00, 'x': -213.04, 'y': -123.00, 'z': -542.85},
     {'r': 216.95, 'x': -213.04, 'y': -41.00, 'z': -542.85},
     {'r': 216.95, 'x': -213.04, 'y': 41.00, 'z': -542.85},
     {'r': 246.00, 'x': -213.04, 'y': 123.00, 'z': -542.85},
     {'r': 295.65, 'x': -213.04, 'y': 205.00, 'z': -542.85},
     {'r': 328.00, 'x': -284.06, 'y': -164.00, 'z': -542.85},
     {'r': 295.65, 'x': -284.06, 'y': -82.00, 'z': -542.85},
     {'r': 284.06, 'x': -284.06, 'y': 0.00, 'z': -542.85},
     {'r': 295.65, 'x': -284.06, 'y': 82.00, 'z': -542.85},
     {'r': 328.00, 'x': -284.06, 'y': 164.00, 'z': -542.85}

]

class DataPreprocessor:
    def __init__(self, min_smearing=5, max_smearing=15):
        self.min_smearing = min_smearing / 100
        self.max_smearing = max_smearing / 100

    def add_gaussian_smearing(self, values):
        # Randomly select smearing percentage between min and max for each batch
        smearing_percentage = np.random.uniform(self.min_smearing, self.max_smearing)
        sigma = smearing_percentage * values
        return values + np.random.normal(0, sigma, size=values.shape)

    def normalize(self, values):
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val + 1e-8)

    def preprocess(self, charges, deltas):
        # Apply smearing and normalization to both charges and deltas
        charges = self.add_gaussian_smearing(charges)
        deltas = self.add_gaussian_smearing(deltas)
        
        # Normalize
        charges = self.normalize(charges)
        deltas = self.normalize(deltas)
        
        return charges, deltas

# Then modify your PMTDataset to use it:
class PMTDataset(Dataset):
    def __init__(
        self, 
        h5_filename: str,
        pmt_positions: List[Dict],
        embedding_dim: int = 8,
        chunk_size: int = 1000,
        cache_size: int = 5000,
        min_smearing: float = 5,
        max_smearing: float = 15
    ):
        self.h5_filename = h5_filename
        self.pmt_positions = pmt_positions
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.cache_size = cache_size
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(min_smearing, max_smearing)
        
        # Rest of your initialization code...
        self.cache = {}
        self.cache_order = []
        self._init_position_encodings()
        
        with tables.open_file(h5_filename, mode='r') as file:
            block0_values = file.root.df.block0_values[:]
            self.event_numbers = np.unique(block0_values[:, 0].astype(int))
            self.length = len(self.event_numbers)

    def _process_event(self, event_num: int) -> torch.Tensor:
        """Process a single event and create its encoding"""
        with tables.open_file(self.h5_filename, mode='r') as file:
            block0_values = file.root.df.block0_values[:]
            block1_values = file.root.df.block1_values[:]
            
            event_mask = block0_values[:, 0].astype(int) == event_num
            event_pmts = block0_values[event_mask, 1].astype(int)
            event_charges = block1_values[event_mask, 0]
            event_deltas = block1_values[event_mask, 1]
        
        # Use preprocessor to handle smearing and normalization
        q_norm, dt_norm = self.preprocessor.preprocess(event_charges, event_deltas)
        
        # Create tensor
        n_pmts = len(self.pmt_positions)
        total_dim = 5 * self.embedding_dim
        event_tensor = torch.zeros((n_pmts, total_dim))
        
        for pmt_idx, (pmt_num, q, dt) in enumerate(zip(event_pmts, q_norm, dt_norm)):
            if pmt_num < n_pmts:
                # Create Q and Δt encodings
                q_emb = self._create_sparse_encoding(np.array([q]), min_freq=1.0, max_freq=5000.0)
                dt_emb = self._create_sparse_encoding(np.array([dt]), min_freq=1.0, max_freq=1000.0)
                
                # Populate tensor
                event_tensor[pmt_num, :self.embedding_dim] = q_emb
                event_tensor[pmt_num, self.embedding_dim:2*self.embedding_dim] = dt_emb
                event_tensor[pmt_num, 2*self.embedding_dim:3*self.embedding_dim] = self.pos_encodings['x'][pmt_num]
                event_tensor[pmt_num, 3*self.embedding_dim:4*self.embedding_dim] = self.pos_encodings['y'][pmt_num]
                event_tensor[pmt_num, 4*self.embedding_dim:] = self.pos_encodings['z'][pmt_num]
        
        return event_tensor


    def _update_cache(self, idx: int, tensor: torch.Tensor):
        """Update the cache with new tensor"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        self.cache[idx] = tensor
        self.cache_order.append(idx)
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx in self.cache:
            return self.cache[idx]
        
        event_num = self.event_numbers[idx]
        tensor = self._process_event(event_num)
        self._update_cache(idx, tensor)
        return tensor

def create_dataloader(
    h5_filename: str,
    pmt_positions: List[Dict],
    embedding_dim: int = 8,
    min_smearing: int = 5,
    max_smearing: int = 15,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader with memory-efficient dataset"""
    dataset = PMTDataset(h5_filename, pmt_positions, embedding_dim)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )



