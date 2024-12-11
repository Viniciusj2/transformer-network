import torch
import torch.nn as nn
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
        
        self.cache = {}
        self.cache_order = []
        
        # Initialize position encodings
        self._init_position_encodings()
        
        with tables.open_file(h5_filename, mode='r') as file:
            block0_values = file.root.df.block0_values[:]
            self.event_numbers = np.unique(block0_values[:, 0].astype(int))
            self.length = len(self.event_numbers)

    def _init_position_encodings(self):
        """Initialize position encodings for x, y, z coordinates"""
        self.pos_encodings = {
            'x': self._create_position_embedding([pos['x'] for pos in self.pmt_positions]),
            'y': self._create_position_embedding([pos['y'] for pos in self.pmt_positions]),
            'z': self._create_position_embedding([pos['z'] for pos in self.pmt_positions])
        }

    def _create_position_embedding(self, coordinates: List[float]) -> torch.Tensor:
        """
        Create position embeddings using min-max normalization
        
        Args:
            coordinates (List[float]): List of coordinate values
        
        Returns:
            torch.Tensor: Normalized and embedded position coordinates
        """
        # Normalize coordinates
        coords_array = np.array(coordinates)
        normalized_coords = (coords_array - np.min(coords_array)) / (np.max(coords_array) - np.min(coords_array) + 1e-8)
        
        # Create embedding
        embeddings = np.zeros((len(coordinates), self.embedding_dim))
        for i, coord in enumerate(normalized_coords):
            embeddings[i] = self._create_sparse_encoding(np.array([coord]))
        
        return torch.tensor(embeddings, dtype=torch.float32)
    def _create_linear_encoding(
        self, 
        values: np.ndarray, 
        min_val: float = 0.0, 
        max_val: float = 1.0
    ) -> torch.Tensor:
     
        # Create linear scales between min_val and max_val across emb_dim
        linear_scale = np.linspace(min_val, max_val, num=self.embedding_dim)
        
        # Expand dimensions for broadcasting
        values = values[:, np.newaxis]  
        
        # Apply linear scaling
        encoding = values * linear_scale  
        
        return torch.tensor(encoding, dtype=torch.float32)

    def _create_sparse_encoding(
        self, 
        values: np.ndarray, 
        min_freq: float = 1.0, 
        max_freq: float = 5000.0
    ) -> torch.Tensor:

        # Logarithmic scaling of frequencies - Following Implementation Done by Zev
        log_freq = np.geomspace(min_freq, max_freq, num=self.embedding_dim//2)
        
        # Create encoding using sine and cosine of scaled frequencies
        encoding = np.zeros((len(values), self.embedding_dim))
        for i, val in enumerate(values):
            # Sine encoding
            sine_encoding = np.sin(val * log_freq * np.pi)
            # Cosine encoding
            cosine_encoding = np.cos(val * log_freq * np.pi)
            
            # Combine sine and cosine
            encoding[i, :self.embedding_dim//2] = sine_encoding
            encoding[i, self.embedding_dim//2:] = cosine_encoding
        
        return torch.tensor(encoding.squeeze(), dtype=torch.float32)

    def _process_event(self, event_num: int) -> torch.Tensor:
        with tables.open_file(self.h5_filename, mode='r') as file:
            block0_values = file.root.df.block0_values[:]
            block1_values = file.root.df.block1_values[:]
            
            event_mask = block0_values[:, 0].astype(int) == event_num
            event_pmts = block0_values[event_mask, 1].astype(int)
            event_charges = block1_values[event_mask, 0]
            event_deltas = block1_values[event_mask, 1]
            
            # Create tensor with original size
            n_pmts = len(self.pmt_positions)
            total_dim = 5 * self.embedding_dim
            event_tensor = torch.zeros((n_pmts, total_dim))
            
            # Preprocess charges and deltas
            q_norm, dt_norm = self.preprocessor.preprocess(event_charges, event_deltas)
            
            # Populate tensor for ALL PMTs
            for i in range(n_pmts):
                # Find if this PMT was in the original event
                pmt_index = np.where(event_pmts == i)[0]
                
                if len(pmt_index) > 0:
                    # PMT was in the event
                    q = q_norm[pmt_index[0]]
                    dt = dt_norm[pmt_index[0]]
                    raw_q = event_charges[pmt_index[0]]
                    raw_dt = event_deltas[pmt_index[0]]
                else:
                    # PMT was not in the event - This avoids previous issue for PMTs not in event
                    q = 0.0
                    dt = 0.0
                    raw_q = 0.0
                    raw_dt = 0.0
                
                # Create embeddings for this PMT
                q_emb = self._create_linear_encoding(np.array([q]), min_val=0.0, max_val=1.0)
                dt_emb = self._create_linear_encoding(np.array([dt]), min_val=0.0, max_val=1.0)
                
                # Populate tensor
                event_tensor[i, 0] = raw_q
                event_tensor[i, 1:self.embedding_dim] = q_emb.squeeze()[1:]
                event_tensor[i, self.embedding_dim] = raw_dt
                event_tensor[i, self.embedding_dim+1:2*self.embedding_dim] = dt_emb.squeeze()[1:]
                
                # Position encodings remain the same
                event_tensor[i, 2*self.embedding_dim:3*self.embedding_dim] = self.pos_encodings['x'][i]
                event_tensor[i, 3*self.embedding_dim:4*self.embedding_dim] = self.pos_encodings['y'][i]
                event_tensor[i, 4*self.embedding_dim:] = self.pos_encodings['z'][i]
            
            return event_tensor
            
    def _update_cache(self, idx: int, tensor: torch.Tensor):

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


class ModifiedTransformerModel(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, num_layers: int):
        super(ModifiedTransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Expand Q and T to embedding dimensions
        self.q_expander = nn.Linear(1, embedding_dim)
        self.dt_expander = nn.Linear(1, embedding_dim)
        
        # Positional encoding for X, Y, Z
        self.pos_encoder = nn.Linear(3, embedding_dim)
        
        # Transformer setup
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
    
        # Decoders
        self.q_decoder = nn.Linear(embedding_dim, 1)
        self.dt_decoder = nn.Linear(embedding_dim, 1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        q_vals = x[:, :, 0:self.embedding_dim]  
        dt_vals = x[:, :, self.embedding_dim:2*self.embedding_dim]  
        x_pos = x[:, :, 2*self.embedding_dim:3*self.embedding_dim] 
        y_pos = x[:, :, 3*self.embedding_dim:4*self.embedding_dim] 
        z_pos = x[:, :, 4*self.embedding_dim:]  
        
        # Extract first element of each embedding
        q_extract = q_vals[:, :, 0:1]  
        dt_extract = dt_vals[:, :, 0:1]  
        pos_extract = torch.cat([x_pos[:, :, 0:1], y_pos[:, :, 0:1], z_pos[:, :, 0:1]], dim=-1)
        
        # Print debug information
        print("Charge extract shape:", q_extract.shape)
        print("Delta t extract shape:", dt_extract.shape)
        print("Position extract shape:", pos_extract.shape)
        
        # Expansion and processing
        q_expanded = self.q_expander(q_extract)
        dt_expanded = self.dt_expander(dt_extract)
        pos_embedded = self.pos_encoder(pos_extract)
        
        # Combine features by addition
        combined_features = q_expanded + dt_expanded + pos_embedded
        print("Input tensor shape:", combined_features.shape)
        print("Input tensor dtype:", combined_features.dtype)
        
        # Transform
        transformer_output = self.transformer(combined_features)
        
        # Generate predictions
        q_pred = self.q_decoder(transformer_output)
        dt_pred = self.dt_decoder(transformer_output)
        
        return torch.cat([q_pred, dt_pred], dim=-1)