import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tables
from typing import List, Dict

# Reduced Model is Similar to previous model but Z encoding treatement is Different were Z is only 1 dimensional
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

# DataPreprocessor - Adds Gaussian Smearing and Normalizes Data
class DataPreprocessor:
    def __init__(self, min_smearing=5, max_smearing=15):
        self.min_smearing = min_smearing / 100
        self.max_smearing = max_smearing / 100

    def add_gaussian_smearing(self, values):
        smearing_percentage = np.random.uniform(self.min_smearing, self.max_smearing)
        sigma = smearing_percentage * values
        return values + np.random.normal(0, sigma, size=values.shape)

    def normalize(self, values):
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val + 1e-8)

    def preprocess(self, charges, deltas):
        charges = self.add_gaussian_smearing(charges)
        deltas = self.add_gaussian_smearing(deltas)
        
        charges = self.normalize(charges)
        deltas = self.normalize(deltas)
        
        return charges, deltas

#PMTDataset - Loads from h5 file, Applied Sinusoidal positional encoding and creates Feature Vector for Transformer Model
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
        x_coords = [pos['x'] for pos in self.pmt_positions]
        y_coords = [pos['y'] for pos in self.pmt_positions]
        z_coords = [pos['z'] for pos in self.pmt_positions]
        
        # Determine x and y embedding dimensions
        x_y_dim = (self.embedding_dim - 1) // 2
        
        # Different frequency ranges for x and y
        self.pos_encodings = {
            'x': self._create_position_embedding(
                x_coords, 
                embedding_dim=x_y_dim,  # Specify the correct dimension
                min_freq=1.0,   
                max_freq=1000.0 
            ).numpy(),
            'y': self._create_position_embedding(
                y_coords, 
                embedding_dim=x_y_dim,  # Specify the correct dimension
                min_freq=1.0,   
                max_freq=10000.0 
            ).numpy(),
            'z': self._create_position_embedding(
                [pos['z'] for pos in self.pmt_positions], 
                embedding_dim=1  # Z will be a single dimension
            ).numpy()
        }

    def _create_position_embedding(
        self, 
        coordinates: List[float], 
        embedding_dim: int = None,
        min_freq: float = 1.0, 
        max_freq: float = 1.0
    ) -> torch.Tensor:

        # Use full embedding_dim if not specified
        if embedding_dim is None:
            embedding_dim = self.embedding_dim
        
        # Normalize coordinates
        coords_array = np.array(coordinates)
        normalized_coords = (coords_array - np.min(coords_array)) / (np.max(coords_array) - np.min(coords_array) + 1e-8)
        
        # Create embedding
        embeddings = np.zeros((len(coordinates), embedding_dim))
        for i, coord in enumerate(normalized_coords):
            # Adjust _create_sparse_encoding to accept embedding_dim
            embeddings[i] = self._create_sparse_encoding(
                np.array([coord]), 
                embedding_dim=embedding_dim,
                min_freq=min_freq, 
                max_freq=max_freq
            )
        
        return torch.tensor(embeddings, dtype=torch.float32)

    def _create_sparse_encoding(
        self, 
        values: np.ndarray, 
        embedding_dim: int = None,
        min_freq: float = 1.0, 
        max_freq: float = 5000.0
    ) -> np.ndarray:
        
        # Use full embedding_dim if not specified
        if embedding_dim is None:
            embedding_dim = self.embedding_dim
        
        values = np.atleast_1d(values)
        
        # If embedding_dim is 1, just normalize the values
        if embedding_dim == 1:
            # Min-max normalization
            normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
            return normalized_values.reshape(-1, 1)
        
        # Logarithmic scaling of frequencies
        log_freq = np.geomspace(min_freq, max_freq, num=embedding_dim//2)
        
        encoding = np.zeros((len(values), embedding_dim))
        for i, val in enumerate(values):
            # Sine encoding
            sine_encoding = np.sin(val * log_freq * np.pi)
            # Cosine encoding
            cosine_encoding = np.cos(val * log_freq * np.pi)
            
            # Combine sine and cosine
            encoding[i, :embedding_dim//2] = sine_encoding
            encoding[i, embedding_dim//2:] = cosine_encoding
        
        return encoding.squeeze()

    def _process_event(self, event_num: int) -> torch.Tensor:
        with tables.open_file(self.h5_filename, mode='r') as file:
            block0_values = file.root.df.block0_values[:]
            block1_values = file.root.df.block1_values[:]
            
            event_mask = block0_values[:, 0].astype(int) == event_num
            event_pmts = block0_values[event_mask, 1].astype(int)
            event_charges = block1_values[event_mask, 0]
            event_deltas = block1_values[event_mask, 1]
            
            # Adjust total dimensions
            x_y_dim = (self.embedding_dim - 1) // 2
            total_dim = 2 * self.embedding_dim - 2  # Reduced dimensions
            
            # Create tensor with original size
            n_pmts = len(self.pmt_positions)
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
                    # PMT was not in the event
                    q = 0.0
                    dt = 0.0
                    raw_q = 0.0
                    raw_dt = 0.0
                
                # Populate tensor
                event_tensor[i, 0] = raw_q  # Raw charge
                event_tensor[i, 1:x_y_dim+1] = torch.from_numpy(self.pos_encodings['x'][i])
                event_tensor[i, x_y_dim+1:2*x_y_dim+1] = torch.from_numpy(self.pos_encodings['y'][i])
                event_tensor[i, 2*x_y_dim+1:2*x_y_dim+2] = torch.from_numpy(self.pos_encodings['z'][i])
                
                event_tensor[i, 2*x_y_dim+2] = raw_dt  # Raw delta t
                event_tensor[i, 2*x_y_dim+3] = dt  # Normalized delta t
            
            return event_tensor
    
    def _update_cache(self, idx: int, tensor: torch.Tensor):
        """Update the cache with new tensor"""
        if len(self.cache) >= self.cache_size:
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

#ModifiedTransformerModel - Takes in Q, delta_t, and position information -- combines them and feeds them to transformer
class ModifiedTransformerModel(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, num_layers: int):
        super(ModifiedTransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Adjust embedding dimensions
        x_y_dim = (embedding_dim - 1) // 2
        
        # Expanders with explicit output dimensions
        self.q_expander = nn.Linear(1, embedding_dim)
        self.dt_expander = nn.Linear(1, embedding_dim)
        
        # Positional encoding for X, Y, Z 
        self.x_encoder = nn.Linear(x_y_dim, embedding_dim)
        self.y_encoder = nn.Linear(x_y_dim, embedding_dim)
        self.z_encoder = nn.Linear(1, 1)  # Keep Z as single dimension
        
        # Transformer setup (unchanged)
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
        x_y_dim = (self.embedding_dim - 1) // 2
        
        # Extract values from reduced tensor
        q_extract = x[:, :, 0:1]  # Raw charge
        x_extract = x[:, :, 1:x_y_dim+1]  # X position
        y_extract = x[:, :, x_y_dim+1:2*x_y_dim+1]  # Y position
        z_extract = x[:, :, 2*x_y_dim+1:2*x_y_dim+2]  # Z position
        dt_normalized = x[:, :, 2*x_y_dim+3:2*x_y_dim+4]  # Normalized delta t
        
        # Expansion and processing
        q_expanded = self.q_expander(q_extract)
        dt_expanded = self.dt_expander(dt_normalized)
        
        # Print debug information
        print("Charge extract shape:", q_extract.shape)
        print("Delta t extract shape:", dt_normalized.shape)
        #print("Position extract shape:", pos_extract.shape)

        x_embedded = self.x_encoder(x_extract)
        y_embedded = self.y_encoder(y_extract)
        z_embedded = self.z_encoder(z_extract).repeat(1, 1, self.embedding_dim)
        
        # Combine features
        combined_features = q_expanded + dt_expanded + x_embedded + y_embedded + z_embedded
        
        print("Input tensor shape:", combined_features.shape)
        print("Input tensor dtype:", combined_features.dtype)
        
        # Transform
        transformer_output = self.transformer(combined_features)
        
        # Generate predictions
        q_pred = self.q_decoder(transformer_output)
        dt_pred = self.dt_decoder(transformer_output)
        
        return torch.cat([q_pred, dt_pred], dim=-1)