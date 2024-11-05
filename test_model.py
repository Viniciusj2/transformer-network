import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import tables
import math
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

# PMT positions
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

class TransformerModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        total_dim = 5 * embedding_dim  # Q, Δt, X, Y, Z each has embedding_dim dimensions
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=total_dim,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Decoder for Q and Δt predictions
        self.q_decoder = nn.Linear(total_dim, 1)
        self.dt_decoder = nn.Linear(total_dim, 1)
        
    def forward(self, x):
        # Handle NaN values by replacing them with 0
        x = torch.nan_to_num(x, nan=0.0)
        x = self.transformer(x)
        
        # Decode Q and Δt
        q_pred = self.q_decoder(x)
        dt_pred = self.dt_decoder(x)
        
        return torch.cat([q_pred, dt_pred], dim=-1)

def create_sparse_encoding(values, dim):
    """Creates sinusoidal sparse encoding for a single feature"""
    assert dim % 2 == 0, "embedding dimension must be even"
    half_dim = dim // 2
    
    # Calculate frequency bands
    freq_bands = torch.exp(
        torch.arange(half_dim, dtype=torch.float32) * 
        (-math.log(10000.0) / (half_dim - 1))
    )
    
    # Calculate embeddings
    values_tensor = torch.tensor(values, dtype=torch.float32)
    if values_tensor.dim() == 1:
        values_tensor = values_tensor.unsqueeze(1)
    
    emb = values_tensor * freq_bands[None, :]
    sin_emb = torch.sin(emb)
    cos_emb = torch.cos(emb)
    
    return torch.cat([sin_emb, cos_emb], dim=1)

def preprocess_pmt_data_for_transformer_with_full_embeddings(h5_filename, embedding_dim=8):
    """
    Preprocesses PMT data with sparse encoding for all features (Q, Δt, X, Y, Z)
    
    Args:
        h5_filename: String path to the h5 file
        embedding_dim: Integer dimension for feature embeddings
    
    Returns:
        torch.Tensor of shape (n_events, n_pmts, 5*embedding_dim)
    """
    # Get raw data
    with tables.open_file(h5_filename, mode='r') as file:
        block0_values = file.root.df.block0_values[:]
        block1_values = file.root.df.block1_values[:]
        
        event_numbers = block0_values[:, 0].astype(int)
        pmt_numbers = block0_values[:, 1].astype(int)
        charges = block1_values[:, 0]
        delta_times = block1_values[:, 1]
    
    unique_events = np.unique(event_numbers)
    n_pmts = len(np.unique(pmt_numbers))
    
    # Normalize coordinates and features
    coords_mean = np.zeros(3)
    coords_std = np.ones(3)
    
    # Create embeddings for coordinates
    x_embeddings = create_sparse_encoding(np.zeros(n_pmts), embedding_dim)
    y_embeddings = create_sparse_encoding(np.zeros(n_pmts), embedding_dim)
    z_embeddings = create_sparse_encoding(np.zeros(n_pmts), embedding_dim)
    
    # Initialize output tensor
    # Shape: (n_events, n_pmts, 5*embedding_dim)
    total_dim = 5 * embedding_dim  # Q, Δt, X, Y, Z
    processed_data = torch.zeros((len(unique_events), n_pmts, total_dim))
    
    # Process each event
    for event_idx, event_num in enumerate(unique_events):
        event_mask = event_numbers == event_num
        event_pmts = pmt_numbers[event_mask]
        event_charges = charges[event_mask]
        event_deltas = delta_times[event_mask]
        
        # Normalize Q and Δt for this event
        q_norm = (event_charges - np.min(event_charges)) / (np.max(event_charges) - np.min(event_charges) + 1e-8)
        dt_norm = (event_deltas - np.min(event_deltas)) / (np.max(event_deltas) - np.min(event_deltas) + 1e-8)
        
        for pmt_idx, (pmt_num, q, dt) in enumerate(zip(event_pmts, q_norm, dt_norm)):
            if pmt_num < n_pmts:
                # Create and add Q embedding
                q_emb = create_sparse_encoding(torch.tensor([q]), embedding_dim)
                processed_data[event_idx, pmt_num, :embedding_dim] = q_emb
                
                # Create and add Δt embedding
                dt_emb = create_sparse_encoding(torch.tensor([dt]), embedding_dim)
                processed_data[event_idx, pmt_num, embedding_dim:2*embedding_dim] = dt_emb
                
                # Add positional embeddings
                processed_data[event_idx, pmt_num, 2*embedding_dim:3*embedding_dim] = x_embeddings[pmt_num]
                processed_data[event_idx, pmt_num, 3*embedding_dim:4*embedding_dim] = y_embeddings[pmt_num]
                processed_data[event_idx, pmt_num, 4*embedding_dim:] = z_embeddings[pmt_num]
    
    return processed_data

def create_masked_data(batch, mask_prob=0.50):
    """Creates masked data for training with masking across all feature dimensions."""
    device = batch.device
    embedding_dim = batch.shape[2] // 5  # Total dims / number of features
    
    # Create mask for all features
    mask = torch.rand(batch.shape[0], batch.shape[1], 5, device=device) < mask_prob
    
    masked_data = batch.clone()
    
    # Apply mask to each feature's embedding
    for i in range(5):
        start_idx = i * embedding_dim
        end_idx = (i + 1) * embedding_dim
        masked_data[:, :, start_idx:end_idx][mask[:, :, i].unsqueeze(-1).expand(-1, -1, embedding_dim)] = float('nan')
    
    return masked_data, batch, mask

def train_model(model, train_data, num_epochs=100, learning_rate=1e-3, batch_size=64, log_dir="runs_3"):
    # Setup TensorBoard
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(log_dir, current_time)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_data = train_data.to(device)
    
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Log model summary
    model_summary = f"""
    Model Architecture:
    {str(model)}
    
    Total Parameters: {sum(p.numel() for p in model.parameters())}
    Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}
    """
    writer.add_text('Model/Architecture', model_summary)
    
    # Training metrics
    best_loss = float('inf')
    steps = 0
    model.train()
    
    # Create a dictionary to store moving averages of metrics
    moving_avg = {
        'loss': [],
        'window_size': 100
    }

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        # Shuffle data
        indices = torch.randperm(len(train_data))
        train_data = train_data[indices]
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            
            # Create masked data
            masked_data, target_data, mask = create_masked_data(batch)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(masked_data)
            
            # Calculate loss
            # We only need to compare Q and Δt predictions (the first two features)
            embedding_dim = batch.shape[2] // 5
            q_pred = predictions[:, :, 0].unsqueeze(-1)
            dt_pred = predictions[:, :, 1].unsqueeze(-1)
            
            q_target = target_data[:, :, :embedding_dim]
            dt_target = target_data[:, :, embedding_dim:2*embedding_dim]
            
            # Only calculate loss on masked values
            q_mask = mask[:, :, 0]
            dt_mask = mask[:, :, 1]
            
            q_loss = criterion(q_pred[q_mask], q_target[q_mask][:, 0].unsqueeze(-1))
            dt_loss = criterion(dt_pred[dt_mask], dt_target[dt_mask][:, 0].unsqueeze(-1))
            
            loss = (q_loss + dt_loss) / 2
            
            if not torch.isnan(loss):
                loss.backward()
                
                # Update moving average
                moving_avg['loss'].append(loss.item())
                if len(moving_avg['loss']) > moving_avg['window_size']:
                    moving_avg['loss'].pop(0)
                
                # Log gradients
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isnan(param.grad).any():
                        writer.add_histogram(f'gradients/{name}', 
                                          param.grad.cpu().detach().numpy(), 
                                          steps)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Log metrics
                if steps % 10 == 0:
                    avg_loss = sum(moving_avg['loss']) / len(moving_avg['loss'])
                    writer.add_scalar('Loss/train', avg_loss, steps)
                    writer.add_scalar('Loss/train_raw', loss.item(), steps)
                    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], steps)
                    
                    # Log prediction statistics
                    if not torch.isnan(predictions).any():
                        writer.add_scalar('Predictions/mean_Q', q_pred[q_mask].mean().item(), steps)
                        writer.add_scalar('Predictions/mean_Δt', dt_pred[dt_mask].mean().item(), steps)
                
                epoch_loss += loss.item()
                n_batches += 1
                steps += 1
        
        if n_batches > 0:
            avg_epoch_loss = epoch_loss / n_batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
            
            # Update learning rate scheduler
            scheduler.step(avg_epoch_loss)
            
            # Log model parameters
            for name, param in model.named_parameters():
                if not torch.isnan(param).any():
                    try:
                        writer.add_histogram(f'parameters/{name}', 
                                          param.cpu().detach().numpy(), 
                                          epoch)
                    except Exception as e:
                        print(f"Warning: Could not log histogram for {name}: {e}")
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                model_save_path = os.path.join(log_dir, 'best_model.pth')
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, model_save_path)
                    print(f"Saved best model to {model_save_path}")
                except Exception as e:
                    print(f"Warning: Could not save model: {e}")
    
    writer.close()
    print("Training completed.")
    return log_dir

def evaluate_model(model, test_data, save_file_path=None, writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_data = test_data.to(device)
    
    model.eval()
    with torch.no_grad():
        masked_data, target_data, mask = create_masked_data(test_data)
        predictions = model(masked_data)
        
        # Extract Q and Δt predictions and targets
        embedding_dim = test_data.shape[2] // 5
        q_pred = predictions[:, :, 0].unsqueeze(-1)
        dt_pred = predictions[:, :, 1].unsqueeze(-1)
        
        q_target = target_data[:, :, :embedding_dim]
        dt_target = target_data[:, :, embedding_dim:2*embedding_dim]
        
        # Calculate MSE for Q and Δt separately
        q_mask = mask[:, :, 0]
        dt_mask = mask[:, :, 1]
        
        q_mse = nn.MSELoss()(q_pred[q_mask], q_target[q_mask][:, 0].unsqueeze(-1)).item()
        dt_mse = nn.MSELoss()(dt_pred[dt_mask], dt_target[dt_mask][:, 0].unsqueeze(-1)).item()
        
        total_mse = (q_mse + dt_mse) / 2
        
        # Log test metrics if writer is provided
        if writer is not None:
            writer.add_scalar('Loss/test', total_mse)
            writer.add_scalar('Loss/test_Q', q_mse)
            writer.add_scalar('Loss/test_Δt', dt_mse)
            
            # Log prediction vs target distributions
            writer.add_histogram('test/predictions_Q', q_pred[q_mask])
            writer.add_histogram('test/predictions_Δt', dt_pred[dt_mask])
            writer.add_histogram('test/targets_Q', q_target[q_mask][:, 0])
            writer.add_histogram('test/targets_Δt', dt_target[dt_mask][:, 0])
    
        # Save the predictions and targets to a file if a file path is provided
        if save_file_path is not None:
            torch.save({
                'predictions_Q': q_pred.cpu(),
                'predictions_Δt': dt_pred.cpu(),
                'targets_Q': q_target.cpu(),
                'targets_Δt': dt_target.cpu()
            }, save_file_path)
            print(f"Predictions and targets saved to: {save_file_path}")
    
    return total_mse, q_mse, dt_mse

def main():
    # Load and preprocess data
    h5_filename = "pmt_Cosmic_analysis1620_5_threshold.h5"
    embedding_dim = 8
    train_data = preprocess_pmt_data_for_transformer_with_full_embeddings(h5_filename, embedding_dim)
    
    # Split data into train and test sets
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    # Create model
    model = TransformerModel(embedding_dim=embedding_dim, num_heads=4, num_layers=2)
    
    # Train the model
    log_dir = train_model(model, train_data, num_epochs=100, learning_rate=1e-3, batch_size=64)
    
    # Evaluate the model on the test set
    test_output_path = os.path.join(log_dir, "test_outputs.pt")
    writer = SummaryWriter(log_dir)
    test_mse, q_mse, dt_mse = evaluate_model(model, test_data, save_file_path=test_output_path, writer=writer)
    
    print(f"Test MSE: {test_mse:.4f}, Q MSE: {q_mse:.4f}, Δt MSE: {dt_mse:.4f}")
    
    writer.close()

if __name__ == "__main__":
    main()
