import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Tuple, List
from data_loader import PMTDataset


def plot_encoded_data(
    model: torch.nn.Module,
    dataset: PMTDataset,
    event_idx: int,
    save_path: str,
    figsize: Tuple[int, int] = (20, 15)
) -> None:
    device = next(model.parameters()).device
    event_data = dataset[event_idx].unsqueeze(0).to(device)
    
    with torch.no_grad():
        x = torch.nan_to_num(event_data, nan=0.0)
        
        # Split input into components
        q_enc = x[:, :, :model.embedding_dim]
        dt_enc = x[:, :, model.embedding_dim:2*model.embedding_dim]
        pos_enc = x[:, :, 2*model.embedding_dim:]
        
        # Process encoded features
        q_processed = model.q_processor(q_enc)
        dt_processed = model.dt_processor(dt_enc)
        pos_processed = model.pos_processor(pos_enc)
        

        # Split input into components
        q_enc = x[:, :, :model.embedding_dim]
        dt_enc = x[:, :, model.embedding_dim:2*model.embedding_dim]
        pos_enc = x[:, :, 2*model.embedding_dim:]

        # Process encoded features
        q_processed = model.q_processor(q_enc)
        dt_processed = model.dt_processor(dt_enc)
        pos_processed = model.pos_processor(pos_enc)

        # Combine features
        q_with_pos = q_processed + pos_processed
        dt_with_pos = dt_processed + pos_processed

        # Concatenate features
        combined = torch.cat([q_with_pos, dt_with_pos], dim=-1)

        # Transformer input is now directly the combined tensor
        transformer_input = combined

        # Detailed logging of feature processing
        print("Q Processed Shape:", q_processed.shape)
        print("Q Processed Sample:", q_processed[0, :5, :])
        
        print("Delta T Processed Shape:", dt_processed.shape)
        print("Delta T Processed Sample:", dt_processed[0, :5, :])
        
        print("Position Processed Shape:", pos_processed.shape)
        print("Position Processed Sample:", pos_processed[0, :5, :])
        
        print("Q with Pos Shape:", q_with_pos.shape)
        print("Q with Pos Sample:", q_with_pos[0, :5, :])
        
        print("Delta T with Pos Shape:", dt_with_pos.shape)
        print("Delta T with Pos Sample:", dt_with_pos[0, :5, :])
  
        print("Combined Shape:", combined.shape)  # Should now be 1, 122, 16
        print("Transformer Input Shape:", transformer_input.shape)  # Also 1, 122, 16

    # Move tensors back to CPU for plotting
    q_enc = q_enc.cpu().numpy()
    dt_enc = dt_enc.cpu().numpy()
    pos_enc = pos_enc.cpu().numpy()
    q_processed = q_processed.cpu().numpy()
    dt_processed = dt_processed.cpu().numpy()
    pos_processed = pos_processed.cpu().numpy()
    combined = combined.cpu().numpy()
    transformer_input = transformer_input.cpu().numpy()
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Raw Encodings
    ax1 = fig.add_subplot(gs[0, 0])
    plot_encoding_heatmap(q_enc[0], ax1, "Q Encoding")
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_encoding_heatmap(dt_enc[0], ax2, "ΔT Encoding")
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_encoding_heatmap(pos_enc[0], ax3, "Position Encoding")
    
    # 2. Processed Features
    ax4 = fig.add_subplot(gs[1, 0])
    plot_encoding_heatmap(q_processed[0], ax4, "Processed Q Features")
    
    ax5 = fig.add_subplot(gs[1, 1])
    plot_encoding_heatmap(dt_processed[0], ax5, "Processed ΔT Features")
    
    ax6 = fig.add_subplot(gs[1, 2])
    plot_encoding_heatmap(pos_processed[0], ax6, "Processed Position Features")
    
    # 3. Combined and Final Features
    ax7 = fig.add_subplot(gs[2, 0:2])
    plot_encoding_heatmap(combined[0], ax7, "Combined Features")
    
   # ax8 = fig.add_subplot(gs[2, 2])
   # plot_encoding_heatmap(transformer_input[0], ax8, "Transformer Input")
    
    # Add PMT positions visualization
   # ax_pmt = fig.add_axes([0.02, 0.02, 0.2, 0.2])
   # plot_pmt_positions(dataset.pmt_positions, ax_pmt)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_encoding_heatmap(
    data: np.ndarray,
    ax: plt.Axes,
    title: str,
    cmap: str = 'viridis'
) -> None:
    """Plot a heatmap of the encoding data."""
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        cbar=True,
        xticklabels=False,
        yticklabels=False
    )
    ax.set_title(title)

def plot_pmt_positions(
    pmt_positions: List[dict],
    ax: plt.Axes
) -> None:
    """Plot PMT positions in 3D."""
    xs = [p['x'] for p in pmt_positions]
    ys = [p['y'] for p in pmt_positions]
    zs = [p['z'] for p in pmt_positions]
    
    # Create scatter plot
    scatter = ax.scatter(xs, ys, c=zs, cmap='coolwarm')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('PMT Positions')
    plt.colorbar(scatter, ax=ax, label='Z Position')

# Example usage:
if __name__ == "__main__":
    # Initialize model and dataset
    model = ModifiedTransformerModel(
        embedding_dim=8,
        num_heads=4,
        num_layers=3
    )
    
    dataset = PMTDataset(
        h5_filename="pmt_Cosmic_analysis1620_5_threshold.h5",
        pmt_positions=pmt_positions,
        embedding_dim=8
    )
    
    # Plot encodings for first event
    plot_encoded_data(
        model=model,
        dataset=dataset,
        event_idx=0,
        save_path="pmt_encodings.png"
    )

