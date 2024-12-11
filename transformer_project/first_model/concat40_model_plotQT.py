import torch
import numpy as np
import matplotlib.pyplot as plt
import tables
import os

from torch.utils.data import DataLoader
from typing import List, Dict

# Import your custom classes
from concat40_model import (
    ModifiedTransformerModel, 
    create_dataloader, 
    pmt_positions,
    DataPreprocessor
)

class ModelAnalyzer:
    def __init__(
        self, 
        model_path: str, 
        h5_filename: str, 
        embedding_dim: int = 8, 
        num_heads: int = 4, 
        num_layers: int = 3
    ):
        """
        Initialize the model analyzer with pre-trained model and dataset
        
        Args:
            model_path (str): Path to the saved model weights
            h5_filename (str): Path to the H5 data file
            embedding_dim (int): Dimension of embeddings
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
        """
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = ModifiedTransformerModel(
            embedding_dim=embedding_dim, 
            num_heads=num_heads, 
            num_layers=num_layers
        ).to(self.device)
        
        # Load pre-trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Create data loader
        self.dataloader = create_dataloader(
            h5_filename, 
            pmt_positions, 
            embedding_dim=embedding_dim
        )
        
        # Initialize preprocessor for comparison
        self.preprocessor = DataPreprocessor()

    def analyze_predictions(self):
        """
        Analyze model predictions across the dataset
        
        Returns:
            Dict containing various prediction statistics and visualizations
        """
        all_true_q = []
        all_pred_q = []
        all_true_dt = []
        all_pred_dt = []

        with torch.no_grad():
            for batch in self.dataloader:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Extract original values before smearing
                true_q = batch[:, :, 0:1].cpu().numpy()
                true_dt = batch[:, :, self.dataloader.dataset.embedding_dim:self.dataloader.dataset.embedding_dim+1].cpu().numpy()
                
                # Make predictions
                predictions = self.model(batch)
                
                # Convert predictions to numpy
                pred_q = predictions[:, :, 0:1].cpu().numpy()
                pred_dt = predictions[:, :, 1:2].cpu().numpy()
                
                # De-normalize predictions
                true_q = self._denormalize(true_q)
                true_dt = self._denormalize(true_dt)
                pred_q = self._denormalize(pred_q)
                pred_dt = self._denormalize(pred_dt)
                
                # Collect results
                all_true_q.append(true_q)
                all_pred_q.append(pred_q)
                all_true_dt.append(true_dt)
                all_pred_dt.append(pred_dt)

        # Concatenate results
        all_true_q = np.concatenate(all_true_q)
        all_pred_q = np.concatenate(all_pred_q)
        all_true_dt = np.concatenate(all_true_dt)
        all_pred_dt = np.concatenate(all_pred_dt)

        return {
            'true_q': all_true_q,
            'pred_q': all_pred_q,
            'true_dt': all_true_dt,
            'pred_dt': all_pred_dt
        }

    def _denormalize(self, normalized_values):
        """
        Denormalize values using the same method as in the preprocessor
        
        Args:
            normalized_values (np.ndarray): Normalized input values
        
        Returns:
            np.ndarray: Denormalized values
        """
        # Use min-max scaling reversal
        min_val = 0
        max_val = 1
        return normalized_values * (max_val - min_val) + min_val

    def plot_prediction_distributions(self, save_dir: str = 'model_analysis_plots'):
        """
        Create comprehensive plots to visualize model predictions
        
        Args:
            save_dir (str): Directory to save generated plots
        """
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Analyze predictions
        results = self.analyze_predictions()
        
        # Set up plotting
        plt.style.use('seaborn')
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Model Prediction Analysis', fontsize=16)
        
        # Q Value Scatter Plot
        axs[0, 0].scatter(results['true_q'], results['pred_q'], alpha=0.5)
        axs[0, 0].plot([results['true_q'].min(), results['true_q'].max()], 
                       [results['true_q'].min(), results['true_q'].max()], 
                       'r--', label='Ideal Prediction')
        axs[0, 0].set_title('Q Value: True vs Predicted')
        axs[0, 0].set_xlabel('True Q Values')
        axs[0, 0].set_ylabel('Predicted Q Values')
        axs[0, 0].legend()
        
        # ΔT Value Scatter Plot
        axs[0, 1].scatter(results['true_dt'], results['pred_dt'], alpha=0.5)
        axs[0, 1].plot([results['true_dt'].min(), results['true_dt'].max()], 
                       [results['true_dt'].min(), results['true_dt'].max()], 
                       'r--', label='Ideal Prediction')
        axs[0, 1].set_title('ΔT Value: True vs Predicted')
        axs[0, 1].set_xlabel('True ΔT Values')
        axs[0, 1].set_ylabel('Predicted ΔT Values')
        axs[0, 1].legend()
        
        # Q Value Error Distribution
        q_errors = results['pred_q'] - results['true_q']
        axs[1, 0].hist(q_errors, bins=50)
        axs[1, 0].set_title('Q Value Prediction Errors')
        axs[1, 0].set_xlabel('Prediction Error')
        axs[1, 0].set_ylabel('Frequency')
        
        # ΔT Value Error Distribution
        dt_errors = results['pred_dt'] - results['true_dt']
        axs[1, 1].hist(dt_errors, bins=50)
        axs[1, 1].set_title('ΔT Value Prediction Errors')
        axs[1, 1].set_xlabel('Prediction Error')
        axs[1, 1].set_ylabel('Frequency')
        
        # Save and close plot
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_analysis.png'), dpi=300)
        plt.close()
        
        # Calculate and print summary statistics
        summary_stats = {
            'Q Values': {
                'True Mean': np.mean(results['true_q']),
                'Pred Mean': np.mean(results['pred_q']),
                'True Std': np.std(results['true_q']),
                'Pred Std': np.std(results['pred_q']),
                'Mean Abs Error': np.mean(np.abs(results['true_q'] - results['pred_q']))
            },
            'ΔT Values': {
                'True Mean': np.mean(results['true_dt']),
                'Pred Mean': np.mean(results['pred_dt']),
                'True Std': np.std(results['true_dt']),
                'Pred Std': np.std(results['pred_dt']),
                'Mean Abs Error': np.mean(np.abs(results['true_dt'] - results['pred_dt']))
            }
        }
        
        # Save summary statistics
        import json
        with open(os.path.join(save_dir, 'prediction_summary.json'), 'w') as f:
            json.dump(summary_stats, f, indent=4)
        
        return summary_stats

def main():
    # Example usage
    analyzer = ModelAnalyzer(
        model_path='/home/vdasil01/transformer-network/transformer_project/checkpoints/best_model.pt',
        h5_filename='pmt_Cosmic_analysis1620_5_threshold.h5'
    )
    
    # Run analysis and generate plots
    summary = analyzer.plot_prediction_distributions()
    print("Analysis complete. Check the 'model_analysis_plots' directory for results.")

if __name__ == "__main__":
    main()