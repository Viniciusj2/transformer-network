import h5py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple, Dict, List
import tables
from torch.utils.data import DataLoader

class DataPreprocessor:
    def __init__(self, min_smearing=5, max_smearing=15):
        self.min_smearing = min_smearing / 100
        self.max_smearing = max_smearing / 100
        
    def process_batch(self, charges, deltas):
        """Process a batch of data, returning both original and smeared values"""
        # Store original values
        original_charges = charges.copy()
        original_deltas = deltas.copy()
        
        # Apply smearing
        smeared_charges = self.add_gaussian_smearing(charges)
        smeared_deltas = self.add_gaussian_smearing(deltas)
        
        return {
            'original_q_values': original_charges,
            'original_delta_t_values': original_deltas,
            'q_values': smeared_charges,
            'delta_t_values': smeared_deltas
        }

    def add_gaussian_smearing(self, values):
        smearing_percentage = np.random.uniform(self.min_smearing, self.max_smearing)
        sigma = smearing_percentage * values
        return values + np.random.normal(0, sigma, size=values.shape)

class ImprovedDistributionAnalyzer:
    def __init__(self, h5_filename: str, min_smearing: float = 5, max_smearing: float = 15):
        self.h5_filename = h5_filename
        self.min_smearing = min_smearing
        self.max_smearing = max_smearing
        self.preprocessor = DataPreprocessor(min_smearing=min_smearing, max_smearing=max_smearing)
        
    def collect_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collect original and smeared values directly from the H5 file"""
        original_q_values = []
        original_delta_t_values = []
        smeared_q_values = []
        smeared_delta_t_values = []
        
        with tables.open_file(self.h5_filename, mode='r') as file:
            block1_values = file.root.df.block1_values[:]
            charges = block1_values[:, 0]
            deltas = block1_values[:, 1]
            
            # Process in batches to manage memory
            batch_size = 1000
            for i in range(0, len(charges), batch_size):
                batch_charges = charges[i:i+batch_size]
                batch_deltas = deltas[i:i+batch_size]
                
                processed = self.preprocessor.process_batch(batch_charges, batch_deltas)
                
                original_q_values.extend(processed['original_q_values'])
                original_delta_t_values.extend(processed['original_delta_t_values'])
                smeared_q_values.extend(processed['q_values'])
                smeared_delta_t_values.extend(processed['delta_t_values'])
        
        return (np.array(original_q_values), np.array(original_delta_t_values),
                np.array(smeared_q_values), np.array(smeared_delta_t_values))

    # Rest of the ImprovedDistributionAnalyzer class remains the same...
    def calculate_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistics for a dataset"""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'median': np.median(data),
            'min': np.min(data),
            'max': np.max(data)
        }

    def plot_distributions(self, save_dir: str = 'plots'):
        """Create and save enhanced distribution plots"""
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Collect data
        orig_q, orig_dt, smeared_q, smeared_dt = self.collect_data()
        
        # Set up the plots
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # Create a 2x2 grid of subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Define colors for consistency
        original_color = '#1f77b4'  # Blue
        smeared_color = '#ff7f0e'   # Orange
        
        # Plot 1: Q Distribution Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(orig_q, bins=500, alpha=0.6, color=original_color, label='Original')
        ax1.hist(smeared_q, bins=500, alpha=0.6, color=smeared_color, label='Smeared')
        ax1.set_xlim(0, 100)
        ax1.set_title('Q Distribution Comparison', fontsize=12, pad=20)
        ax1.set_xlabel('Q Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: ΔT Distribution Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(orig_dt, bins=10, alpha=0.6, color=original_color, label='Original')
        ax2.hist(smeared_dt, bins=10, alpha=0.6, color=smeared_color, label='Smeared')
        ax2.set_xlim(0, 10)
        ax2.set_title('ΔT Distribution Comparison', fontsize=12, pad=20)
        ax2.set_xlabel('ΔT Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add a main title
        fig.suptitle(f'Distribution Analysis (Smearing Range: {self.min_smearing}% - {self.max_smearing}%)',
                    fontsize=16, y=0.95)
        
        # Save the plot
        plt.savefig(os.path.join(save_dir, 'distribution_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate and save statistics
        stats_summary = {
            'Q_original': self.calculate_statistics(orig_q),
            'Q_smeared': self.calculate_statistics(smeared_q),
            'DT_original': self.calculate_statistics(orig_dt),
            'DT_smeared': self.calculate_statistics(smeared_dt)
        }
        
        return stats_summary

# Example usage
if __name__ == "__main__":
    analyzer = ImprovedDistributionAnalyzer(
        h5_filename="pmt_Cosmic_analysis1620_5_threshold.h5",
        min_smearing=5,  # 5%
        max_smearing=15  # 15%
    )
    stats = analyzer.plot_distributions(save_dir='distribution_plots')
    print("\nAnalysis complete! Check the 'distribution_plots' directory for results.")

