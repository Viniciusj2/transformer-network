import numpy as np
import tables
import pandas as pd
import numpy as np
import tables
import gc

class DataPreprocessor:
    def __init__(self, smearing_percentage=15):
        self.smearing_percentage = smearing_percentage / 100

    def add_gaussian_smearing(self, values):
        # Apply Gaussian smearing directly to values in place (to reduce memory usage)
        sigma = self.smearing_percentage * values
        return values + np.random.normal(0, sigma, size=values.shape)

    def normalize(self, values):
        # Normalize values in place (also avoids creating new arrays)
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val)

    def preprocess(self, q_values, delta_t_values):
        # Perform preprocessing in-place (no need to store extra arrays)
        q_values = self.add_gaussian_smearing(q_values)
        delta_t_values = self.add_gaussian_smearing(delta_t_values)

        # Normalize in place
        q_values = self.normalize(q_values)
        delta_t_values = self.normalize(delta_t_values)

        #print(f"Preprocessed Q: {q_values[:10]}, delta_t: {delta_t_values[:10]}")
        return q_values, delta_t_values

class DataLoader:
    def __init__(self, filepath, batch_size, preprocessor):
        self.filepath = filepath
        self.batch_size = batch_size
        self.preprocessor = preprocessor

    def load_data_batch(self, batch_start, batch_end):
        # Open HDF5 file and load the slice for the current batch
        with tables.open_file(self.filepath, mode='r') as file:
            block0_values = file.root.df.block0_values[batch_start:batch_end]
            block1_values = file.root.df.block1_values[batch_start:batch_end]
        
        # Return a dictionary of relevant data for this batch
        return {
            "waveforms": block0_values[:, 0],
            "pmt_numbers": block0_values[:, 1].astype(int),
            "q_values": block1_values[:, 0],
            "delta_t_values": block1_values[:, 1],
        }

    def __iter__(self):
        with tables.open_file(self.filepath, mode='r') as file:
            total_entries = len(file.root.df.block0_values)
        
        # Iterate through the dataset in batches
        for batch_start in range(0, total_entries, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_entries)

            # Load the current batch of data
            batch_data = self.load_data_batch(batch_start, batch_end)

            # Preprocess the batch data
            q_preprocessed, dt_preprocessed = self.preprocessor.preprocess(batch_data["q_values"], batch_data["delta_t_values"])

            # Yield the preprocessed batch and explicitly free memory for the current batch
            yield {
                "waveforms": batch_data["waveforms"],
                "pmt_numbers": batch_data["pmt_numbers"],
                "q_values": q_preprocessed,
                "delta_t_values": dt_preprocessed,
            }

            # Free memory for this batch and trigger garbage collection
            del batch_data
            gc.collect()
