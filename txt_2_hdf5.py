import pandas as pd

# Step 1: Read the TXT file into a pandas DataFrame
df = pd.read_csv('pmt_Marley_analysis_5_threshold.txt', delimiter=',')  # Replace 'your_file.txt' with the name of your actual file

# Step 2: Save the DataFrame to an HDF5 file
df.to_hdf('pmt_Cosmic_analysis1620_5_threshold.h5', key='df', mode='w')  # This will create a new HDF5 file called data.h5

print("Conversion complete! HDF5 file saved as 'pmt_Cosmic_analysis1620_5_threshold.h5'")

'''
Structure of File Seems to be the Following

block0_values[:, 0] -> Waveform numbers
block0_values[:, 1] -> PMT numbers
block1_values[:, 0] -> Q values
block1_values[:, 1] -> delta_t values 

'''