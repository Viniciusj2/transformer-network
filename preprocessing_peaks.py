import sys
import os
import numpy as np
import ROOT as rt
from scipy.signal import find_peaks
from math import sqrt
from ROOT import cenns as cenns
import time

# Configuration parameters
sampling_rate_MHz = 1.25
time_per_sample_us = 1 / sampling_rate_MHz
t0_ns = -100.0
t0_us = t0_ns / 1000.0
window_pretrig_samples = 1
THRESHOLD = 5  # Trigger threshold

def analyze_pmt_waveforms(rootfile_path, output_file):
    """
    Analyzes individual PMT waveforms and saves Q, delta t, and PMT information
    Uses dynamic window that ends when signal drops below threshold
    
    Parameters:
    rootfile_path (str): Path to the ROOT file
    output_file (str): Path to save the output txt file
    """
    print(f"\nStarting analysis of: {rootfile_path}")
    print(f"Output will be saved to: {output_file}\n")
    
    start_time = time.time()
    rootfile = rt.TFile(rootfile_path)
    CENNS = rootfile.Get("CENNS")
    nentries = CENNS.GetEntries()
    
    print(f"Total entries to process: {nentries}")
    print("Beginning analysis...\n")
    
    # Initialize counters for per-PMT statistics
    pulses_per_pmt = {}  # Dictionary to store pulse counts per PMT
    
    # Open output file
    with open(output_file, 'w') as f:
        # Write header
        f.write("Entry,PMT,Q,delta_t\n")
        
        total_pulses = 0
        # Process entries
        for ientry in range(nentries):
            if ientry % 100 == 0:
                elapsed_time = time.time() - start_time
                progress = (ientry / nentries) * 100
                est_total_time = elapsed_time / (ientry + 1) * nentries if ientry > 0 else 0
                est_remaining = est_total_time - elapsed_time
                
                print(f"Entry [{ientry}/{nentries}] ({progress:.1f}%)")
                print(f"Elapsed time: {elapsed_time:.1f}s")
                print(f"Estimated time remaining: {est_remaining:.1f}s")
                print(f"Total pulses found so far: {total_pulses}")
                print("Pulses per PMT:")
                for pmt, count in sorted(pulses_per_pmt.items()):
                    print(f"  PMT {pmt}: {count} pulses")
                print("-" * 50 + "\n")

            CENNS.GetEntry(ientry)
            
            # Get DAQ data
            daq_branch_v = CENNS.DAQ
            daq_data = {}
            for idaq in range(daq_branch_v.size()):
                daq = daq_branch_v.at(idaq)
                daqname = daq.name
                daqdict = cenns.pytools.CENNSPyTools.to_numpy(daq)
                daq_data[daqname] = daqdict

            DAQ = "caen_vx2740"
            wfms = daq_data[DAQ]["waveforms"]
            
            entry_pulses = {pmt: 0 for pmt in range(wfms.shape[0])}  # Track pulses per PMT in this entry
            
            # Process each PMT separately
            for pmt_idx in range(wfms.shape[0]):  # Iterate over PMTs
                waveform = wfms[pmt_idx].copy()  # Make a copy to avoid modifying original
                
                # Skip if waveform is empty
                if np.sum(waveform) == 0.0:
                    continue
                
                # Find all threshold crossings
                above_threshold = waveform > THRESHOLD
                threshold_crossings = np.diff(above_threshold.astype(int))
                rising_edges = np.where(threshold_crossings == 1)[0]
                falling_edges = np.where(threshold_crossings == -1)[0]
                
                # Handle case where waveform starts above threshold
                if above_threshold[0]:
                    rising_edges = np.insert(rising_edges, 0, 0)
                
                # Handle case where waveform ends above threshold
                if above_threshold[-1]:
                    falling_edges = np.append(falling_edges, len(waveform) - 1)
                
                # Process each pulse
                for start_idx, end_idx in zip(rising_edges, falling_edges):
                    # Adjust window to include pre-trigger samples
                    window_start = max(start_idx - window_pretrig_samples, 0)
                    window_end = end_idx + 1  # Add 1 to include the last sample
                    
                    # Calculate times
                    start_time_us = window_start * time_per_sample_us + t0_us
                    end_time_us = window_end * time_per_sample_us + t0_us
                    delta_t = end_time_us - start_time_us
                    
                    # Calculate integral (Q)
                    Q = np.sum(waveform[window_start:window_end])
                    
                    # Write to file
                    f.write(f"{ientry},{pmt_idx},{Q:.2f},{delta_t:.2f}\n")
                    
                    # Update counters
                    total_pulses += 1
                    entry_pulses[pmt_idx] += 1
                    pulses_per_pmt[pmt_idx] = pulses_per_pmt.get(pmt_idx, 0) + 1
            
            # Print entry results if pulses were found (and not already printed in progress update)
            if sum(entry_pulses.values()) > 0 and ientry % 100 != 0:
                print(f"Entry {ientry} pulses:")
                for pmt, count in sorted(entry_pulses.items()):
                    if count > 0:
                        print(f"  PMT {pmt}: {count} pulses")

    total_time = time.time() - start_time
    print("\nAnalysis complete!")
    print("-" * 50)
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Total entries processed: {nentries}")
    print("\nFinal pulse statistics:")
    print(f"Total pulses across all PMTs: {total_pulses}")
    print("\nPulses per PMT:")
    for pmt, count in sorted(pulses_per_pmt.items()):
        print(f"  PMT {pmt}: {count} pulses ({count/nentries:.2f} pulses/entry)")
    print(f"\nResults saved to: {output_file}")
    
    rootfile.Close()

if __name__ == "__main__":
    # Example usage
    rootfile_path = "/home/vdasil01/Research/Coherent/g4-coh-ar-750/NeutrinoPosterFiles/Cosmic_100_19999.root" 
    output_file = "pmt_test_cosmic_analysis_6_threshold.txt"
    analyze_pmt_waveforms(rootfile_path, output_file)
