import os
from glob import glob
import psutil # You might need to run: pip install psutil
import sys

# --- Define Project Directories ---
# This makes the script runnable from the project root directory
try:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments
    ROOT_DIR = os.getcwd()

PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

# Check if the directory exists
if not os.path.isdir(PROCESSED_DATA_DIR):
    print(f"Error: Processed data directory not found at '{PROCESSED_DATA_DIR}'")
    print("Please make sure you have run Notebook 2 to generate the processed files.")
    sys.exit(1) # Exit the script if the directory is missing

# --- Calculate Dataset Size ---
all_files = glob(os.path.join(PROCESSED_DATA_DIR, "*.npy"))
if not all_files:
    print(f"Error: No '.npy' files found in '{PROCESSED_DATA_DIR}'.")
    print("Please make sure Notebook 2 ran successfully.")
    sys.exit(1)

total_size_bytes = sum(os.path.getsize(f) for f in all_files)
total_size_gb = total_size_bytes / (1024**3)

# --- Get System RAM ---
system_ram_gb = psutil.virtual_memory().total / (1024**3)

print("--- System Memory Analysis ---")
print(f"Total size of processed dataset on disk: {total_size_gb:.2f} GB")
print(f"Total system RAM detected: {system_ram_gb:.2f} GB")

# --- Recommendation ---
# Use a safe threshold (e.g., 75%) to decide if loading into RAM is feasible
can_load_into_ram = total_size_gb < (system_ram_gb * 0.75)
print("\nRecommendation:")
if can_load_into_ram:
    print(f"✅ It looks SAFE to load the entire dataset into RAM.")
    print("   You can now proceed to replace the BrainMRIDataset class in your notebook.")
else:
    print(f"❌ CAUTION: The dataset size ({total_size_gb:.2f} GB) is large compared to your system RAM ({system_ram_gb:.2f} GB).")
    print("   Loading the entire dataset into memory may cause the system to become unresponsive.")
    print("   If you proceed, monitor your system's memory usage closely.")
print("------------------------------")