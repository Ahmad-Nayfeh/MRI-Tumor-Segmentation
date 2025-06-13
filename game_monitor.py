import time
import subprocess
import psutil
import pandas as pd
import os

def get_gpu_metrics():
    """Fetches key GPU metrics using nvidia-smi."""
    try:
        command = "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits"
        gpu_result = subprocess.check_output(command, shell=True, encoding='utf-8')
        gpu_parts = gpu_result.strip().split(', ')
        return {
            'GPU_Util_%': int(gpu_parts[0]),
            'VRAM_Used_MB': int(gpu_parts[1]),
            'GPU_Temp_C': int(gpu_parts[2]),
            'GPU_Power_W': float(gpu_parts[3])
        }
    except Exception:
        # Return default values if nvidia-smi fails
        return {
            'GPU_Util_%': 0, 'VRAM_Used_MB': 0,
            'GPU_Temp_C': 0, 'GPU_Power_W': 0.0
        }

def monitor_game_performance(interval=1):
    """Monitors and logs system performance until stopped by the user."""
    print("--- Starting Advanced Game Performance Monitor ---")
    print("Play your game and trigger the stutter. Press Ctrl+C to stop logging and see the results.")
    
    all_readings = []
    
    try:
        while True:
            # --- Gather All Metrics ---
            timestamp = time.strftime('%H:%M:%S')
            gpu_metrics = get_gpu_metrics()
            cpu_overall_util = psutil.cpu_percent()
            cpu_per_core_util = psutil.cpu_percent(percpu=True)
            ram_info = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()

            # --- Store Metrics in a Dictionary ---
            current_reading = {
                'Time': timestamp,
                'GPU_Util_%': gpu_metrics['GPU_Util_%'],
                'CPU_Util_%': cpu_overall_util,
                'RAM_Util_%': ram_info.percent,
                'Disk_Read_MB': disk_io.read_bytes / (1024**2),
                'Disk_Write_MB': disk_io.write_bytes / (1024**2),
                'GPU_Temp_C': gpu_metrics['GPU_Temp_C']
            }
            # Add per-core CPU utilization
            for i, core_util in enumerate(cpu_per_core_util):
                current_reading[f'CPU_{i}_%'] = core_util
            
            all_readings.append(current_reading)
            
            # Simple live indicator so you know it's running
            print(f"Logging... GPU Util: {gpu_metrics['GPU_Util_%']}% | CPU Util: {cpu_overall_util}%", end='\r')
            
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n--- Monitor stopped. Generating report... ---")
        if not all_readings:
            print("No data was collected.")
            return

        # Create a pandas DataFrame from the collected data
        df = pd.DataFrame(all_readings)

        # Calculate deltas for disk I/O to see per-second changes
        df['Disk_Read_MB/s'] = df['Disk_Read_MB'].diff().fillna(0)
        df['Disk_Write_MB/s'] = df['Disk_Write_MB'].diff().fillna(0)
        
        # Select and reorder columns for the final display
        cpu_core_cols = [f'CPU_{i}_%' for i in range(len(cpu_per_core_util))]
        display_cols = ['Time', 'GPU_Util_%', 'CPU_Util_%'] + cpu_core_cols + ['RAM_Util_%', 'Disk_Read_MB/s', 'Disk_Write_MB/s', 'GPU_Temp_C']
        
        # Ensure all columns exist before trying to display them
        final_cols = [col for col in display_cols if col in df.columns]

        # Configure pandas to display all rows and columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)

        print(df[final_cols])
        print("\nReport finished. Look for spikes in CPU core usage when you moved your mouse.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    monitor_game_performance(interval=1) # Log data every 1 second