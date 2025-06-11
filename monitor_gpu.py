import subprocess
import time
import pandas as pd
import psutil # Ensure you have run: pip install psutil

def get_system_metrics():
    """Executes nvidia-smi and psutil to get comprehensive system metrics."""
    try:
        # --- GPU Metrics via nvidia-smi ---
        # Query for name, utilization, memory, temperature, and power
        command = "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits"
        gpu_result = subprocess.check_output(command, shell=True, encoding='utf-8')
        gpu_lines = gpu_result.strip().split('\n')
        
        # Assuming one GPU for simplicity
        gpu_parts = gpu_lines[0].split(', ')
        
        metrics = {
            'gpu_name': gpu_parts[0],
            'gpu_util_%': int(gpu_parts[1]),
            'gpu_mem_used_mb': int(gpu_parts[2]),
            'gpu_mem_total_mb': int(gpu_parts[3]),
            'gpu_temp_c': int(gpu_parts[4]),
            'gpu_power_w': float(gpu_parts[5])
        }
        
        # --- CPU and RAM Metrics via psutil ---
        metrics['cpu_util_%'] = psutil.cpu_percent()
        ram_info = psutil.virtual_memory()
        metrics['ram_util_%'] = ram_info.percent
        metrics['ram_used_gb'] = ram_info.used / (1024**3)
        metrics['ram_total_gb'] = ram_info.total / (1024**3)

        return metrics
        
    except Exception as e:
        print(f"An error occurred: {e}. Is nvidia-smi in your PATH and psutil installed?")
        return None

def monitor_system(duration_seconds=60, interval_seconds=1):
    """Monitors the entire system for a specified duration."""
    print("--- Starting Full System Monitoring ---")
    print(f"Capturing data for {duration_seconds} seconds...")
    
    all_readings = []
    start_time = time.time()
    
    # Print header
    header = "| {0:<8} | {1:<8} | {2:<8} | {3:<15} | {4:<15} |".format(
        "GPU %", "CPU %", "Temp C", "VRAM (MB)", "RAM (GB)"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    while time.time() - start_time < duration_seconds:
        metrics = get_system_metrics()
        if metrics:
            all_readings.append(metrics)
            # Live printout
            vram_str = f"{metrics['gpu_mem_used_mb']}/{metrics['gpu_mem_total_mb']}"
            ram_str = f"{metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.1f}"
            
            line = "| {0:<8} | {1:<8} | {2:<8} | {3:<15} | {4:<15} |".format(
                metrics['gpu_util_%'],
                metrics['cpu_util_%'],
                metrics['gpu_temp_c'],
                vram_str,
                ram_str
            )
            print(line, end='\r') # Use carriage return to update the line in place
        
        time.sleep(interval_seconds)
        
    print("\n" + "-" * len(header))
    print("--- System Monitoring Finished ---")
    return all_readings

if __name__ == "__main__":
    # Run the monitoring
    system_data = monitor_system(duration_seconds=60, interval_seconds=1)
    
    # Display a summary of the collected data
    if system_data:
        df = pd.DataFrame(system_data)
        print("\n--- Collected System Data Summary ---")
        
        # Select and reorder columns for clarity
        display_cols = ['gpu_util_%', 'cpu_util_%', 'ram_util_%', 'gpu_mem_used_mb', 'gpu_temp_c', 'gpu_power_w']
        print(df[display_cols])
        
        avg_gpu_util = df['gpu_util_%'].mean()
        avg_cpu_util = df['cpu_util_%'].mean()
        print(f"\nAverage GPU Utilization: {avg_gpu_util:.2f}%")
        print(f"Average CPU Utilization: {avg_cpu_util:.2f}%")