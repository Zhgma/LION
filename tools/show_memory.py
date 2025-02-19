import time
import subprocess
import matplotlib.pyplot as plt
from collections import deque

def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    memory_used = result.stdout.decode('utf-8').strip().split('\n')
    memory_used = [int(x) for x in memory_used]
    return sum(memory_used)  # Sum memory usage of all GPUs

def plot_memory_usage(memory_usage, save_path=None):
    plt.clf()
    plt.plot(memory_usage)
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('GPU Memory Usage Over Time')
    if save_path:
        plt.savefig(save_path)
    plt.pause(0.1)

def main():
    memory_usage = deque(maxlen=600)  # Store up to 600 seconds (10 minutes) of data
    start_time = time.time()
    
    plt.ion()  # Turn on interactive mode for live updating
    try:
        error_count = 0
        while True:
            memory_usage.append(get_gpu_memory())
            if len(memory_usage) % 10 == 0:  # Update plot every 10 seconds
                plot_memory_usage(memory_usage, save_path='../output/cfgs/lion_models/cross_modal/cross_modal/output.png')
                if max(list(memory_usage)[-10:]) < 1000 and error_count < 5:  # Convert deque to list for slicing
                    plot_memory_usage(memory_usage, save_path=f'../output/cfgs/lion_models/cross_modal/cross_modal/output_err{error_count}.png')
                    error_count += 1
            time.sleep(1)  # Collect data every second
    except KeyboardInterrupt:
        end_time = time.time()
        print(f"Monitoring stopped after {end_time - start_time:.2f} seconds")
        plt.ioff()
        plot_memory_usage(memory_usage, save_path='../output/cfgs/lion_models/cross_modal/cross_modal/output.png')
        plt.show()

if __name__ == "__main__":
    main()