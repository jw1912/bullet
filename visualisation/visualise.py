import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# Set the seaborn theme for better aesthetics
sns.set_theme()

# Create the output directory if it doesn't exist
os.makedirs("visualisation/plots", exist_ok=True)

# Define the list of log files
log_files = [
    "checkpoints/optimiser-benchmark-16n-10/log.txt",
    "checkpoints/optimiser-benchmark-8n-10/log.txt",
    "checkpoints/optimiser-benchmark-4n-10/log.txt",
    # Add more log files as needed
]

# Define the colors for each log file plot
colors = ['blue', 'red', 'orange', 'purple', 'cyan', 'green', 'brown']

# Calculate a simple moving average for smoothing
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Plotting the full loss sequences
plt.figure(figsize=(12, 6))

for i, log_file in enumerate(log_files):
    with open(log_file) as f:
        data = f.readlines()

    # Parse the loss values
    loss_sequence = [float(l.split()[1]) for l in data]

    # Smoothing the loss sequence
    window_size = 10
    smoothed_loss = moving_average(loss_sequence, window_size)

    # Plot raw and smoothed loss
    run_name = log_file.split("/")[-2].rstrip("-0123456789")
    plt.plot(loss_sequence, label=f'Raw Loss {run_name}', alpha=0.3, color=colors[i % len(colors)])
    plt.plot(range(window_size - 1, len(smoothed_loss) + window_size - 1), smoothed_loss, label=f'Smoothed Loss {run_name}', color=colors[i % len(colors)], linewidth=2)

plt.title('Training Loss Over Time (Full)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
output_path_full = "visualisation/plots/training_loss_full.png"
plt.savefig(output_path_full)
plt.close()

# Plotting the loss sequences excluding the initial few epochs
# cutoff = 400  # Change this value as needed to adjust how many initial epochs to cut off

plt.figure(figsize=(12, 6))

# determine cutoff guard
guard = 0
for i, log_file in enumerate(log_files):
    with open(log_file) as f:
        data = f.readlines()
    # dynamically determine the cutoff using a heuristic:
    max_tail = max(loss_sequence[len(loss_sequence)//4:])
    min_tail = min(loss_sequence[len(loss_sequence)//4:])
    diff = max_tail - min_tail
    this_guard = min_tail + diff * 2
    guard = max(guard, this_guard)

for i, log_file in enumerate(log_files):
    with open(log_file) as f:
        data = f.readlines()

    # Parse the loss values
    loss_sequence = [float(l.split()[1]) for l in data]

    # Smoothing the loss sequence
    window_size = 200
    smoothed_loss = moving_average(loss_sequence, window_size)

    # find acceptable datapoints for this run
    last_exceeding_instance = next(i for i in reversed(range(len(loss_sequence))) if loss_sequence[i] > guard)
    cutoff = last_exceeding_instance + 1

    # Plot raw and smoothed loss excluding initial epochs
    run_name = log_file.split("/")[-2].rstrip("-0123456789")
    plt.plot(range(cutoff, len(loss_sequence)), loss_sequence[cutoff:], label=f'Raw Loss {run_name}', alpha=0.3, color=colors[i % len(colors)])
    plt.plot(range(cutoff + window_size - 1, len(smoothed_loss) + window_size - 1), smoothed_loss[cutoff:], label=f'Smoothed Loss {run_name}', color=colors[i % len(colors)], linewidth=2)

plt.title(f'Training Loss Over Time (Dynamically Excluding Initial Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
output_path_trimmed = "visualisation/plots/training_loss_trimmed.png"
plt.savefig(output_path_trimmed)
plt.close()

print(f"Full plot saved to {output_path_full}")
print(f"Trimmed plot saved to {output_path_trimmed}")
