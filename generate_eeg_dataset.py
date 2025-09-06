import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
num_channels = 8
sampling_rate = 256
duration = 30  # seconds
num_samples_per_class = 100
output_dir = '/mnt/data/eeg_synthetic_dataset'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Time vector
t = np.arange(0, duration, 1/sampling_rate)

def generate_brainwave(f_low, f_high):
    """Generate a brainwave signal in a given frequency band."""
    freq = np.random.uniform(f_low, f_high)
    return np.sin(2 * np.pi * freq * t)

def add_noise_and_artifacts(signal, noise_level=0.5, num_spikes=5):
    """Add white noise and blink-like spike artifacts to a signal."""
    # White noise
    noisy_signal = signal + np.random.normal(0, noise_level, signal.shape)
    
    # Blink-like spikes
    for _ in range(num_spikes):
        spike_center = np.random.randint(0, len(t))
        spike_width = np.random.randint(5, 20)
        spike = np.exp(-0.5 * ((np.arange(len(t)) - spike_center) / spike_width)**2)
        noisy_signal += spike * np.random.uniform(1, 2)
    
    return noisy_signal

def save_sample(data, label, idx):
    """Save a single sample as CSV."""
    df = pd.DataFrame(data.T, index=t, columns=[f'ch{i}' for i in range(num_channels)])
    filename = f"{label}_{idx:03d}.csv"
    df.to_csv(os.path.join(output_dir, filename))

# Generate and save samples
for label, (f_low, f_high) in [('alpha', (8, 12)), ('beta', (13, 30))]:
    for i in range(num_samples_per_class):
        # Multi-channel signal
        signals = np.array([add_noise_and_artifacts(generate_brainwave(f_low, f_high)) 
                            for _ in range(num_channels)])
        save_sample(signals, label, i+1)

print(f"Dataset saved to: {output_dir}")

# Visualization: One alpha and one beta sample, first channel
alpha_sample = pd.read_csv(os.path.join(output_dir, 'alpha_001.csv'), index_col=0)
beta_sample = pd.read_csv(os.path.join(output_dir, 'beta_001.csv'), index_col=0)

# Plot Alpha
plt.figure()
plt.plot(alpha_sample.index, alpha_sample['ch0'])
plt.title('Alpha Sample (Channel 1)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot Beta
plt.figure()
plt.plot(beta_sample.index, beta_sample['ch0'])
plt.title('Beta Sample (Channel 1)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
