import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs_original = 8000  # Original sampling rate
fs_new = 1000  # New sampling rate
frequency = 200  # Signal frequency
duration = 0.5  # Signal duration

# Generate original signal
num_samp=int(duration*fs_original)
t=np.arange(num_samp)/fs_original
#t = np.arange(0, duration, 1/fs_original)
signal = np.sin(2 * np.pi * frequency * t)

# Resample signal
N_new = int(duration * fs_new)
t_resampled = np.linspace(0, duration, N_new)
signal_resampled = np.interp(t_resampled, t, signal)

# Plot original and resampled signals
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.plot(t, signal)
plt.title('Original Signal (8000 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(122)
plt.plot(t_resampled, signal_resampled)
plt.title('Resampled Signal (1000 Hz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
