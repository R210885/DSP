import numpy as np
import matplotlib.pyplot as plt

# Define the original signal
signal = np.array([3, 4, 1, 2])

# Specify the desired padding length on each side
left_pad = 3
right_pad = 0

# Zero-pad the signal using numpy.pad
padded_signal = np.pad(signal, (left_pad, right_pad), mode='constant')

# Create the time axis for plotting
time_axis = np.arange(len(padded_signal))

# Plot the original and padded signals
plt.figure(figsize=(10, 6))
plt.stem(np.arange(len(signal)), signal, linefmt='b-', markerfmt='bo', basefmt='b-', use_line_collection=True)
plt.stem(time_axis, padded_signal, linefmt='r-', markerfmt='ro', basefmt='r-', use_line_collection=True)
plt.xlabel('Time Index')
plt.ylabel('Amplitude')
plt.title('Original Signal (Blue) vs. Zero-Padded Signal (Red)')
plt.grid(True)
plt.show()
