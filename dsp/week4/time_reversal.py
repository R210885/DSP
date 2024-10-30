import numpy as np
import matplotlib.pyplot as plt

#def time_reversal(signal):
"""
    Reverses the input signal in time.

    Args:
        signal (numpy array): Input signal.

    Returns:
        reversed_signal (numpy array): Time-reversed signal.
"""
    #reversed_signal = np.flipud(signal)
    #return reversed_signal

# Example usage
sample_rate = 100.0  # Sample rate in Hz
t = np.arange(0, 1, 1/sample_rate)  # Time array from 0 to 1 second
signal = np.sin(2 * np.pi * 10 * t) #+ 0.5 * np.sin(2 * np.pi * 20 * t)  # Example signal
reversed_signal = np.flipud(signal)
print(np.flipud(signal))
print(reversed_signal)
# Plot the original and reversed signals
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(122)
plt.plot(t, reversed_signal)
plt.title('Time-Reversed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

