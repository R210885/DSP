'''def moving_average(data, window_size):
    """
    Calculate the moving average of a given dataset.
    Args:
        data (list): The input time series data.
        window_size (int): The number of data points to consider for the moving average ie the length of array
    Returns:
        list: The moving average values.
    """
    moving_averages = []
    for i in range(len(data)):
        if i < window_size - 1:
            # Calculate the average for the initial data points
            avg = sum(data[:i+1]) / (i+1)
        else:
            # Calculate the moving average
            avg = sum(data[i-window_size+1:i+1]) / window_size
        moving_averages.append(avg)
    return moving_averages

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
window_size = 3

moving_avg_values = moving_average(data, window_size)
print("Moving Average Values:")
for i, avg in enumerate(moving_avg_values):
    print(f"Data Point {i+1}: {avg}")'''
import numpy as np
import matplotlib.pyplot as plt

# Original signal
original_signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Window size for moving average
window_size = 3

# Calculate the moving average
moving_average = np.convolve(original_signal, np.ones(window_size) / window_size, mode='valid')
#np.convolve to calculate the moving average. This function performs convolution, which is a mathematical operation used to compute the moving average efficiently
print(original_signal)
print(moving_average)
# Plot the original and moving average signals
plt.figure(figsize=(10, 6))
plt.stem(np.arange(len(original_signal)), original_signal, linefmt='b-', markerfmt='bo', basefmt='b-', label='Original Signal')
plt.stem(np.arange(len(moving_average)), moving_average, linefmt='r-', markerfmt='ro', basefmt='r-', label='Moving Average')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

