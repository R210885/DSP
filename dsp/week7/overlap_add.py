import numpy as np
import matplotlib.pyplot as plt

def overlap_add(x, h):
    """
    Performs overlap-add convolution without using FFT.

    Args:
        x: Input signal.
        h: Impulse response.

    Returns:
        y: Output signal.
    """

    M = len(h)  # Filter length
    N = len(x)  # Signal length

    # Determine block size and overlap
    L = M  # Block size
    R = L // 2  # Overlap

    # Initialize output signal
    y = np.zeros(N + M - 1)

    # Process blocks
    for n in range(0, N, R):
        # Extract block from input signal
        x_block = x[n:n+L]

        # Zero-pad the block
        x_padded = np.pad(x_block, (0, M-L), 'constant', constant_values=(0, 0))

        # Linear convolution of block with filter
        y_block = np.convolve(x_padded, h)

        # Overlap-add the block to the output
        y[n:n+L+M-1] += y_block

    return y

# Example usage
x = np.array([1,2,3,1,2,3,1,2,3,1,2,3])
h = np.array([1, -1,1])

y = overlap_add(x, h)
print(y)

# Plot the signals
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.stem(x)
plt.title('Input Signal')
plt.xlabel('n')
plt.ylabel('x[n]')

plt.subplot(3, 1, 2)
plt.stem(h)
plt.title('Impulse Response')
plt.xlabel('n')
plt.ylabel('h[n]')

plt.subplot(3, 1, 3)
plt.stem(y)
plt.title('Output Signal (Overlap-Add)')
plt.xlabel('n')
plt.ylabel('y[n]')

#plt.tight_layout()
plt.show()
