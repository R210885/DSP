import numpy as np
import matplotlib.pyplot as plt

def dtft(x, omega):
    """
    Computes the Discrete-Time Fourier Transform (DTFT) of a signal x[n].

    Args:
        x: The input signal.
        omega: The angular frequency.

    Returns:
        The DTFT of x at the given frequency.
    """
    N = len(x)
    X = np.sum(x * np.exp(-1j * omega * np.arange(N)))
    return X
# Define a signal
x = [1, 2, 3, 4, 5]
# Define a range of frequencies
omega = np.linspace(-np.pi, np.pi, 100)
# Compute the DTFT of x
X = [dtft(x, w) for w in omega]
# Define a frequency shift
omega0 = np.pi / 2
# Shift the signal in the time domain
x_shifted = x * np.exp(1j * omega0 * np.arange(len(x)))
# Compute the DTFT of the shifted signal
X_shifted = [dtft(x_shifted, w) for w in omega]

# Plot the magnitude and phase spectra
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(omega, np.abs(X), label='|X(ω)|')
plt.plot(omega, np.abs(X_shifted), label='|X(ω-ω₀)|')
plt.xlabel('ω')
plt.ylabel('Magnitude')
plt.legend()
plt.title('Magnitude Spectrum')

plt.subplot(2, 1, 2)
plt.plot(omega, np.angle(X), label='∠X(ω)')
plt.plot(omega, np.angle(X_shifted), label='∠X(ω-ω₀)')
plt.xlabel('ω')
plt.ylabel('Phase')
plt.legend()
plt.title('Phase Spectrum')

plt.tight_layout()
plt.show()
