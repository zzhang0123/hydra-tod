import numpy as np
import matplotlib.pyplot as plt

# Generate flicker noise
def generate_flicker_noise(N):
    freqs = np.fft.fftfreq(N)
    psd = np.abs(freqs)**(-0.5)  # Power spectral density proportional to 1/f
    phases = np.exp(2j * np.pi * np.random.rand(N//2))
    noise = np.fft.ifft(psd * phases)
    return np.real(noise)

# Compute covariance function
def covariance_function(data, max_lag):
    mean = np.mean(data)
    N = len(data)
    covariances = []
    for tau in range(max_lag):
        cov = np.mean((data[:N-tau] - mean) * (data[tau:] - mean))
        covariances.append(cov)
    return covariances

# Generate flicker noise and compute covariance function
N = 1024
max_lag = 100
flicker_noise = generate_flicker_noise(N)
covariances = covariance_function(flicker_noise, max_lag)

# Plot covariance function
plt.figure(figsize=(10, 6))
plt.plot(covariances)
plt.xlabel('Lag')
plt.ylabel('Covariance')
plt.title('Covariance Function of Flicker Noise')
plt.show()