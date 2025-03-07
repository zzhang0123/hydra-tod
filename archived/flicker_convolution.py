def sqrt_flicker_PSD(size, dt, fknee, alpha, wn=True):
    temp_freqs = np.fft.fftfreq(size, d=dt) # Temperal frequency array
    temp_freqs[0] = np.inf
    weights = (np.abs(fknee / temp_freqs))**alpha
    if wn:
        weights += 1.0
    weights[0] = 0.
    weights = np.sqrt(weights)
    return weights

def flicker_t_kernel(size, dt, fknee, alpha, wn=True):
    coeffs_f = sqrt_flicker_PSD(size, dt, fknee, alpha, wn=wn) # sqrt(noise PSD) coefficient array
    return np.fft.ifft(coeffs_f).real 

def custom_convolve(x, h, begin_xind=0):
    # Get the lengths of the input signals
    len_x = len(x)
    len_h = len(h)
    
    # The length of the output signal
    len_y = len_h
    
    # Initialize the output signal with zeros
    y = np.zeros(len_y)
    
    # Perform the convolution operation
    for i in range(len_y):
        for j in range(len_h):
            y[i] += x[int(begin_xind) + i - j] * h[j]
    
    return y

def sim_flicker_noise(size, dt, fknee, alpha, std, wn=True):

    n=int(size)

    coeffs_f = sqrt_flicker_PSD(n, dt, fknee, alpha, wn=wn) # sqrt(noise PSD) coefficient array
    coeffs_t = np.fft.ifft(coeffs_f).real 
    
    # Generate Gaussian random noise (white)
    aux_n = np.random.normal(0, std, 2*n-1)

    flicker_n = custom_convolve(aux_n, coeffs_t, begin_xind=n-1)

    return flicker_n

def sim_noise(t_kernel, std):
    
    n=int(len(t_kernel))
    # Generate Gaussian random noise (white)
    aux_n = np.random.normal(0, std, 2*n)
    custom_n = custom_convolve(aux_n, t_kernel, begin_xind=n)
    return custom_n



f_n = sim_flicker_noise(1024, 2, 0.01, 2, 1, wn=False)


f_n_DFT = np.fft.rfft(f_n)
f_n_DFT_abs = np.abs(f_n_DFT)
rfreqs = np.fft.rfftfreq(1024)
plt.plot(rfreqs, f_n_DFT_abs)
plt.yscale('log')
plt.xscale('log')

# Estimated the covariance matrix of the flicker noise

n_samples = 100
n=1024
cov_mat = np.zeros((n, n))

time_kernel = flicker_t_kernel(n, 2, 0.01, 2, wn=True)

for i in range(n_samples):
    f_n = sim_noise(time_kernel, 1)
    cov_mat += np.outer(f_n, f_n)

cov_mat /= n_samples

# Estimate the correlation function as mean of each diagonal
corrs = np.zeros(2*n-1)

for i in range(2*n-1):
    corrs[i] = np.mean(cov_mat.diagonal(i-n))

lags = np.arange(1-n, n)
plt.plot(lags, corrs)