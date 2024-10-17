import numpy as np
from numpy.polynomial.polynomial import polyval2d
from numpy.polynomial.chebyshev import Chebyshev

Chebyshev(coeffs, domain=[t_i, t_f])

class polynomialGain:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __call__(self, time, freq):
        T, F = np.meshgrid(time, freq)
        return polyval2d(T, F, self.coeffs)
    
def chebyshevCoeffs2D(deg_x, deg_y):
    # Generate Chebyshev polynomial coefficients for given degrees
    cheb_poly_x = Chebyshev.basis(deg_x)
    cheb_poly_y = Chebyshev.basis(deg_y)

    # Convert to Polynomial basis to get coefficients
    coefficients_x = cheb_poly_x.convert(kind=np.polynomial.Polynomial).coef
    coefficients_y = cheb_poly_y.convert(kind=np.polynomial.Polynomial).coef

    return np.outer(coefficients_x, coefficients_y)

def legendreCoeffs2D(deg_x, deg_y):
    # Generate Legendre polynomial coefficients for given degrees
    leg_poly_x = np.polynomial.legendre.Legendre.basis(deg_x)
    leg_poly_y = np.polynomial.legendre.Legendre.basis(deg_y)

    # Convert to Polynomial basis to get coefficients
    coefficients_x = leg_poly_x.convert(kind=np.polynomial.Polynomial).coef
    coefficients_y = leg_poly_y.convert(kind=np.polynomial.Polynomial).coef

    return np.outer(coefficients_x, coefficients_y)

# Evaluate correlation function of data
def correlation_function(data, max_lag):
    mean = np.mean(data)
    N = len(data)
    covariances = []
    for tau in range(max_lag):
        cov = np.mean((data[:N-tau] - mean) * (data[tau:] - mean))
        covariances.append(cov)
    return covariances