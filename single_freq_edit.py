#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:43:13 2022

@author: mirfan
"""

#!/usr/bin/env python

import warnings
import numpy as np
from numpy.fft import fft, ifft
from scipy.sparse.linalg import lgmres, LinearOperator

def nrows(x):
    return np.shape(x)[0]

def ncols(x):
    return np.shape(x)[1]

def deconv(signal, kernel):
    result= ifft(fft(signal)/kernel).real
    assert(result.shape==signal.shape)
    return result

def batch_deconv(signal, kernels, signal_length, blocknum):
    assert(isinstance(signal, np.ndarray))

    if blocknum > 1:
        result=np.zeros_like(signal)
        kernels = np.concatenate(kernels)
        nscan = int(signal_length[0]/blocknum)
        for ss in range(0, len(kernels) - 100, nscan):
            result[ss:ss+nscan]=deconv(signal[ss:ss+nscan], fft(kernels[ss:ss+nscan]))  
    else:
        result=np.zeros_like(signal)
        cnt=0
        for (i, (k1, l)) in enumerate(zip(kernels, signal_length)):
            result[cnt:(cnt+l)]=deconv(signal[cnt:cnt+l], k1)
            cnt+=l
    return result

class Solver:
    @classmethod
    def naive(cls, tod, ptr_mat):
        l=nrows(tod)
        noise=np.zeros((l,))
        noise[0]=1.0
        noise = fft(noise)
        return Solver(tod, ptr_mat, noise)

    def __init__(self, tod, ptr_mat, noise_cov):
        self.ptr_mat=[]
        self.noise_cov=[]
        self.tod=[]
        self.tol=1e-4
        self.x=None
        self.signal_length=[]
        self.with_obs(tod, ptr_mat, noise_cov)
        self.iter_max=1000
        self.block_num=1
    
    def with_obs(self, tod, ptr_mat, noise_cov):
        assert(nrows(tod)==nrows(ptr_mat))
        assert(np.ndim(tod)==1)
        assert(np.ndim(ptr_mat)==2)
        assert(np.ndim(noise_cov)==1)
        if self.tod:
            assert(ncols(self.ptr_mat[-1])==ncols(ptr_mat))

        self.tod.append(tod)
        self.noise_cov.append(noise_cov)
        self.ptr_mat.append(ptr_mat)
        self.signal_length.append(nrows(tod))
        return self

    def with_naive_obs(self, tod, ptr_mat):
        l=nrows(tod)
        noise=np.zeros((l,))
        noise[0]=1.0
        noise = fft(noise)
        return self.with_obs(tod, ptr_mat, noise)

    def with_tol(self, tol):
        self.tol=tol
        return self

    def with_iter_max(self, n):
        self.iter_max=n
        return self

    def with_init_x(self, x):
        assert(self.ptr_mat)
        assert(ncols(self.ptr_mat[-1])==nrows(x))
        self.x=x
        return self

    def apply_ptr_mat(self, x):
        result=np.zeros(np.sum([nrows(x) for x in self.ptr_mat]), dtype=self.tod[-1].dtype)
        rcnt=0
        for p in self.ptr_mat:
            result[rcnt:rcnt+nrows(p)]=p.dot(x)
            rcnt+=nrows(p)
        return result

    def apply_ptr_mat_t(self, x):
        result = np.zeros(ncols(self.ptr_mat[0]))
        ccnt=0
        for p in self.ptr_mat:
            result += p.T.dot(x[ccnt: ccnt+nrows(p)])
            ccnt+=nrows(p)
        return result

    def concated_tod(self):
        return np.concatenate(self.tod)

    def calc_b(self, concated_tod):
        return self.apply_ptr_mat_t(batch_deconv(concated_tod, self.noise_cov, \
            self.signal_length, self.block_num))

    def gen_A_func(self):
        A_func=lambda x: self.apply_ptr_mat_t(batch_deconv(self.apply_ptr_mat(x), \
            self.noise_cov, self.signal_length, self.block_num))
        return A_func

    def solve(self):
        return self.solve_with_tod(self.tod)
    
    def solve_with_tod(self, tod, x0=None):
        A_func=self.gen_A_func()
        concated_tod=np.concatenate(tod)
        b=self.calc_b(concated_tod)
        A=LinearOperator((len(b), len(b)), matvec=A_func)
        #solution, result=gmres(A, b, tol=self.tol, maxiter=1000000)
        solution, result=lgmres(A, b, x0=x0, tol=self.tol, maxiter=self.iter_max)
        if result!=0:
            warnings.warn("result={0} is not zero, check the input carefully, the solution can be ill-conditioned".format(result))

        resid=concated_tod-self.apply_ptr_mat(solution)
        resid1=[]
        cnt=0
        for nt in self.signal_length:
            resid1.append(resid[cnt:cnt+nt])
            cnt+=nt
        return solution, resid1
    