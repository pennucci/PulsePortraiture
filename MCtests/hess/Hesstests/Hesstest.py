#!/usr/bin/env python

import pptoas as pt
import ppgauss as pg
from pplib import *

datafile = "GUPPI_M28A_110702_0001.B0450_pcm_noRM_T"
modelfile = "M28A_L.model"
title = "GUPPI_M28A_110702_0001.B0450_pcm_noRM_T; 512x512, 2.5hr"
#datafile = "example-1.fits"
#modelfile = "example-fit.gmodel"
#title = "example-1.fits; 256x256"

dp = pg.DataPortrait(datafile)

data = dp.portx
P = dp.Ps[0]
name, ngauss, model = read_model(modelfile, dp.phases, dp.freqsxs[0], quiet=True)
P = dp.Ps[0]
freqs = dp.freqsxs[0]
init_params = np.array([0.192692692693, 119.893879])    #M28A
#init_params = np.array([0.125625625626, 34.56789])      #example
channel_SNRs = data.std(axis=1) / get_noise(data, chans=True)
nu_fit = guess_fit_freq(freqs, channel_SNRs)
hess = []

dFFT = fft.rfft(data, axis=1)
mFFT = fft.rfft(model, axis=1)
errs = np.real(dFFT[:, -len(dFFT[0])/4:]).std(axis=1)
d = np.real(np.sum(np.transpose(errs**-2.0 * np.transpose(dFFT *
    np.conj(dFFT)))))
p_n = np.real(np.sum(mFFT * np.conj(mFFT), axis=1))
minimize = opt.minimize
method = 'TNC'
bounds = [(None, None), (None, None)]

tryfreqs = freqs
other_args = (mFFT, p_n, dFFT, errs, P, freqs, nu_fit)
results = minimize(fit_portrait_function, init_params, args=other_args,
        method=method, jac=fit_portrait_function_deriv, bounds=bounds,
        options={'disp':False})
phi = results.x[0]
DM = results.x[1]
for freq in tryfreqs:
    nu_ref = freq
    phi_prime = phase_transform(phi, DM, nu_fit, nu_ref, P)
    hessian, nu = fit_portrait_function_2deriv(np.array([phi_prime, DM]),
        mFFT, p_n, dFFT, errs, P, freqs, nu_ref, False)
    param_errs = list(pow(hessian[:2], -0.5))
    DoF = len(data.ravel()) - (len(freqs) + 2)
    hess.append(hessian[2])

nu_ref = nu_fit
other_args = (mFFT, p_n, dFFT, errs, P, freqs, nu_ref)
results = minimize(fit_portrait_function, init_params, args=other_args,
        method=method, jac=fit_portrait_function_deriv, bounds=bounds,
        options={'disp':False})
phi = results.x[0]
DM = results.x[1]
hessian, nu_zero = fit_portrait_function_2deriv(np.array([phi, DM]),
    mFFT, p_n, dFFT, errs, P, freqs, nu_ref, True)
print nu_zero, hessian[2]
hess = np.array(hess)
print abs(hess).min()

plt.plot(tryfreqs,hess,'k+',ms=10)
plt.vlines(nu_fit, hess.min(),hess.max(),'k',label="calculated nu_fit")
plt.vlines(nu_zero, hess.min(),hess.max(),'r',label="calculated nu_zero")
plt.hlines(0,tryfreqs.min(),tryfreqs.max(),'k',':')
plt.hlines(hessian[2],tryfreqs.min(),freqs.max(),'r','--',label="nu_zero covariance")
plt.xlabel("nu_fit [MHz]")
plt.ylabel("Hessian cross-term: d2(chi2)/dphidDM")
plt.title(title)
plt.legend(loc=4)
#plt.ylim(-1.5e8,1.0e8)
#plt.xlim(freqs.min(),freqs.max())
plt.show()
