#!/usr/bin/env python

#To be used with PSRCHIVE Archive files

#This fitting algorithm lays on the bed of Procrustes all too comfortably.

#Contributions by Scott M. Ransom (SMR) and Paul B. Demorest (PBD) 

#Next two lines needed for dispatching on nodes (not implemented yet)
#import matplotlib
#matplotlib.use("Agg")

import sys
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import numpy.fft as fft
import scipy.optimize as opt
import scipy.signal as ss
import lmfit as lm
import psrchive as pr

#Colormap preference
#Comment these out for dispatching on nodes (not implemented yet)
#plt.copper()
#plt.gray()
#plt.bone()
#plt.summer()
plt.pink()
plt.close("all")

#List of colors
cols = ['b', 'g', 'r', 'c', 'm', 'y',
        'b', 'g', 'r', 'c', 'm', 'y',
        'b', 'g', 'r', 'c', 'm', 'y',
        'b', 'g', 'r', 'c', 'm', 'y',
        'b', 'g', 'r', 'c', 'm', 'y']

#List of observatory codes; not sure what "0" corresponds to.
#Cross-check against TEMPO's obsys.dat; need TEMPO2 codes
obs_codes = {"bary":"@", "???":"0", "gbt":"1", "atca":"2", "ao":"3",
             "arecibo":"3", "nanshan":"5", "tid43":"6", "pks":"7", "jb":"8",
             "vla":"c", "ncy":"f", "eff":"g", "jbdfb":"q", "wsrt":"i",
             "lofar":"t"}

#RCSTRINGS dictionary, for the return codes given by scipy.optimize.fmin_tnc
RCSTRINGS = {"-1":"INFEASIBLE: Infeasible (low > up).",
             "0":"LOCALMINIMUM: Local minima reach (|pg| ~= 0).",
             "1":"FCONVERGED: Converged (|f_n-f_(n-1)| ~= 0.)",
             "2":"XCONVERGED: Converged (|x_n-x_(n-1)| ~= 0.)",
             "3":"MAXFUN: Max. number of function evaluations reach.",
             "4":"LSFAIL: Linear search failed.",
             "5":"CONSTANT: All lower bounds are equal to the upper bounds.",
             "6":"NOPROGRESS: Unable to progress.",
             "7":"USERABORT: User requested end of minimization."}

#Exact dispersion constant (e**2/(2*pi*m_e*c))
#(used by PRESTO)
Dconst_exact = 4.148808e3  #[MHz**2 cm**3 pc**-1 s]

#"Traditional" dispersion constant
#(used by PSRCHIVE)
Dconst_trad = 0.000241**-1 #[MHz**2 cm**3 pc**-1 s]

#Choose wisely.
Dconst = Dconst_trad

#Power-law index for scattering law
scattering_alpha = -4.0

#Default get_noise method
default_noise_method = "quarter"

#Ignore DC component in Fourier fit if DC_fact == 0, else DC_fact == 1.
DC_fact = 0

class DataBunch(dict):
    """
    This class is a great little recipe!  Creates a simple class instance
    db = DataBunch(a=1, b=2,....) that has attributes a and b, which are
    callable and update-able using either syntax db.a or db['a'].
    """
    def __init__(self, **kwds):
        dict.__init__(self, kwds)
        self.__dict__ = self

def gaussian_function(xs, loc, wid, norm=False):
    """
    See gaussian_profile.
    """
    mean = loc
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))
    scale = 1.0
    zs = (xs - mean) / sigma
    ys = np.exp(-0.5 * zs**2)
    if norm:
        scale *= (sigma**2.0 * 2.0 * np.pi)**-0.5
    return scale * ys


def gaussian_profile(nbin, loc, wid, norm=False, abs_wid=False, zeroout=True):
    """
    Taken and tweaked from SMR's pygaussfit.py

    Return a gaussian pulse profile with 'N' bins and peak amplitude of 1.
    If norm=True, return the profile such that the integrated density = 1.
        'nbin' = the number of bins in the profile
        'loc' = the pulse phase location (0-1) [rot]
        'wid' = the gaussian pulse's full width at half-max (FWHM) [rot]
    If abs_wid=True, will use abs(wid).
    If zeroout=True and wid <= 0, return a zero array.
    Note: The FWHM of a gaussian is approx 2.35482 "sigma", or exactly
          exactly 2*sqrt(2*ln(2)).
    """
    #Maybe should move these checks to gen_gaussian_portrait?
    if abs_wid:
        wid = abs(wid)
    if wid > 0.0:
        pass
    elif wid == 0.0:
        return np.zeros(nbin, 'd')
    elif wid < 0.0 and zeroout:
        return np.zeros(nbin, 'd')
    elif wid < 0.0 and not zeroout:
        pass
    else:
        return 0
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))
    mean = loc % 1.0
    locval = np.arange(nbin, dtype='d') / float(nbin)
    if (mean < 0.5):
        locval = np.where(np.greater(locval, mean + 0.5), locval - 1.0, locval)
    else:
        locval = np.where(np.less(locval, mean - 0.5), locval + 1.0, locval)
    try:
        zs = (locval - mean) / sigma
        okzinds = np.compress(np.fabs(zs) < 20.0, np.arange(nbin))   #Why 20?
        okzs = np.take(zs, okzinds)
        retval = np.zeros(nbin, 'd')
        np.put(retval, okzinds, np.exp(-0.5 * (okzs)**2.0)/(sigma * np.sqrt(2 *
            np.pi)))
        if norm:
            return retval
        #else:
        #    return retval / np.max(retval)
        else:
            if np.max(abs(retval)) == 0.0:
                return retval   #TTP hack
            else:
                return retval / np.max(abs(retval))  #TTP hack
    except OverflowError:
        print "Problem in gaussian prof:  mean = %f  sigma = %f" %(mean, sigma)
        return np.zeros(nbin, 'd')

def gen_gaussian_profile(params, nbin):
    """
    Taken and tweaked from SMR's pygaussfit.py

    Return a model of a DC-component + ngauss gaussian functions.
    params is a sequence of 1 + (ngauss*3) values; the first value is the DC
    component.  Each remaining group of three represents the gaussians
    loc (0-1), wid (i.e. FWHM) (0-1), and amplitude (>0.0).
    nbin is the number of bins in the model.

    cf. gaussian_profile(...)
    """
    ngauss = (len(params) - 2) / 3
    model = np.zeros(nbin, dtype='d') + params[0]
    for igauss in xrange(ngauss):
        loc, wid, amp = params[(2 + igauss*3):(5 + igauss*3)]
        model += amp * gaussian_profile(nbin, loc, wid)
    if params[1] != 0.0:
        bins = np.arange(nbin)
        sk = scattering_kernel(params[1], 1.0, np.array([1.0]), bins, P=1.0)[0]
        model = add_scattering(model, sk, repeat=3)
    return model

def gen_gaussian_portrait(params, phases, freqs, nu_ref, join_ichans=[],
        P=None):
    """
    Build the gaussian model portrait based on params.

    params is an array of 2 + (ngauss*6) values; the first value is the DC
    component.  Each remaining group of six represent the gaussians loc (0-1),
    linear slope in loc, wid (i.e. FWHM) (0-1), linear slope in wid,
    amplitude (>0,0), and spectral index alpha (no implicit negative).
    phases is the array of phase values (will pass nbin to
    gen_gaussian_profile).
    freqs in the array of frequencies at which to calculate the model.
    nu_ref [MHz] is the frequency to which the locs, wids, and amps reference.

    cf. gaussian_profile(...), gen_gaussian_profile(...)
    """
    njoin = len(join_ichans)
    if njoin:
        join_params = params[-njoin*2:]
        params = params[:-njoin*2]
    #Below, params[1] is multiplied by 0 so that scattering is taken care of
    #outside of gen_gaussian_profile
    refparams = np.array([params[0]] + [params[1]*0.0] + list(params[2::2]))
    tau = params[1]
    locparams = params[3::6]
    widparams = params[5::6]
    ampparams = params[7::6]
    ngauss = len(refparams[2::3])
    nbin = len(phases)
    nchan = len(freqs)
    gport = np.empty([nchan, nbin])
    gparams = np.empty([nchan, len(refparams)])
    #DC term
    gparams[:,0] = refparams[0]
    #Scattering term - first make unscattered portrait
    gparams[:,1] = refparams[1]
    #Locs
    gparams[:,2::3] = np.outer(freqs - nu_ref, locparams) + np.outer(np.ones(
        nchan), refparams[2::3])
    #Wids
    gparams[:,3::3] = np.outer(freqs - nu_ref, widparams) + np.outer(np.ones(
        nchan), refparams[3::3])
    #Amps
    gparams[:,4::3] = np.exp(np.outer(np.log(freqs) - np.log(nu_ref),
        ampparams) + np.outer(np.ones(nchan), np.log(refparams[4::3])))
    #Amps; I am unsure why I needed this fix at some point
    #gparams[:, 0::3][:, 1:] = np.exp(np.outer(np.log(freqs) - np.log(nu_ref),
    #    ampparams) + np.outer(np.ones(nchan), np.log(refparams[0::3][1:])))
    for ichan in xrange(nchan):
        #Need to contrain so values don't go negative, etc., which is currently
        #done in gaussian_profile
        gport[ichan] = gen_gaussian_profile(gparams[ichan], nbin)
    if tau != 0.0:
        sk = scattering_kernel(tau, nu_ref, freqs, np.arange(nbin), 1.0,
                alpha=scattering_alpha)
        gport = add_scattering(gport, sk, repeat=3)
    if njoin:
        for ij in range(njoin):
            join_ichan = join_ichans[ij]
            phi = join_params[0::2][ij]
            DM =  join_params[1::2][ij]
            gport[join_ichan] = rotate_portrait(gport[join_ichan], phi,
                    DM, P, freqs[join_ichan], nu_ref)
    return gport

def powlaw(nu, nu_ref, A, alpha):
    """
    Returns power-law 'spectrum' given by:
    F(nu) = A*(nu/nu_ref)**alpha
    """
    return A * (nu/nu_ref)**alpha

def powlaw_integral(nu2, nu1, nu_ref, A, alpha):
    """
    Returns the integral over a powerlaw of form A*(nu/nu_ref)**alpha
    from nu1 to nu2.

    cf. powlaw(...)
    """
    alpha = np.float(alpha)
    if alpha == -1.0:
        return A * nu_ref * np.log(nu2/nu1)
    else:
        C = A * (nu_ref**-alpha) / (1 + alpha)
        diff = ((nu2**(1+alpha)) - (nu1**(1+alpha)))
        return C * diff

def powlaw_freqs(lo, hi, N, alpha, mid=False):
    """
    Returns frequencies such that a bandwidth from lo to hi frequencies
    split into N channels contains the same amount of flux in each channel,
    given a power-law across the band with spectral index alpha.  Default
    behavior returns N+1 frequencies (includes both lo and hi freqs); if
    mid=True, will return N frequencies, corresponding to the middle frequency
    in each channel.

    cf. powlaw(...)
    """
    alpha = np.float(alpha)
    nus = np.zeros(N + 1)
    if alpha == -1.0:
        nus = np.exp(np.linspace(np.log(lo), np.log(hi), N+1))
    else:
        nus = np.power(np.linspace(lo**(1+alpha), hi**(1+alpha), N+1),
                (1+alpha)**-1)
        #Equivalently:
        #for ii in xrange(N+1):
        #    nus[ii] = ((ii / np.float(N)) * (hi**(1+alpha)) + (1 - (ii /
        #        np.float(N))) * (lo**(1+alpha)))**(1 / (1+alpha))
    if mid:
        midnus = np.zeros(N)
        for ii in xrange(N):
            midnus[ii] = 0.5 * (nus[ii] + nus[ii+1])
        nus = midnus
    return nus

def scattering_kernel(tau, nu_ref, freqs, phases, P, alpha=scattering_alpha):
    """
    Phase-bin centers...tau in [sec] or [bin]
    """
    nchan = len(freqs)
    nbin = len(phases)
    if tau == 0.0:
        ts = np.zeros([nchan, nbin])
        ts[:,0] = 1.0
    else:
        ts = np.array([phases*P for ichan in xrange(nchan)])
        taus = tau * (freqs / nu_ref)**alpha
        sk = np.exp(-np.transpose(np.transpose(ts) * taus**-1.0))
    return sk

def add_scattering(data, kernel, repeat=3):
    mid = repeat/2
    d = np.array(list(data.transpose()) * repeat).transpose()
    k = np.array(list(kernel.transpose()) * repeat).transpose()
    if len(data.shape) == 1:
        nbin = data.shape[0]
        norm_kernel = kernel / kernel.sum()
        scattered_data = ss.convolve(norm_kernel, d)[mid * nbin : (mid+1) *
                nbin]
    else:
        nbin = data.shape[1]
        norm_kernel = np.transpose(np.transpose(k) * k.sum(axis=1)**-1)
        scattered_data = np.fft.irfft(np.fft.rfft(norm_kernel) *
                np.fft.rfft(d))[:, mid * nbin : (mid+1) * nbin]
    return scattered_data

def add_scintillation(port, params=None, random=True, nsin=2, amax=1.0,
        wmax=3.0):
    """
    Should be used before adding noise.
    params are triplets of "amps", "freqs" [cycles], and "phases" [cycles]
    """
    nchan = len(port)
    pattern = np.zeros(nchan)
    if params is None and random is False:
        return port
    elif params is not None:
        nsin = len(params)/3
        for isin in xrange(nsin):
            a,w,p = params[isin*3:isin*3 + 3]
            pattern += a * np.sin(np.linspace(0, w * np.pi, nchan) +
                    p*np.pi)**2
    else:
        for isin in range(nsin):
            (a,w,p) = (np.random.uniform(0,amax),
                    np.random.chisquare(wmax), np.random.uniform(0,1))
            pattern += a * np.sin(np.linspace(0, w * np.pi, nchan) +
                    p*np.pi)**2
    return np.transpose(np.transpose(port) * pattern)

def fit_powlaw_function(params, freqs, nu_ref, data=None, errs=None):
    """
    Function to pass to minimizer in fit_powlaw that returns residuals weighted    by errs**-1.

    params is an array = [amplitude at reference frequency, spectral index].
    freqs is an nchan array of frequencies.
    nu_ref is the frequency at which the amplitude is referenced.
    data are the data values.
    errs is/are the standard error(s) on the data values.

    cf. powlaw(...), fit_powlaw(...)
    """
    prms = np.array([param.value for param in params.itervalues()])
    A = prms[0]
    alpha = prms[1]
    return (data - powlaw(freqs, nu_ref, A, alpha)) / errs

def fit_gaussian_profile_function(params, data=None, errs=None):
    """
    Function to pass to minimizer in fit_gaussian_profile that returns
    residuals weighted by errs**-1.

    params is a sequence of 1 + (ngauss*3) values; the first value is the DC
    component.  Each remaining group of three represents the gaussians
    loc (0-1), wid (i.e. FWHM) (0-1), and amplitude (>0.0).
    data are the data values.
    errs is/are the standard errors(s) on the data values.

    cf. gen_gaussian_profile(...), fit_gaussian_profile(...)
    """
    prms = np.array([param.value for param in params.itervalues()])
    return (data - gen_gaussian_profile(prms, len(data))) / errs

def fit_gaussian_portrait_function(params, phases, freqs, nu_ref, data=None,
        errs=None, join_ichans=None, P=None):
    """
    Function to pass to minimizer in fit_gaussian_portrait that returns
    residuals weighted by errs**-1.

    params is an array of 1 + (ngauss*6) values; the first value is the DC
    component.  Each remaining group of six represent the gaussians loc (0-1),
    linear slope in loc, wid (i.e. FWHM) (0-1), linear slope in wid,
    amplitude (>0,0), and spectral index alpha (no implicit negative).
    phases is the array of phase values (will pass nbin to
    gen_gaussian_profile).
    freqs in the array of frequencies at which to calculate the model.
    nu_ref [MHz] is the frequency to which the locs, wids, and amps reference.
    errs is/are the standard errors(s) on the data values.

    cf. gen_gaussian_portrait(...), fit_gaussian_portrait(...)
    """
    prms = np.array([param.value for param in params.itervalues()])
    deviates = np.ravel((data - gen_gaussian_portrait(prms, phases, freqs,
        nu_ref, join_ichans, P)) / errs)
    return deviates

def fit_phase_shift_function(phase, model=None, data=None, err=None):
    """
    Returns phase-offset such that data would have to be rotated (with
    rotate_data or similar) by phase amount to match model.
    """
    harmind = np.arange(len(model))
    phasor = np.exp(harmind * 2.0j * np.pi * phase)
    C = -np.real((data * np.conj(model) * phasor).sum()) / err**2.0
    return C

def fit_phase_shift_function_deriv(phase, model=None, data=None, err=None):
    """
    """
    harmind = np.arange(len(model))
    phasor = np.exp(harmind * 2.0j * np.pi * phase)
    dC = -np.real((2.0j * np.pi * harmind * data * np.conj(model) *
        phasor).sum()) / err**2.0
    return dC

def fit_phase_shift_function_2deriv(phase, model=None, data=None, err=None):
    """
    """
    harmind = np.arange(len(model))
    phasor = np.exp(harmind * 2.0j * np.pi * phase)
    d2C = -np.real((-4.0 * (np.pi**2.0) * (harmind**2.0) * data *
        np.conj(model) * phasor).sum()) / err**2.0
    return d2C

def fit_portrait_function(params, model=None, p_n=None, data=None, errs=None,
        P=None, freqs=None, nu_ref=np.inf):
    """
    This is the function to be minimized by fit_portrait.  The returned value
    is equivalent to the chi**2 value of the model and data, given the input
    parameters, differing only by a constant depending on a weighted sum of the
    data (see 'd' in fit_portrait(...)).

    NB: both model and data must already be in the Fourier domain.

    params is an array = [phase, DM], with phase in [rot] and DM in
    [cm**-3 pc].
    model is the nchan x nbin phase-frequency model portrait that has been
    DFT'd along the phase axis.
    p_n is an nchan array containing a phase-average of the model; see p_n in
    fit_portrait(...).
    data is the nchan x nbin phase-frequency data portrait that has been DFT'd
    along the phase axis.
    errs is/are the standard error(s) on the data values (in the Fourier
    domain!).
    P is the period [s] of the pulsar at the data epoch.
    freqs is an nchan array of frequencies [MHz].
    nu_ref is the frequency [MHz] that is designated to have zero delay from a
    non-zero dispersion measure.

    cf. fit_portrait(...)
    """
    phase = params[0]
    m = 0.0
    if P == None or freqs == None:
        D = 0.0
        freqs = np.inf * np.ones(len(model))
    else: D = Dconst * params[1] / P
    for nn in xrange(len(freqs)):
        freq = freqs[nn]
        p = p_n[nn]
        err = errs[nn]
        harmind = np.arange(len(model[nn]))
        phasor = np.exp(harmind * 2.0j * np.pi * (phase + (D * (freq**-2.0 -
            nu_ref**-2.0))))
        #Cdp is related to the inverse DFT of the cross-correlation 
        Cdp = np.real(data[nn,:] * np.conj(model[nn,:]) * phasor).sum()
        m += (Cdp**2.0) / (err**2.0 * p)
    return -m

def fit_portrait_function_deriv(params, model=None, p_n=None, data=None,
        errs=None, P=None, freqs=None, nu_ref=np.inf):
    """
    Returns the first-derivatives of the fit_portrait_function(...)
    (i.e. chi**2) with respect to the two parameters, phase and DM.

    cf. fit_portrait_function(...)
    """
    phase = params[0]
    D = Dconst * params[1] / P
    d_phi, d_DM = 0.0, 0.0
    for nn in xrange(len(freqs)):
        freq = freqs[nn]
        p = p_n[nn]
        err = errs[nn]
        harmind = np.arange(len(model[nn]))
        phasor = np.exp(harmind * 2.0j * np.pi * (phase + (D * (freq**-2.0 -
            nu_ref**-2.0))))
        Cdp = np.real(data[nn,:] * np.conj(model[nn,:]) * phasor).sum()
        dCdp1 = np.real(2.0j * np.pi * harmind * data[nn,:] *
                np.conj(model[nn,:]) * phasor).sum()
        dDM = (freq**-2.0 - nu_ref**-2.0) * (Dconst/P)
        d_phi += -2 * Cdp * dCdp1 / (err**2.0 * p)
        d_DM += -2 * Cdp * dCdp1 * dDM / (err**2.0 * p)
    return np.array([d_phi, d_DM])

def fit_portrait_function_2deriv(params, model=None, p_n=None, data=None,
        errs=None, P=None, freqs=None, nu_ref=np.inf):
    """
    Returns the three unique values in the Hessian, which is a 2x2
    symmetric matrix of the second-derivatives of fit_portrait_function(...).
    The curvature matrix is one-half the second-derivative of the chi**2
    function (this function).
    The covariance matrix is the inverse of the curvature matrix.

    If transform is True, return the diagonalized Hessian values, i.e. the
    zero-covariance values.

    cf. fit_portrait, fit_portrait_function
    """
    phase = params[0]
    D = Dconst * params[1] / P
    d2_phi, d2_DM, d2_cross = 0.0, 0.0, 0.0
    W_n = np.empty(len(freqs))
    for nn in xrange(len(freqs)):
        freq = freqs[nn]
        p = p_n[nn]
        err = errs[nn]
        harmind = np.arange(len(model[nn]))
        phasor = np.exp(harmind * 2.0j * np.pi * (phase + (D *
            (freq**-2.0 - nu_ref**-2.0))))
        Cdp = np.real(data[nn,:] * np.conj(model[nn,:]) * phasor).sum()
        dCdp1 = np.real(2.0j * np.pi * harmind * data[nn,:] *
                np.conj(model[nn,:]) * phasor).sum()
        dCdp2 = np.real(pow(2.0j * np.pi * harmind, 2.0) * data[nn,:] *
                np.conj(model[nn,:]) * phasor).sum()
        dDM = (freq**-2.0 - nu_ref**-2.0) * (Dconst/P)
        W = (pow(dCdp1, 2.0) + (Cdp * dCdp2))
        W_n[nn] = W / (err**2.0 * p)
        d2_phi += -2.0 * W / (err**2.0 * p)
        d2_DM += -2.0 * W * dDM**2.0 / (err**2.0 * p)
        d2_cross += -2.0 * W * dDM / (err**2.0 * p)
    nu_zero = (W_n.sum() / np.sum(W_n * freqs**-2))**0.5
    return (np.array([d2_phi, d2_DM, d2_cross]), nu_zero)

def fit_portrait_function_2deriv_full(params, model=None, p_n=None, data=None,
        errs=None, P=None, freqs=None, nu_ref=np.inf):
    return 0

def estimate_portrait(phase, DM, scales, data, errs, P, freqs, nu_ref=np.inf):
    #here, all vars have additional epoch-index except nu_ref
    #i.e. all have to be arrays of at least len 1; errs are precision
    """
    An early attempt to make a '2-D' version of PBD's autotoa.  That is, to
    iterate over all epochs of data portraits to build a non-gaussian model
    that can be smoothed.

    <UNDER CONSTRUCTION>

    cf. PBD's autotoa, PhDT
    """
    dFFT = fft.rfft(data, axis=2)
    dFFT[:, :, 0] *= DC_fact
    #errs = np.real(dFFT[:, :, -len(dFFT[0,0])/4:]).std(axis=2)**-2.0
    #errs = get_noise(data, chans=True) * np.sqrt(len(data[0])/2.0)
    D = Dconst * DM / P
    freqs2 = freqs**-2.0 - nu_ref**-2.0
    phiD = np.outer(D, freqs2)
    phiprime = np.outer(phase, np.ones(len(freqs))) + phiD
    weight = np.sum(pow(scales, 2.0) / errs**2.0, axis=0)**-1
    phasor = np.array([np.exp(2.0j * np.pi * kk * phiprime) for kk in xrange(
        len(dFFT[0,0]))]).transpose(1,2,0)
    p = np.sum(np.transpose(np.transpose(scales / errs**2.0) *
        np.transpose(phasor * dFFT)), axis=0)
    wp = np.transpose(weight * np.transpose(p))
    return wp

def wiener_filter(prof, noise):
    #FIX does not work
    """
    Returns the 'optimal' Wiener filter given a noisy pulse profile.

    <UNDER CONSTRUCTION>

    prof is a noisy pulse profile.
    noise is standard deviation of the data.

    cf. PBD's PhDT
    """
    FFT = fft.rfft(prof)
    pows = np.real(FFT * np.conj(FFT)) / len(prof)
    return pows / (pows + (noise**2))
    #return (pows - (noise**2)) / pows

def brickwall_filter(N, kc):
    """
    Returns a 'brickwall' filter with N points; the first kc are ones, the
    remainder are zeros.
    """
    fk = np.zeros(N)
    fk[:kc] = 1.0
    return fk

def fit_brickwall(prof, noise):
    """
    Attempts to find the critical harmonic bin index given a noisey pulse
    profile, beyond which there is no signal.

    <UNDER CONSTRUCTION>

    """
    wf = wiener_filter(prof, noise)
    N = len(wf)
    X2 = np.zeros(N)
    for ii in xrange(N):
        X2[ii] = np.sum((wf - brickwall_filter(N, ii))**2)
    return X2.argmin()

def half_triangle_function(a, b, dc, N):
    fn = np.zeros(N) + dc
    a = int(np.floor(a))
    fn[:a] += -(float(b)/a)*np.arange(a) + b
    return fn

def find_kc_function(params, data):
    a, b, dc = params[0], params[1], params[2]
    return np.sum((data - half_triangle_function(a, b, dc, len(data)))**2.0)

def find_kc(pows):
    """
    """
    data = np.log10(pows)
    other_args = [data]
    results = opt.brute(find_kc_function, [tuple((1, len(data))),
        tuple((0, data.max()-data.min())), tuple((data.min(), data.max()))],
        args=other_args, Ns=10, full_output=False, finish=None)
    a, b, dc = results[0], results[1], results[2]
    return int(np.floor(a))

def fit_powlaw(data, init_params, errs, freqs, nu_ref):
    """
    Fits a power-law to input data using lmfit's least-squares algorithm.
    Returns an array of fitted parameters, an array of parameter error
    estimates, the chi**2 value, the degrees of freedom, and the residuals
    between the best-fit model and the data.

    data is the input array of data values used in the fit.
    init_params are the initial guesses for the [amplitude at nu_ref,
    spectral index].
    errs is/are the standard errors(s) on the data values.
    freqs is an nchan array of frequencies.
    nu_ref is the frequency at which the amplitude is referenced.

    """
    #Generate the parameter structure
    params = lm.Parameters()
    params.add('amp', init_params[0], vary=True, min=None, max=None)
    params.add('alpha', init_params[1], vary=True, min=None, max=None)
    other_args = {'freqs':freqs, 'nu_ref':nu_ref, 'data':data, 'errs':errs}
    #Now fit it
    results = lm.minimize(fit_powlaw_function, params, kws=other_args)
    fitted_params = np.array([param.value for param in
        results.params.itervalues()])
    dof = results.nfree
    chi_sq = results.chisqr
    redchi_sq = results.redchi
    residuals = results.residual
    fit_errs = np.array([param.stderr for param in
        results.params.itervalues()])
    return fitted_params, fit_errs, chi_sq, dof, residuals

def fit_gaussian_profile(data, init_params, errs, fit_scattering=False,
        quiet=True):
    """
    Fits a set of gaussians to a pulse profile using lmfit's least-squares
    algorithm.
    Returns an array of fitted parameters, the reduced chi**2 value, the number
    of degrees of freedom, and the residuals between the best-fit model and the
    data.

    data is the nbin pulse profile used in the fit.
    init_params is an array of initial guesses for the 1 + (ngauss*3) values;
    the first value is the DC component.  Each remaining group of three
    represents the gaussians loc (0-1), wid (i.e. FWHM) (0-1), and
    amplitude (>0.0).
    errs is/are the standard errors(s) on the data values.
    quiet=True suppresses output [default].
    """
    nparam = len(init_params)
    ngauss = (len(init_params) - 2) / 3
    fs = fit_scattering
    #Generate the parameter structure
    params = lm.Parameters()
    for ii in xrange(nparam):
        if ii == 0:
            params.add('dc', init_params[ii], vary=True, min=None, max=None,
                    expr=None)
        elif ii ==1:
            params.add('tau', init_params[ii], vary=fs, min=0.0, max=None,
                    expr=None)
        elif ii in range(nparam)[2::3]:
            params.add('loc%s'%str((ii-2)/3 + 1), init_params[ii], vary=True,
                    min=None, max=None, expr=None)
        elif ii in range(nparam)[3::3]:
            params.add('wid%s'%str((ii-3)/3 + 1), init_params[ii], vary=True,
                    min=0.0, max=None, expr=None)
        elif ii in range(nparam)[4::3]:
            params.add('amp%s'%str((ii-4)/3 + 1), init_params[ii], vary=True,
                    min=0.0, max=None, expr=None)
        else:
            print "Undefined index %d."%ii
            sys.exit()
    other_args = {'data':data, 'errs':errs}
    #Now fit it
    results = lm.minimize(fit_gaussian_profile_function, params,
            kws=other_args)
    fitted_params = np.array([param.value for param in
        results.params.itervalues()])
    dof = results.nfree
    redchi_sq = results.redchi
    residuals = results.residual
    if not quiet:
        print "---------------------------------------------------------------"
        print "Multi-Gaussian Profile Fit Results"
        print "---------------------------------------------------------------"
        print "lmfit status:", results.message
        print "gaussians:", ngauss
        print "DOF:", dof
        print "reduced chi-sq: %.2f" % redchi_sq
        print "residuals mean: %.3g" % np.mean(residuals)
        print "residuals stdev: %.3g" % np.std(residuals)
        print "---------------------------------------------------------------"
    return fitted_params, redchi_sq, dof, residuals

def fit_gaussian_portrait(data, init_params, errs, fit_flags, phases, freqs,
        nu_ref, join_params=[], P=None, quiet=True):
    """
    Fits a set of evolving gaussians to a phase-frequency pulse portrait using
    lmfit's least-squares algorithm.
    Returns an array of fitted values, the chi**2 value, and the number of
    degrees of freedom.

    data is the nchan x nbin phase-frequency data portrait used in the fit.
    init_params is an array of initial guesses for the 1 + (ngauss*6)
    parameters in the model; the first value is the DC component.  Each
    remaining group of six represent the gaussians loc (0-1), linear slope in
    loc, wid (i.e. FWHM) (0-1), linear slope in wid, amplitude (>0,0), and
    spectral index alpha (no implicit negative).
    errs is/are the standard errors(s) on the data values.
    fit_flags is an array of 1 + (ngauss*6) values, where non-zero entries
    signify that the parameter should be fit.
    phases is the array of phase values (will pass nbin to
    gen_gaussian_profile).
    freqs in the array of frequencies at which to calculate the model.
    nu_ref [MHz] is the frequency to which the locs, wids, and amps reference.
    quiet=True suppresses output [default].

    """
    nparam = len(init_params)
    ngauss = (len(init_params) - 2) / 6
    #Generate the parameter structure
    params = lm.Parameters()
    for ii in xrange(nparam):
        if ii == 0:         #DC, not limited
            params.add('dc', init_params[ii], vary=bool(fit_flags[ii]),
                    min=None, max=None, expr=None)
        elif ii == 1:       #tau, limited by 0
            params.add('tau', init_params[ii], vary=bool(fit_flags[ii]),
                    min=0.0, max=None, expr=None)
        elif ii%6 == 2:     #loc limits
            params.add('loc%s'%str((ii-2)/6 + 1), init_params[ii],
                    vary=bool(fit_flags[ii]), min=None, max=None, expr=None)
        elif ii%6 == 3:     #loc slope limits
            params.add('m_loc%s'%str((ii-3)/6 + 1), init_params[ii],
                    vary=bool(fit_flags[ii]), min=None, max=None, expr=None)
        elif ii%6 == 4:     #wid limits, limited by 0
            params.add('wid%s'%str((ii-4)/6 + 1), init_params[ii],
                    vary=bool(fit_flags[ii]), min=0.0, max=None, expr=None)
        elif ii%6 == 5:     #wid slope limits
            params.add('m_wid%s'%str((ii-5)/6 + 1), init_params[ii],
                    vary=bool(fit_flags[ii]), min=None, max=None, expr=None)
        elif ii%6 == 0:     #amp limits, limited by 0
            params.add('amp%s'%str((ii-6)/6 + 1), init_params[ii],
                    vary=bool(fit_flags[ii]), min=0.0, max=None, expr=None)
        elif ii%6 == 1:     #amp index limits
            params.add('alpha%s'%str((ii-7)/6 + 1), init_params[ii],
                    vary=bool(fit_flags[ii]), min=None, max=None, expr=None)
        else:
            print "Undefined index %d."%ii
            sys.exit()
    if len(join_params):
        join_ichans = join_params[0]
        njoin = len(join_ichans)
        for ii in xrange(njoin):
            params.add('phase%s'%str(ii+1), join_params[1][0::2][ii],
                    vary=bool(join_params[2][0::2][ii]), min=None, max=None,
                        expr=None)
            params.add('DM%s'%str(ii+1), join_params[1][1::2][ii],
                    vary=bool(join_params[2][1::2][ii]), min=None, max=None,
                        expr=None)
    else:
        join_ichans = []
    other_args = {'data':data, 'errs':errs, 'phases':phases, 'freqs':freqs,
            'nu_ref':nu_ref, 'join_ichans':join_ichans, 'P':P}
    #Now fit it
    results = lm.minimize(fit_gaussian_portrait_function, params,
            kws=other_args)
    fitted_params = np.array([param.value for param in
        results.params.itervalues()])
    dof = results.nfree
    chi_sq = results.chisqr
    redchi_sq = results.redchi
    residuals = results.residual
    if not quiet:
        print "---------------------------------------------------------------"
        print "Gaussian Portrait Fit"
        print "---------------------------------------------------------------"
        print "lmfit status:", results.message
        print "gaussians:", ngauss
        print "DOF:", dof
        print "reduced chi-sq: %.2f" %redchi_sq
        print "residuals mean: %.3g" %np.mean(residuals)
        print "residuals stdev: %.3g" %np.std(residuals)
        print "---------------------------------------------------------------"
    return fitted_params, chi_sq, dof

def fit_portrait(data, model, init_params, P, freqs, nu_fit=np.inf,
        nu_out=None, errs=None, bounds=[(None, None), (None, None)], id=None,
        quiet=True):
    """
    """
    dFFT = fft.rfft(data, axis=1)
    dFFT[:, 0] *= DC_fact
    mFFT = fft.rfft(model, axis=1)
    mFFT[:, 0] *= DC_fact
    if errs is None:
        #errs = np.real(dFFT[:, -len(dFFT[0])/4:]).std(axis=1)
        errs = get_noise(data, chans=True) * np.sqrt(len(data[0])/2.0)
    d = np.real(np.sum(np.transpose(errs**-2.0 * np.transpose(dFFT *
        np.conj(dFFT)))))
    p_n = np.real(np.sum(mFFT * np.conj(mFFT), axis=1))
    #BEWARE BELOW! Order matters!
    other_args = (mFFT, p_n, dFFT, errs, P, freqs, nu_fit)
    minimize = opt.minimize
    #fmin_tnc seems to work best, fastest
    method = 'TNC'
    start = time.time()
    results = minimize(fit_portrait_function, init_params, args=other_args,
            method=method, jac=fit_portrait_function_deriv, bounds=bounds,
            options={'maxiter':1000, 'disp':False})
    duration = time.time() - start
    phi = results.x[0]
    DM = results.x[1]
    nfeval = results.nfev
    return_code = results.status
    rcstring = RCSTRINGS["%s"%str(return_code)]
    #If the fit fails...????  These don't seem to be great indicators of the
    #fit failing
    #if results.success is not True:
    if not quiet and results.success is not True:
        if id is not None:
            ii = id[::-1].index("_")
            isubx = id[-ii:]
            ii += 1
            jj = ii + id[:-ii][::-1].index("_")
            isub = id[-jj:-ii]
            filename = id[:-jj-1]
            sys.stderr.write(
                    "Fit failed with return code %d -- %s; %s subint %s subintx %s\n"%(results.status, rcstring, filename, isub, isubx))
        else:
            sys.stderr.write(
                    "Fit failed with return code %d -- %s"%(results.status, rcstring))
    if not quiet and results.success is True:
        sys.stderr.write("Fit succeeded with return code %d -- %s\n"
                %(results.status, rcstring))
    #Curvature matrix = 1/2 2deriv of chi2 (cf. Gregory sect 11.5)
    #Parameter errors are related to curvature matrix by **-0.5 
    #Calculate nu_zero
    nu_zero = fit_portrait_function_2deriv(np.array([phi, DM]), mFFT,
            p_n, dFFT, errs, P, freqs, nu_fit)[1]
    if nu_out is None:
        nu_out = nu_zero
    phi_out = phase_transform(phi, DM, nu_fit, nu_out, P)
    #Calculate Hessian
    hessian = fit_portrait_function_2deriv(np.array([phi_out, DM]),
            mFFT, p_n, dFFT, errs, P, freqs, nu_out)[0]
    hessian = np.array([[hessian[0], hessian[2]], [hessian[2], hessian[1]]])
    covariance_matrix = np.linalg.inv(0.5*hessian)
    covariance = covariance_matrix[0,1]
    #These are true 1-sigma errors iff covariance = 0
    param_errs = list(covariance_matrix.diagonal()**0.5)
    DoF = len(data.ravel()) - (len(freqs) + 2)
    red_chi2 = (d + results.fun) / DoF
    scales = get_scales(data, model, phi, DM, P, freqs, nu_fit)
    #Errors on scales, if ever needed (these are wrong b/c of covariances)
    param_errs += list(pow(p_n / errs**2.0, -0.5))
    #The below should be changed to a DataBunch
    return (phi_out, DM, scales, np.array(param_errs), nu_out, covariance,
            red_chi2, duration, nfeval, return_code)

def fit_phase_shift(data, model, err=None, bounds=[-0.5, 0.5]):
    """
    """
    dFFT = fft.rfft(data)
    dFFT[0] *= DC_fact
    mFFT = fft.rfft(model)
    mFFT[0] *= DC_fact
    if err is None:
        #err = np.real(dFFT[-len(dFFT)/4:]).std()
        err = get_noise(data) * np.sqrt(len(data)/2.0)
    d = np.real(np.sum(dFFT * np.conj(dFFT))) / err**2.0
    p = np.real(np.sum(mFFT * np.conj(mFFT))) / err**2.0
    other_args = (mFFT, dFFT, err)
    results = opt.brute(fit_phase_shift_function, [tuple(bounds)],
            args=other_args, Ns=100, full_output=True)
    phase = results[0][0]
    fmin = results[1]
    scale = -fmin / p
    #In the next two error equations, consult fit_portait for factors of 2
    phase_error = (scale * fit_phase_shift_function_2deriv(phase, mFFT, dFFT,
        err))**-0.5
    scale_error = p**0.5
    errors = [phase_error, scale_error]
    red_chi2 = (d - ((fmin**2) / p)) / (len(data) - 2)
    return DataBunch(errors=errors, phase=phase, red_chi2=red_chi2,
            scale=scale)

def get_noise(data, method=default_noise_method, **kwargs):
    """
    """
    if method == "quarter":
        return get_noise_quarter(data, **kwargs)
    elif method == "fit":
        return get_noise_fit(data, **kwargs)
    else:
        print "Unknown get_noise method."
        return 0

def get_noise_quarter(data, frac=4, chans=False):
    """
    """
    if chans:
        noise = np.zeros(len(data))
        for ichan in xrange(len(noise)):
            prof = data[ichan]
            FFT = fft.rfft(prof)
            pows = np.real(FFT * np.conj(FFT)) / len(prof)
            kc = (1 - frac**-1)*len(pows)
            noise[ichan] = np.sqrt(np.mean(pows[kc:]))
        return noise
    else:
        raveld = data.ravel()
        FFT = fft.rfft(raveld)
        pows = np.real(FFT * np.conj(FFT)) / len(raveld)
        kc = (1 - frac**-1)*len(pows)
        return np.sqrt(np.mean(pows[kc:]))

def get_noise_fit(data, fact=1.1, chans=False):
    """
    Relatively slow; could use a speed up.
    """
    if chans:
        noise = np.zeros(len(data))
        for ichan in xrange(len(noise)):
            prof = data[ichan]
            FFT = fft.rfft(prof)
            pows = np.real(FFT * np.conj(FFT)) / len(prof)
            k_crit = fact * find_kc(pows)
            if k_crit >= len(pows):
                #Will only matter in unresolved or super narrow, high SNR cases
                k_crit = min(int(0.99*len(pows)), k_crit)
            noise[ichan] = np.sqrt(np.mean(pows[k_crit:]))
        return noise
    else:
        raveld = data.ravel()
        FFT = fft.rfft(raveld)
        pows = np.real(FFT * np.conj(FFT)) / len(raveld)
        k_crit = fact * find_kc(pows)
        if k_crit >= len(pows):
            #Will only matter in unresolved or super narrow, high SNR cases
            k_crit = min(int(0.99*len(pows)), k_crit)
        return np.sqrt(np.mean(pows[k_crit:]))

def get_SNR(prof, fudge=3.25):
    """
    From Lorimer & Kramer (2005).
    Assuming baseline removed!
    fudge is factor in a (bad) attempt to match PSRCHIVE
    """
    noise = get_noise(prof)
    nbin = len(prof)
    #dc = np.real(np.fft.rfft(data))[0]
    dc = 0
    Weq = (prof - dc).sum() / (prof - dc).max()
    mask = np.where(Weq <= 0.0, 0.0, 1.0)
    Weq = np.where(Weq <= 0.0, 1.0, Weq)
    SNR = (prof - dc).sum() / (noise * Weq**0.5)
    return (SNR * mask) / fudge

def get_scales(data, model, phase, DM, P, freqs, nu_ref=np.inf):
    """
    """
    scales = np.zeros(len(freqs))
    dFFT = fft.rfft(data, axis=1)
    dFFT[:, 0] *= DC_fact
    mFFT = fft.rfft(model, axis=1)
    mFFT[:, 0] *= DC_fact
    p_n = np.real(np.sum(mFFT * np.conj(mFFT), axis=1))
    D = Dconst * DM / P
    harmind = np.arange(len(mFFT[0]))
    phasor = np.exp(2.0j * np.pi * np.outer((phase + (D * (freqs**-2.0 -
        nu_ref**-2.0))), harmind))
    scales = np.real(np.sum(dFFT * np.conj(mFFT) * phasor, axis=1))
    scales /= p_n
    return scales

def rotate_data(data, phase=0.0, DM=0.0, Ps=None, freqs=None,
        nu_ref=np.inf, taxis=None, faxis=-2, baxis=-1,):
    """
    Positive values of phase and DM rotate to earlier phase ("dedisperses").
    When used to dediserpse, rotate_portrait is virtually identical to
    arch.dedisperse() in PSRCHIVE.

    faxis, freqs needed if rotating for DM != 0.0
    taxis needed if len(Ps) > 1, for DM != 0.0

    ***Need to change this convention.  Seems intuituve to have positive phase
    rotate to later, and positive DM rotate to later.***
    """
    shape = data.shape
    ndim = len(shape)
    idim = 'ijklmnop'
    idim = idim[:ndim]
    iaxis = range(ndim)
    baxis = iaxis[baxis]
    bdim = list(idim)[baxis]
    dFFT = fft.rfft(data, axis=baxis)
    nharm = dFFT.shape[baxis]
    harmind = np.arange(nharm)
    if DM == 0.0:
        D = 0.0
        baxis = iaxis.pop(baxis)
        othershape = np.take(shape, iaxis)
        ones = np.ones(othershape)
        order = np.take(list(idim), iaxis)
        order = ''.join([order[xx] for xx in xrange(len(order))])
        phasor = np.exp(harmind * 2.0j * np.pi * phase)
        phasor = np.einsum(order + ',' + bdim, ones, phasor)
        dFFT *= phasor
    else:
        D = Dconst * DM / Ps
        if taxis:
            taxis = iaxis[taxis]
            tdim = list(idim)[taxis]
            nsub = D.shape[0]
        else:
            pass
        faxis = iaxis[faxis]
        fdim = list(idim)[faxis]
        nchan = shape[faxis]
        fterm = freqs**-2.0 - nu_ref**-2.0
        if taxis:
            phase += np.einsum('i,j', D, fterm)
            phase = np.einsum('ij,k', phase, harmind)
        else:
            phase += D * fterm
            phase = np.einsum('i,j', phase, harmind)
        phasor = np.exp(2.0j * np.pi * phase)
        if taxis:
            inds = np.array([taxis, faxis, baxis])
        else:
            inds = np.array([faxis, baxis])
        inds.sort()
        inds = inds[::-1]
        inds = np.array([iaxis.pop(ii) for ii in inds])
        othershape = np.take(shape, iaxis)
        order = np.take(list(idim), iaxis)
        order = ''.join([order[xx] for xx in xrange(len(order))])
        ones = np.ones(othershape)
        if taxis:
            phasor = np.einsum(order + ',' + tdim + fdim + bdim, ones, phasor)
        else:
            phasor = np.einsum(order + ',' + fdim + bdim, ones, phasor)
        dFFT *= phasor
    return fft.irfft(dFFT, axis=baxis)

def rotate_portrait(port, phase=0.0, DM=None, P=None, freqs=None,
        nu_ref=np.inf):
    """
    Positive values of phase and DM rotate to earlier phase ("dedisperses").
    When used to dediserpse, rotate_portrait is virtually identical to
    arch.dedisperse() in PSRCHIVE.
    """
    pFFT = fft.rfft(port, axis=1)
    for nn in xrange(len(pFFT)):
        if DM is None and freqs is None:
            pFFT[nn,:] *= np.exp(np.arange(len(pFFT[nn])) * 2.0j * np.pi *
                    phase)
        else:
            D = Dconst * DM / P
            freq = freqs[nn]
            phasor = np.exp(np.arange(len(pFFT[nn])) * 2.0j * np.pi * (phase +
                (D * (freq**-2.0 - nu_ref**-2.0))))
            pFFT[nn,:] *= phasor
    return fft.irfft(pFFT)

def rotate_profile(profile, phase=0.0):
    """
    Positive values rotate to earlier phase.
    """
    pFFT = fft.rfft(profile)
    pFFT *= np.exp(np.arange(len(pFFT)) * 2.0j * np.pi * phase)
    return fft.irfft(pFFT)

def fft_rotate(arr, bins):
    """
    Ripped and altered from PRESTO

    Return array 'arr' rotated by 'bins' places to the left.
    The rotation is done in the Fourier domain using the Shift Theorem.
    'bins' can be fractional.
    The resulting vector will have the same length as the original.
    """
    arr = np.asarray(arr)
    freqs = np.arange(arr.size/2 + 1, dtype=np.float)
    phasor = np.exp(complex(0.0, 2*np.pi) * freqs * bins / float(arr.size))
    return np.fft.irfft(phasor * np.fft.rfft(arr), arr.size)

def DM_delay(DM, freq, freq2=np.inf, P=None):
    """
    Calculates the delay of emitted frequency freq [MHz] from
    dispersion measure DM [cm**-3 pc] relative to freq2 [default=inf].
    If a period P [s] is provided, the delay is returned in phase [rot],
    otherwise in seconds.
    """
    delay = Dconst * DM * ((freq**-2.0) - (freq2**-2.0))
    if P:
        return delay / P
    else:
        return delay

def phase_transform(phi, DM, nu_ref1=np.inf, nu_ref2=np.inf, P=None, mod=True):
    """
    From nu_ref1 --> nu_ref2
    Default behavior is for P=1.0, i.e. transform delays [sec]
    """
    if P is None: P = 1.0
    phi_prime =  phi + (Dconst * DM * P**-1 * (nu_ref2**-2.0 - nu_ref1**-2.0))
    if mod:
        #phi_prime %= 1
        phi_prime = np.where(abs(phi_prime) >= 0.5, phi_prime % 1, phi_prime)
        phi_prime = np.where(phi_prime >= 0.5, phi_prime - 1.0, phi_prime)
    return phi_prime

def guess_fit_freq(freqs, SNRs=None):
    """
    Returns an estimate of an "optimal" frequency for fitting DM and phase
    in the sense that it minimizes the covariance.
    Intuited: "center of mass" where the weight in a given channel is given by:
    SNR/freq**2
    Default SNRs are ones.
    """
    nu0 = (freqs.min() + freqs.max()) * 0.5
    if SNRs is None:
        SNRs = np.ones(len(freqs))
    diff = np.sum((freqs - nu0) * SNRs * freqs**-2) / np.sum(SNRs * freqs**-2)
    return nu0 + diff

def doppler_correct_freqs(freqs, doppler_factor):
    """
    Input topocentric frequencies, output barycentric frequencies.
    doppler_factor = nu_source / nu_observed = sqrt( (1+beta) / (1-beta)),
    for beta = v/c, and v is positive for increasing source distance.
    NB: PSRCHIVE defines doppler_factor as the inverse of the above.
    """
    return doppler_factor * freqs

def calculate_TOA(epoch, P, phi, DM=0.0, nu_ref1=np.inf, nu_ref2=np.inf):
    """
    Calculates TOA given epoch [PSRCHIVE MJD], period P [sec], and phase
    offset phi [rot].  If phi was measured w.r.t. nu_ref1 [MHZ], providing
    DM [cm**-3 pc], nu_ref1 and nu_ref2 [MHz] will calculate the TOA w.r.t
    nu_ref2.
    """
    #Phase conversion (hopefully, the signs are correct)...
    #The pre-Doppler corrected DM must be used
    phi_prime = phase_transform(phi, DM, nu_ref1, nu_ref2, P)
    TOA = epoch + pr.MJD((phi_prime * P) / (3600 * 24.))
    return TOA

def load_data(filename, dedisperse=False, dededisperse=False, tscrunch=False,
        pscrunch=False, fscrunch=False, rm_baseline=True, flux_prof=False,
        norm_weights=True, quiet=False):
    """
    Will read and return data using PSRCHIVE.
    The returned archive is 'refreshed'.
    Perhaps should have default to not returning arch (poten. memory problem)
    """
    #Load archive
    arch = pr.Archive_load(filename)
    source = arch.get_source()
    if not quiet:
        print "\nReading data from %s on source %s..."%(filename, source)
    #De/dedisperse?
    if dedisperse: arch.dedisperse()
    if dededisperse: arch.dededisperse()
    DM = arch.get_dispersion_measure()
    #Maybe use better baseline subtraction??
    if rm_baseline: arch.remove_baseline()
    #tscrunch?
    if tscrunch: arch.tscrunch()
    nsub = arch.get_nsubint()
    #pscrunch?
    if pscrunch: arch.pscrunch()
    state = arch.get_state()
    npol = arch.get_npol()
    #fscrunch?
    if fscrunch: arch.fscrunch()
    #Nominal "center" of the band, but not necessarily
    nu0 = arch.get_centre_frequency()
    #For the negative BW cases.  Good fix
    #bw = abs(arch.get_bandwidth())
    bw = arch.get_bandwidth()
    nchan = arch.get_nchan()
    #Centers of frequency channels
    freqs = np.array([arch.get_Integration(0).get_centre_frequency(ichan) for
        ichan in xrange(nchan)])
    #Again, for the negative BW cases.  Good fix?
    #freqs.sort()
    #By-hand frequency calculation, equivalent to above from PSRCHIVE for
    #native resolution
    #chanwidth = bw / nchan
    #lofreq = nu0 - (bw/2)
    #freqs = np.linspace(lofreq + (chanwidth/2.0), lofreq + bw -
    #        (chanwidth/2.0), nchan)
    nbin = arch.get_nbin()
    #Centers of phase bins
    phases = np.linspace(0.0 + (nbin*2)**-1, 1.0 - (nbin*2)**-1, nbin)
    #These are NOT the bin centers...
    #phases = np.arange(nbin, dtype='d') / nbin
    #Get data
    #PSRCHIVE indices [subint:pol:chan:bin]
    subints = arch.get_data()
    Ps = np.array([arch.get_Integration(isub).get_folding_period() for isub in
        xrange(nsub)],dtype=np.double)
    epochs = [arch.get_Integration(isub).get_epoch() for isub in xrange(nsub)]
    #Get weights
    weights = arch.get_weights()
    weights_norm = np.where(weights == 0.0, np.zeros(weights.shape),
            np.ones(weights.shape))
    okisub = np.compress(weights_norm.mean(axis=1), np.arange(nsub))
    okichan = np.compress(weights_norm.mean(axis=0), np.arange(nchan))
    #np.einsum is AWESOME
    masks = np.einsum('ij,k', weights_norm, np.ones(nbin))
    masks = np.einsum('j,ikl', np.ones(npol), masks)
    #These are the data free of zapped channels and subints
    subintsxs = [np.compress(weights_norm[isub], subints[isub], axis=1) for
            isub in xrange(nsub)]
    #The channel center frequencies for the non-zapped subints
    freqsxs = [np.compress(weights_norm[isub], freqs) for isub in xrange(nsub)]
    SNRs = np.zeros([nsub, npol, nchan])
    for isub in range(nsub):
        for ipol in range(npol):
            for ichan in range(nchan):
                SNRs[isub, ipol, ichan] = \
                        arch.get_Integration(isub).get_Profile(ipol,
                                ichan).snr()
    noise_stds = np.array([arch.get_Integration(isub).baseline_stats()[1]**0.5
        for isub in xrange(nsub)])
    #The rest is now ignoring npol...
    arch.pscrunch()
    #Estimate noise
    #noise_stds = np.array([get_noise(subints[isub,0]) for isub in xrange(
    #nsub)])
    if flux_prof:
        #Flux profile
        #The below is about equal to bscrunch to ~6 places
        arch.dedisperse()
        arch.tscrunch()
        flux_prof = arch.get_data().mean(axis=3)[0][0]
        #Non-zapped data
        flux_profx = np.compress(arch.get_weights()[0], flux_prof)
    else:
        flux_prof = np.array([])
        flux_profx = np.array([])
    #Get pulse profile
    arch.tscrunch()
    arch.fscrunch()
    prof = arch.get_data()[0,0,0]
    prof_noise = arch.get_Integration(0).baseline_stats()[1][0,0]**0.5
    prof_SNR = arch.get_Integration(0).get_Profile(0,0).snr()
    #Number unzapped channels, subints
    nchanx = int(round(np.mean([subintsxs[isub].shape[1] for isub in
        xrange(nsub)])))
    nsubx = int(np.compress([subintsxs[isub].shape[1] for isub in
        xrange(nsub)], np.ones(nsub)).sum())
    if not quiet:
        P = arch.get_Integration(0).get_folding_period()*1000.0
        print "\tP [ms]             = %.1f\n\
        DM [cm**-3 pc]     = %.6f\n\
        center freq. [MHz] = %.4f\n\
        bandwidth [MHz]    = %.1f\n\
        # bins in prof     = %d\n\
        # channels         = %d\n\
        # chan (mean)      = %d\n\
        # subints          = %d\n\
        # unzapped subint  = %d\n\
        pol'n state        = %s\n"%(P, DM, nu0, abs(bw), nbin, nchan, nchanx,
                nsub, nsubx, state)
    #Returns refreshed arch; could be changed...
    arch.refresh()
    if norm_weights:
        weights = weights_norm
    #Return getitem/attribute-accessible class!
    data = DataBunch(arch=arch, bw=bw, DM=DM, epochs=epochs, filename=filename,
            flux_prof=flux_prof, flux_profx=flux_profx, freqs=freqs,
            freqsxs=freqsxs, masks=masks, nbin=nbin, nchan=nchan,
            nchanx=nchanx, noise_stds=noise_stds, nsub=nsub, nsubx=nsubx,
            nu0=nu0, okichan=okichan, okisub=okisub, phases=phases, prof=prof,
            prof_noise=prof_noise, prof_SNR=prof_SNR, Ps=Ps, SNRs=SNRs,
            source=source, state=state, subints=subints, subintsxs=subintsxs,
            weights=weights)
    return data

def unpack_dict(data):
    """
    This does not work yet; just for reference...
    Dictionary has to be named 'data'...
    """
    for key in data.keys():
        exec(key + " = data['" + key + "']")

def write_model(filename, name, nu_ref, model_params, fit_flags, append=False,
        quiet=False):
    """
    """
    if append:
        outfile = open(filename, "a")
    else:
        outfile = open(filename, "w")
    outfile.write("MODEL  %s\n"%name)
    outfile.write("FREQ   %.4f\n"%nu_ref)
    outfile.write("DC     %.8f  %d\n"%(model_params[0], fit_flags[0]))
    outfile.write("TAU    %.8f  %d\n"%(model_params[1], fit_flags[1]))
    ngauss = (len(model_params) - 2) / 6
    for igauss in xrange(ngauss):
        comp = model_params[(2 + igauss*6):(8 + igauss*6)]
        fit_comp = fit_flags[(2 + igauss*6):(8 + igauss*6)]
        line = (igauss + 1, ) + tuple(np.array(zip(comp, fit_comp)).ravel())
        outfile.write("COMP%02d %1.8f  %d  % 10.8f  %d  % 10.8f  %d  % 10.8f  %d  % 12.8f  %d  % 12.8f  %d\n"%line)
    outfile.close()
    if not quiet: print "%s written."%filename

def read_model(modelfile, phases=None, freqs=None, P=None, quiet=False):
    """
    """
    if phases is None and freqs is None:
        read_only = True
    else:
        read_only = False
    ngauss = 0
    comps = []
    modeldata = open(modelfile, "r").readlines()
    for line in modeldata:
        info = line.split()
        try:
            if info[0] == "MODEL":
                name = info[1]
            elif info[0] == "FREQ":
                nu_ref = float(info[1])
            elif info[0] == "DC":
                dc = float(info[1])
                fit_dc = int(info[2])
            elif info[0] == "TAU":
                tau = float(info[1])
                fit_tau = int(info[2])
            elif info[0][:4] == "COMP":
                comps.append(line)
                ngauss += 1
            else:
                pass
        except IndexError:
            pass
    params = np.zeros(ngauss*6 + 2)
    fit_flags = np.zeros(len(params))
    params[0] = dc
    params[1] = tau
    fit_flags[0] = fit_dc
    fit_flags[1] = fit_tau
    for igauss in xrange(ngauss):
        comp = map(float, comps[igauss].split()[1::2])
        fit_comp = map(int, comps[igauss].split()[2::2])
        params[2 + igauss*6 : 8 + (igauss*6)] = comp
        fit_flags[2 + igauss*6 : 8 + (igauss*6)] = fit_comp
    if not read_only:
        nbin = len(phases)
        nchan = len(freqs)
        if params[1] != 0:
            if P is None:
                print "Need period P for non-zero scattering value TAU."
                return 0
            else:
                params[1] *= nbin / P
        model = gen_gaussian_portrait(params, phases, freqs, nu_ref)
    if not quiet and not read_only:
        print "Model Name: %s"%name
        print "Made %d component model with %d profile bins,"%(ngauss, nbin)
        if len(freqs) != 1:
            bw = (freqs[-1] - freqs[0]) + ((freqs[-1] - freqs[-2]))
        else:
            bw = 0.0
        print "%d frequency channels, %.0f MHz bandwidth, centered near %.3f MHz,"%(nchan, abs(bw), freqs.mean())
        print "with model parameters referenced at %.3f MHz."%nu_ref
    if read_only:
        return name, nu_ref, ngauss, params, fit_flags
    else:
        return name, ngauss, model

def file_is_ASCII(filename):
    """
    """
    from os import popen4
    cmd = "file -L %s"%filename
    i,o = popen4(cmd)
    line = o.readline().split()
    try:
        line.index("ASCII")
        return True
    except ValueError:
        try:
            line.index("FITS")
            return False
        except ValueError:
            pass

def write_archive(data, ephemeris, freqs, nu0=None, bw=None,
        outfile="pparchive.fits", tsub=1.0, start_MJD=None, weights=None,
        dedispersed=False, state="Coherence", obs="GBT", quiet=False):
    """
    Mostly written by PBD.
    Takes dedispersed data, please.  If dedispersed=True, will save archive
    this way.
    """
    nsub, npol, nchan, nbin = data.shape
    if nu0 is None:
        #This is off by a tiny bit...
        nu0 = freqs.mean()
    if bw is None:
        #This is off by a tiny bit...
        bw = (freqs.max() - freqs.min()) + abs(freqs[1] - freqs[0])
    #Phase bin centers
    phases = np.linspace(0.0 + (nbin*2)**-1, 1.0 - (nbin*2)**-1, nbin)
    #Create the Archive instance.
    #This is kind of a weird hack, if we create a PSRFITS
    #Archive directly, some header info does not get filled
    #in.  If we instead create an ASP archive, it will automatically
    #be (correctly) converted to PSRFITS when it is unloaded...
    arch = pr.Archive_new_Archive("ASP")
    arch.resize(nsub, npol, nchan, nbin)
    try:
        import parfile
        par = parfile.psr_par(ephemeris)
        try:
            PSR = par.PSR
        except AttributeError:
            PSR = par.PSRJ
        DECJ = par.DECJ
        RAJ = par.RAJ
        DM = par.DM
    except ImportError:
        parfile = open(ephemeris,"r").readlines()
        for xx in xrange(len(parfile)):
            param = parfile[xx].split()
            if len(param) == 0:
                pass
            elif param[0] == ("PSR" or "PSRJ"):
                PSR = param[1]
            elif param[0] == "RAJ":
                RAJ = param[1]
            elif param[0] == "DECJ":
                DECJ = param[1]
            elif param[0] == "DM":
                DM = float(param[1])
            else:
                pass
    #Dec needs to have a sign for the following sky_coord call
    if (DECJ[0] != '+' and DECJ[0] != '-'):
        DECJ = "+" + DECJ
    arch.set_dispersion_measure(DM)
    arch.set_source(PSR)
    arch.set_coordinates(pr.sky_coord(RAJ + DECJ))
    #Set some other stuff
    arch.set_centre_frequency(nu0)
    arch.set_bandwidth(bw)
    arch.set_telescope(obs)
    if npol==4:
        arch.set_state(state) #Could also do 'Stokes' here
    #Fill in some subintegration attributes
    if start_MJD is None:
        start_MJD = pr.MJD(50000, 0, 0.0)
    epoch = start_MJD
    epoch += tsub/2.0
    for subint in arch:
        subint.set_epoch(epoch)
        subint.set_duration(tsub)
        epoch += tsub
        for ichan in xrange(nchan):
            subint.set_centre_frequency(ichan, freqs[ichan])
    #Fill in polycos
    arch.set_ephemeris(ephemeris)
    #Now finally, fill in the data!
    arch.set_dedispersed(True)
    arch.dedisperse()
    if weights is None: weights = np.ones([nsub, nchan])
    isub = 0
    for subint in arch:
        for ipol in xrange(npol):
            for ichan in xrange(nchan):
                subint.set_weight(ichan, weights[isub, ichan])
                prof = subint.get_Profile(ipol, ichan)
                prof.get_amps()[:] = data[isub, ipol, ichan]
        isub += 1
    if not dedispersed: arch.dededisperse()
    arch.unload(outfile)
    if not quiet: print "\nUnloaded %s.\n"%outfile

def make_fake_pulsar(modelfile, ephemeris, outfile="fake_pulsar.fits", nsub=1,
        npol=1, nchan=512, nbin=1048, nu0=1500.0, bw=800.0, tsub=300.0,
        phase=0.0, dDM=0.0, start_MJD=None, weights=None, noise_std=1.0,
        scale=1.0, dedisperse=False, t_scat=0.0, alpha=scattering_alpha,
        scint=False, state="Coherence", obs="GBT", quiet=False):
    """
    Mostly written by PBD.
    'phase' [rot] is an arbitrary rotation to all subints, with respect to nu0.
    'dDM' [cm**-3 pc] is an additional DM to what is given in ephemeris.
    Dispersion occurs at infinite frequency.
    """
    chanwidth = bw / nchan
    lofreq = nu0 - (bw/2)
    #Channel frequency centers
    freqs = np.linspace(lofreq + (chanwidth/2.0), lofreq + bw -
            (chanwidth/2.0), nchan)
    #Phase bin centers
    phases = np.linspace(0.0 + (nbin*2)**-1, 1.0 - (nbin*2)**-1, nbin)
    #Channel noise
    try:
        if len(noise_std) != nchan:
            print "\nlen(noise_std) != nchan\n"
            return 0
    except TypeError:
        noise_std = noise_std * np.ones(nchan)
    #Create the Archive instance.
    #This is kind of a weird hack, if we create a PSRFITS
    #Archive directly, some header info does not get filled
    #in.  If we instead create an ASP archive, it will automatically
    #be (correctly) converted to PSRFITS when it is unloaded...
    arch = pr.Archive_new_Archive("ASP")
    arch.resize(nsub,npol,nchan,nbin)
    try:
        import parfile
        par = parfile.psr_par(ephemeris)
        try:
            PSR = par.PSR
        except AttributeError:
            PSR = par.PSRJ
        DECJ = par.DECJ
        RAJ = par.RAJ
        P0 = par.P0
        DM = par.DM
    except ImportError:
        parfile = open(ephemeris,"r").readlines()
        for xx in xrange(len(parfile)):
            param = parfile[xx].split()
            if param[0] == ("PSR" or "PSRJ"):
                PSR = param[1]
            elif param[0] == "RAJ":
                RAJ = param[1]
            elif param[0] == "DECJ":
                DECJ = param[1]
            elif param[0] == "F0":
                P0 = float(param[1])**-1
            elif param[0] == "P0":
                P0 = float(param[1])
            elif param[0] == "DM":
                DM = float(line[1])
            else:
                pass
    #Dec needs to have a sign for the following sky_coord call
    if (DECJ[0] != '+' and DECJ[0] != '-'):
        DECJ = "+" + DECJ
    arch.set_dispersion_measure(DM)
    arch.set_source(PSR)
    arch.set_coordinates(pr.sky_coord(RAJ + DECJ))
    #Set some other stuff
    arch.set_centre_frequency(nu0)
    arch.set_bandwidth(bw)
    arch.set_telescope(obs)
    if npol==4:
        arch.set_state(state) #Could also do 'Stokes' here
    #Fill in some subintegration attributes
    if start_MJD is None:
        start_MJD = pr.MJD(50000, 0, 0.0)
    epoch = start_MJD
    epoch += tsub/2.0
    for subint in arch:
        subint.set_epoch(epoch)
        subint.set_duration(tsub)
        epoch += tsub
        for ichan in xrange(nchan):
            subint.set_centre_frequency(ichan, freqs[ichan])
    #Fill in polycos
    arch.set_ephemeris(ephemeris)
    #Now finally, fill in the data!
    #NB the different pols are not realistic: same model, same noise_std
    name, ngauss, model = read_model(modelfile, phases, freqs, P0, quiet)
    if scint is not False:
        if scint is True:
            model = add_scintillation(model, random=True, nsin=2, amax=1.0,
                    wmax=5.0)
        else:
            model = add_scintillation(model, scint)
    #If wanting to use PSRCHIVE's rotation scheme, uncomment the dedisperse and
    #dededisperse lines, and set rotmodel = model.
    arch.set_dedispersed(False)
    arch.dededisperse()
    if weights is None: weights = np.ones([nsub, nchan])
    isub = 0
    for subint in arch:
        P = subint.get_folding_period()
        for ipol in xrange(npol):
            rotmodel = rotate_portrait(model, -phase, -(DM+dDM), P, freqs, nu0)
            #rotmodel = model
            if t_scat:
                sk = scattering_kernel(t_scat, nu0, freqs, phases, P,
                        alpha=alpha)
                rotmodel = add_scattering(rotmodel, sk, repeat=3)
            for ichan in xrange(nchan):
                subint.set_weight(ichan, weights[isub, ichan])
                prof = subint.get_Profile(ipol, ichan)
                noise = noise_std[ichan]
                if noise:
                    prof.get_amps()[:] = rotmodel[ichan] + np.random.normal(
                            0.0, noise, nbin)
                    prof.get_amps()[:] *= scale
                else:
                    prof.get_amps()[:] = rotmodel[ichan]
                    prof.get_amps()[:] *= scale
        isub += 1
    if dedisperse: arch.dedisperse()
    arch.unload(outfile)
    if not quiet: print "\nUnloaded %s.\n"%outfile

def quick_add_archs(metafile, outfile, rotate=False, fiducial=0.5,
        quiet=False):
    """
    """
    from os import system
    datafiles = open(metafile, "r").readlines()
    datafiles = [datafiles[ifile][:-1] for ifile in xrange(len(datafiles))]
    for ifile in xrange(len(datafiles)):
        datafile = datafiles[ifile]
        data = load_data(datafile, dedisperse=True, dededisperse=False,
                tscrunch=True, pscrunch=True, fscrunch=False, rm_baseline=True,
                flux_prof=False, norm_weights=True, quiet=True)
        if ifile == 0:
            nchan = data.nchan
            nbin = data.nbin
            cmd = "cp %s %s"%(datafile, outfile)
            system(cmd)
            arch = pr.Archive_load(outfile)
            arch.set_dispersion_measure(0.0)
            arch.tscrunch()
            arch.pscrunch()
            totport = np.zeros([nchan, nbin])
            totweights = np.zeros(nchan)
        if rotate:
            maxbin = data.prof.argmax()
            rotport = rotate_portrait(data.subints[0,0], maxbin/float(nbin) -
                    fiducial)
        else:
            rotport = data.subints[0,0]
        totport += rotport
        #The below assumes equal weight to all files!
        totweights += data.weights[0]
        if not quiet: print "Added %s"%datafile
    totweights = np.where(totweights==0, 1, totweights)
    totport = np.transpose((totweights**-1) * np.transpose(totport))
    I = arch.get_Integration(0)
    for ichan in xrange(nchan):
        prof = I.get_Profile(0,ichan)
        prof.get_amps()[:] = totport[ichan]
    arch.dedisperse()
    arch.unload()
    print "\nUnloaded %s"%outfile

def write_princeton_TOA(TOA_MJDi, TOA_MJDf, TOA_err, nu_ref, dDM, obs='@',
        name=' ' * 13):
    """
    Ripped and altered from PRESTO

    Princeton Format

    columns     item
    1-1     Observatory (one-character code) '@' is barycenter
    2-2     must be blank
    16-24   Reference (not necessarily observing) frequency [MHz]
    25-44   TOA [MJD] (decimal point must be in column 30 or column 31)
    45-53   TOA uncertainty [us]
    69-78   DM correction [cm**-3 pc]
    """
    if nu_ref == np.inf: nu_ref = 0.0
    #Splice together the fractional and integer MJDs
    TOA = "%5d"%int(TOA_MJDi) + ("%.13f"%TOA_MJDf)[1:]
    #if dDM != 0.0:
    print obs + " %13s %8.3f %s %8.3f              %9.5f"%(name, nu_ref, TOA,
            TOA_err, dDM)
    #else:
    #    print obs + " %13s %8.3f %s %8.3f"%(name, nu_ref, TOA, TOA_err)

def show_portrait(port, phases=None, freqs=None, title=None, prof=True,
        fluxprof=True, rvrsd=False, colorbar=True, savefig=False,
        aspect="auto", interpolation="none", origin="lower", extent=None,
        **kwargs):
    """
    """
    nn = 2*3*5  #Need something divisible by 2,3,5...
    grid = gs.GridSpec(nn, nn)
    pi = 0
    fi = 0
    ci = 0
    if freqs is None:
        freqs = np.arange(len(port))
        ylabel = "Channel Number"
    else:
        ylabel = "Frequency [MHz]"
    if phases is None:
        phases = np.arange(len(port[0]))
        xlabel = "Bin Number"
    else:
        xlabel = "Phase [rot]"
    if rvrsd:
        freqs = freqs[::-1]
        port = port[::-1]
    if extent is None:
        extent = (phases[0], phases[-1], freqs[0], freqs[-1])
    if prof or fluxprof:
        weights = port.mean(axis=1)
    if prof:
        portx = np.compress(weights, port, axis=0)
        prof = portx.mean(axis=0)
        pi = 1
    if fluxprof:
        fluxprof = port.mean(axis=1)
        fluxprofx = np.compress(weights, fluxprof)
        freqsx = np.compress(weights, freqs)
        fi = 1
    if colorbar: ci = 1
    ax1 = plt.subplot(grid[(pi*nn/6):, (fi*nn/6):])
    im = plt.imshow(port, aspect=aspect, origin=origin, extent=extent,
        interpolation=interpolation, **kwargs)
    if colorbar: plt.colorbar(im, ax=ax1, use_gridspec=False)
    plt.xlabel(xlabel)
    if not fi:
        plt.ylabel(ylabel)
    else:
        #ytklbs = ax1.get_yticklabels()
        ax1.set_yticklabels(())
    if not pi:
        if title: plt.title(title)
    if pi:
        if ci:
            ax2 = plt.subplot(grid[:(pi*nn/6), (fi*nn/6):((3+fi+ci)*nn) /
                (4+fi+ci)])
        else:
            ax2 = plt.subplot(grid[:(pi*nn/6), (fi*nn/6):])
        ax2.plot(phases, prof, 'k-')
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels(())
        ax2.set_yticks([0, round(prof.max() / 2, 1), round(prof.max(), 1)])
        plt.xlim(phases.min(), phases.max())
        rng = prof.max() - prof.min()
        plt.ylim(prof.min() - 0.03*rng, prof.max() + 0.05*rng)
        plt.ylabel("Flux Units")
        if title: plt.title(title)
    if fi:
        ax3 = plt.subplot(grid[(pi*nn/6):, :(fi*nn/6)])
        ax3.plot(fluxprofx, freqsx, 'kx')
        ax3.set_xticks([0, round(fluxprofx.max() / 2, 2),
            round(fluxprofx.max(), 2)])
        rng = fluxprofx.max() - fluxprofx.min()
        plt.xlim(fluxprofx.max() + 0.03*rng, fluxprofx.min() - 0.01*rng)
        plt.xlabel("Flux Units")
        ax3.set_yticks(ax1.get_yticks())
        #ax3.set_yticklabels(ytklbs)
        plt.ylim(freqs[0], freqs[-1])
        plt.ylabel(ylabel)
    #if title: plt.suptitle(title)
    #plt.tight_layout(pad = 1.0, h_pad=0.0, w_pad=0.0)
    if savefig:
        plt.savefig(savefig, format='png')
        plt.close()
    else:
        plt.show()

def show_residual_plot(port, model, resids=None, phases=None, freqs=None,
        titles=(None,None,None), rvrsd=False, colorbar=True, savefig=False,
        aspect="auto", interpolation="none", origin="lower", extent=None,
        **kwargs):
    """
    """
    mm = 6
    nn = (2*mm) + (mm/3)
    grid = gs.GridSpec(nn, nn)
    if freqs is None:
        freqs = np.arange(len(port))
        ylabel = "Channel Number"
    else:
        ylabel = "Frequency [MHz]"
    if phases is None:
        phases = np.arange(len(port[0]))
        xlabel = "Bin Number"
    else:
        xlabel = "Phase [rot]"
    if rvrsd:
        freqs = freqs[::-1]
        port = port[::-1]
        model = model[::-1]
    if extent is None:
        extent = (phases[0], phases[-1], freqs[0], freqs[-1])
    ax1 = plt.subplot(grid[:mm, :mm])
    im = plt.imshow(port, aspect=aspect, origin=origin, extent=extent,
            interpolation=interpolation, **kwargs)
    if colorbar: plt.colorbar(im, ax=ax1, use_gridspec=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if titles[0] is None:
        plt.title("")
    else:
        plt.title(titles[0])
    ax2 = plt.subplot(grid[:mm, -mm:])
    im = plt.imshow(model, aspect=aspect, origin=origin, extent=extent,
            interpolation=interpolation, **kwargs)
    if colorbar: plt.colorbar(im, ax=ax2, use_gridspec=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if titles[1] is None:
        plt.title("")
    else:
        plt.title(titles[1])
    ax3 = plt.subplot(grid[-mm:, :mm])
    if resids is None: resids = port - model
    im = plt.imshow(resids, aspect=aspect, origin=origin, extent=extent,
            interpolation=interpolation, **kwargs)
    if colorbar: plt.colorbar(im, ax=ax3, use_gridspec=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if titles[2] is None:
        plt.title("")
    else:
        plt.title(titles[2])
    ax4 = plt.subplot(grid[-mm:, -mm:])
    weights = port.mean(axis=1)
    portx = np.compress(weights, port, axis=0)
    residsx = np.compress(weights, resids, axis=0)
    text =  "Residuals mean ~ %.2e\nResiduals std ~ %.2e\nData std ~ %.2e"%(
            residsx.mean(), residsx.std(), np.median(get_noise(portx,
                chans=True)))
    ax4.text(0.5, 0.5, text, ha="center", va="center")
    ax4.set_xticklabels(())
    ax4.set_xticks(())
    ax4.set_yticklabels(())
    ax4.set_yticks(())
    #plt.tight_layout(pad=1.0, h_pad=-10.0, w_pad=-10.0)
    if savefig:
        plt.savefig(savefig, format='png')
        plt.close()
    else:
        plt.show()

def plot_lognorm(mu, tau, lo=0.0, hi=5.0, npts=500, plot=1, show=0):
    """
    """
    import pymc as pm
    pts = np.empty(npts)
    xs = np.linspace(lo, hi, npts)
    for ii in xrange(npts):
        pts[ii] = np.exp(pm.lognormal_like(xs[ii], mu, tau))
    if plot:
        plt.plot(xs, pts)
    if show:
        plt.show()
    return xs, pts

def plot_gamma(alpha, beta, lo=0.0, hi=5.0, npts=500, plot=1, show=0):
    """
    """
    import pymc as pm
    pts = np.empty(npts)
    xs = np.linspace(lo, hi, npts)
    for ii in xrange(npts):
        pts[ii] = np.exp(pm.gamma_like(xs[ii], alpha, beta))
    if plot:
        plt.plot(xs, pts)
    if show:
        plt.show()
    return xs, pts
