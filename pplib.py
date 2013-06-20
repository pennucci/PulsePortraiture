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
             "vla":"c", "ncy":"f", "eff":"g", "jbdfb":"q", "wsrt":"i"}

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

class DataBunch(dict):
    """
    This class is a great little recipe!  Creates a simple class instance
    db = DataBunch(a=1, b=2,....) that has attributes a and b, which are
    callable and update-able using either syntax db.a or db['a'].
    """
    def __init__(self, **kwds):
        dict.__init__(self, kwds)
        self.__dict__ = self

def gaussian_profile(N, loc, wid, norm=False, abs_wid=False, zeroout=True):
    """
    Taken and tweaked from SMR's pygaussfit.py

    Return a gaussian pulse profile with 'N' bins and peak amplitude of 1.
    If norm=True, return the profile such that the integrated density = 1.
        'N' = the number of points in the profile
        'loc' = the pulse phase location (0-1)
        'wid' = the gaussian pulse's full width at half-max (FWHM)
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
        return np.zeros(N, 'd')
    elif wid < 0.0 and zeroout:
        return np.zeros(N, 'd')
    elif wid < 0.0 and not zeroout:
        pass
    else:
        return 0
    sigma = wid / (2 * np.sqrt(2 * np.log(2)))
    mean = loc % 1.0
    locval = np.arange(N, dtype='d') / float(N)
    if (mean < 0.5):
        locval = np.where(np.greater(locval, mean + 0.5), locval - 1.0, locval)
    else:
        locval = np.where(np.less(locval, mean - 0.5), locval + 1.0, locval)
    try:
        zs = (locval - mean) / sigma
        okzinds = np.compress(np.fabs(zs) < 20.0, np.arange(N))   #Why 20?
        okzs = np.take(zs, okzinds)
        retval = np.zeros(N, 'd')
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
        return np.zeros(N, 'd')

def gen_gaussian_profile(params, N):
    """
    Taken and tweaked from SMR's pygaussfit.py

    Return a model of a DC-component + ngauss gaussian functions.
    params is a sequence of 1 + (ngauss*3) values; the first value is the DC
    component.  Each remaining group of three represents the gaussians
    loc (0-1), wid (i.e. FWHM) (0-1), and amplitude (>0.0).  N is the number of
    points in the model.

    cf. gaussian_profile(...)
    """
    ngauss = (len(params) - 1) / 3
    model = np.zeros(N, dtype='d') + params[0]
    for igauss in xrange(ngauss):
        loc, wid, amp = params[(1 + igauss*3):(4 + igauss*3)]
        model += amp * gaussian_profile(N, loc, wid)
    return model

def gen_gaussian_portrait(params, phases, freqs, nu_ref):
    """
    Build the gaussian model portrait based on params.

    params is an array of 1 + (ngauss*6) values; the first value is the DC
    component.  Each remaining group of six represent the gaussians loc (0-1),
    linear slope in loc, wid (i.e. FWHM) (0-1), linear slope in wid,
    amplitude (>0,0), and spectral index alpha (no implicit negative).
    phases is the array of phase values (will pass nbin to
    gen_gaussian_profile).
    freqs in the array of frequencies at which to calculate the model.
    nu_ref [MHz] is the frequency to which the locs, wids, and amps reference.

    cf. gaussian_profile(...), gen_gaussian_profile(...)
    """
    refparams = np.array([params[0]]+ list(params[1::2]))
    locparams = params[2::6]
    widparams = params[4::6]
    ampparams = params[6::6]
    ngauss = len(refparams[1::3])
    nbin = len(phases)
    nchan = len(freqs)
    gport = np.empty([nchan, nbin])
    gparams = np.empty([nchan, len(refparams)])
    #DC term
    gparams[:,0] = refparams[0]
    #Locs
    gparams[:,1::3] = np.outer(freqs - nu_ref, locparams) + np.outer(np.ones(
        nchan), refparams[1::3])
    #Wids
    gparams[:,2::3] = np.outer(freqs - nu_ref, widparams) + np.outer(np.ones(
        nchan), refparams[2::3])
    #Amps
    gparams[:,3::3] = np.exp(np.outer(np.log(freqs) - np.log(nu_ref),
        ampparams) + np.outer(np.ones(nchan), np.log(refparams[3::3])))
    #Amps; I am unsure why I needed this fix at some point
    #gparams[:, 0::3][:, 1:] = np.exp(np.outer(np.log(freqs) - np.log(nu_ref),
    #    ampparams) + np.outer(np.ones(nchan), np.log(refparams[0::3][1:])))
    for ichan in xrange(nchan):
        #Need to contrain so values don't go negative, etc., which is currently
        #done in gaussian_profile
        gport[ichan] = gen_gaussian_profile(gparams[ichan], nbin)
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
        errs=None):
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
        nu_ref)) / errs)
    return deviates

def fit_phase_shift_function(phase, model=None, data=None, err=None):
    """
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
        errs=None, P=None, freqs=None, nu_ref=np.inf, transform=False):
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
    if transform:
        ii = 2
    else:
        ii = 1
    while(ii):
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
        ii -= 1
        if ii:
            nu_zero = (W_n.sum() / np.sum(W_n * freqs**-2))**0.5
            phase = phase_transform(phase, params[1], nu_ref, nu_zero, P)
            nu_ref = nu_zero
    if transform:
        return (np.array([d2_phi, d2_DM, d2_cross]), nu_zero)
    else:
        return (np.array([d2_phi, d2_DM, d2_cross]), nu_ref)

def estimate_portrait(phase, DM, data, scales, P, freqs, nu_ref=np.inf):
    #here, all vars have additional epoch-index except nu_ref
    #i.e. all have to be arrays of at least len 1; errs are precision
    """
    An early attempt to make a '2-D' version of PBD's autotoa.  That is, to
    iterate over all epochs of data portraits to build a non-gaussian model
    that can be smoothed.

    <UNDER CONSTRUCTION>

    cf. PBD's autotoa, PhDT
    """
    #Next lines should be done just as in fit_portrait
    dFFT = fft.rfft(data, axis=2)
    #Below is Precision FIX
    unnorm_errs = np.real(dFFT[:, :, -len(dFFT[0,0])/4:]).std(axis=2)**-2.0
    #norm_dFFT = np.transpose((unnorm_errs**0.5) * np.transpose(dFFT))
    #norm_errs = np.real(norm_dFFT[:, :, -len(norm_dFFT[0,0])/4:]
    #        ).std(axis=2)**-2.
    errs = unnorm_errs
    D = Dconst * DM / P
    freqs2 = freqs**-2.0 - nu_ref**-2.0
    phiD = np.outer(D, freqs2)
    phiprime = np.outer(phase, np.ones(len(freqs))) + phiD
    weight = np.sum(pow(scales, 2.0) * errs, axis=0)**-1
    phasor = np.array([np.exp(2.0j * np.pi * kk * phiprime) for kk in xrange(
        len(dFFT[0,0]))]).transpose(1,2,0)
    p = np.sum(np.transpose(np.transpose(scales * errs) * np.transpose(phasor *
        dFFT)), axis=0)
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
    #Check Normalization below
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

def find_kc(prof, noise):
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
    nparam = len(init_params)
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

def fit_gaussian_profile(data, init_params, errs, quiet=True):
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
    ngauss = (len(init_params) - 1) / 3
    #Generate the parameter structure
    params = lm.Parameters()
    for ii in xrange(nparam):
        if ii == 0:
            params.add('dc', init_params[ii], vary=True, min=None, max=None,
                    expr=None)
        elif ii in range(nparam)[1::3]:
            params.add('loc%s'%str((ii-1)/3 + 1), init_params[ii], vary=True,
                    min=None, max=None, expr=None)
        elif ii in range(nparam)[2::3]:
            params.add('wid%s'%str((ii-1)/3 + 1), init_params[ii], vary=True,
                    min=0.0, max=None, expr=None)
        elif ii in range(nparam)[3::3]:
            params.add('amp%s'%str((ii-1)/3 + 1), init_params[ii], vary=True,
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
        nu_ref, quiet=True):
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
    ngauss = (len(init_params) - 1) / 6
    #Generate the parameter structure
    params = lm.Parameters()
    for ii in xrange(nparam):
        if ii == 0:         #DC, not limited
            params.add('dc', init_params[ii], vary=bool(fit_flags[ii]),
                    min=None, max=None, expr=None)
        elif ii%6 == 1:     #loc limits
            params.add('loc%s'%str((ii-1)/6 + 1), init_params[ii],
                    vary=bool(fit_flags[ii]), min=None, max=None, expr=None)
        elif ii%6 == 2:     #loc slope limits
            params.add('m_loc%s'%str((ii-1)/6 + 1), init_params[ii],
                    vary=bool(fit_flags[ii]), min=None, max=None, expr=None)
        elif ii%6 == 3:     #wid limits, limited by 0
            params.add('wid%s'%str((ii-1)/6 + 1), init_params[ii],
                    vary=bool(fit_flags[ii]), min=0.0, max=None, expr=None)
        elif ii%6 == 4:     #wid slope limits
            params.add('m_wid%s'%str((ii-1)/6 + 1), init_params[ii],
                    vary=bool(fit_flags[ii]), min=None, max=None, expr=None)
        elif ii%6 == 5:     #amp limits, limited by 0
            params.add('amp%s'%str((ii-1)/6 + 1), init_params[ii],
                    vary=bool(fit_flags[ii]), min=0.0, max=None, expr=None)
        elif ii%6 == 0:     #amp index limits
            params.add('alpha%s'%str((ii-1)/6 + 1), init_params[ii],
                    vary=bool(fit_flags[ii]), min=None, max=None, expr=None)
        else:
            print "Undefined index %d."%ii
            sys.exit()
    other_args = {'data':data, 'errs':errs, 'phases':phases, 'freqs':freqs,
            'nu_ref':nu_ref}
    #Now fit it
    results = lm.minimize(fit_gaussian_portrait_function, params,
            kws=other_args)
    fitted_params = np.array([param.value for param in
        results.params.itervalues()])
    dof = results.nfree
    chi_sq = results.chisqr
    redchi_sq = results.redchi
    residuals = results.residual
    model = gen_gaussian_portrait(fitted_params, phases, freqs, nu_ref)
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

def fit_portrait(data, model, init_params, P, freqs, nu_ref=np.inf,
        bounds=[(None, None), (None, None)], id=None, quiet=True):
    """
    Fits a phase-frequency pulse portrait to data for a phase offset and
    dispersion measure (DM) by minimizing the calculated chi**2 function using
    scipy.optimize's truncated Netownian algorithm.
    Returns the best-fit phase and DM values, an nchan array of scaling
    amplitudes, a 2 + nchan array of parameter error estimates, the
    zero-covariance frequency [MHz] used to estimate the errors, the covariance
    between the phase and DM parameters (should be close to zero), the reduced
    chi**2 value, the duration of the fit, the number of function calls, and
    the fit's return code.

    data and model are both nchan x nbin phase-frequency portraits.
    init_params is the array containing the initial guesses for [phase, DM],
    where phase has units [rot] and DM has units [cm**-3 pc].
    P is the period [s] of the pulsar at the given epoch.
    freqs in the array of frequencies [MHz] at which to calculate the model.
    nu_ref is the frequency [MHz] that is designated to have zero delay from a
    non-zero dispersion measure.
    bounds is an array of two 2-tuples giving upper and lower bounds for the
    two parameters (default is no bounds).
    id is an option tag to identify this particular fit.
    quiet=True suppresses output [default].
    """
    #tau = precision = 1/variance
    #FIX Need to use better filtering instead of frac in get_noise
    #FIX get_noise is not right
    #errs = get_noise(data, tau=True, chans=True, fd=True, frac=4) 
    dFFT = fft.rfft(data, axis=1)
    mFFT = fft.rfft(model, axis=1)
    #Precision FIX
    unnorm_errs = np.real(dFFT[:, -len(dFFT[0])/4:]).std(axis=1)
    norm_dFFT = np.transpose(unnorm_errs * np.transpose(dFFT))
    norm_errs = np.real(norm_dFFT[:, -len(norm_dFFT[0])/4:]).std(axis=1)
    errs = unnorm_errs
    d = np.real(np.sum(np.transpose(errs**-2.0 * np.transpose(dFFT *
        np.conj(dFFT)))))
    p_n = np.real(np.sum(mFFT * np.conj(mFFT), axis=1))
    #BEWARE BELOW! Order matters!
    other_args = (mFFT, p_n, dFFT, errs, P, freqs, nu_ref)
    minimize = opt.minimize
    #fmin_tnc seems to work best, fastest
    method = 'TNC'
    start = time.time()
    results = minimize(fit_portrait_function, init_params, args=other_args,
            method=method, jac=fit_portrait_function_deriv, bounds=bounds,
            options={'disp':False})
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
            filenm = id[:-jj-1]
            sys.stderr.write(
                    "Fit failed with return code %d -- %s; %s subint %s subintx %s\n"%(results.status, rcstring, filenm, isub, isubx))
        else:
            sys.stderr.write(
                    "Fit failed with return code %d -- %s"%(results.status, rcstring))
    if not quiet and results.success is True:
        sys.stderr.write("Fit succeeded with return code %d -- %s\n"
                %(results.status, rcstring))
    #Curvature matrix = 1/2 2deriv of chi2 (cf. gregory sect 11.5)
    #Parameter errors are related to curvature matrix by **-0.5 
    hessian, nu_zero = fit_portrait_function_2deriv(np.array([phi, DM]),
        mFFT, p_n, dFFT, errs, P, freqs, nu_ref, True)
    param_errs = list(pow(0.5*hessian[:2], -0.5))
    DoF = len(data.ravel()) - (len(freqs) + 2)
    red_chi2 = (d + results.fun) / DoF
    scales = get_scales(data, model, phi, DM, P, freqs, nu_ref)
    #Errors on scales, if ever needed
    param_errs += list(pow(p_n / errs**2.0, -0.5))
    #The below should be changed to a DataBunch
    return (phi, DM, scales, np.array(param_errs), nu_zero, hessian[2],
            red_chi2, duration, nfeval, return_code)

def fit_phase_shift(data, model, bounds=[-0.5, 0.5]):
    """
    """
    dFFT = fft.rfft(data)
    mFFT = fft.rfft(model)
    #Substitute get_noise below when ready
    err = np.real(dFFT[-len(dFFT)/4:]).std()
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

def get_noise(data, frac=4, tau=False, chans=False, fd=False):
    #FIX: Make sure to use on portraits w/o zapped freq. channels
    #i.e. portxs
    #FIX: MAKE SIMPLER!!!
    #FIX: Implement k_max from wiener/brick-wall filter fit
    #FIX This is not right 
    """
    """
    shape = data.shape
    if len(shape) == 1:
        prof = data
    elif shape[0] == 1:
        prof = data[0]
    elif shape[1] == 1:
        prof = data[:,0]
    else: pass
    try:
        FFT = fft.rfft(prof)
        if fd:
            if tau: return np.std(np.real(FFT)[-len(FFT)/frac:])**-2
            #if tau: return (np.std(np.real(FFT)[-len(FFT)/frac:])**-2,
            #        np.std(np.imag(FFT)[-len(FFT)/frac:])**-2)
            else: return np.std(np.real(FFT)[-len(FFT)/frac:])
            #else: return (np.std(np.real(FFT)[-len(FFT)/frac:]), np.std(
            #    np.imag(FFT)[-len(FFT)/frac:]))
        else:
            #!!!CHECK NORMALIZATION BELOW
            pows = np.real(FFT * np.conj(FFT)) / len(prof)
            if tau:
                return (np.mean(pows[-len(pows)/frac:]))**-1
            else:
                return np.sqrt(np.mean(pows[-len(pows)/frac:]))
    except NameError:
        noise = np.zeros(len(data))
        if fd:
            for ichan in xrange(len(noise)):
                    prof = data[ichan]
                    FFT = fft.rfft(prof)
                    noise[ichan] = np.std(np.real(FFT)[-len(FFT)/frac:])
            if chans:
                if tau: return noise**-2
                else: return noise
            else:
                if tau:
                    #not statistically rigorous
                    return np.median(noise)**-2
                else:
                    return np.median(noise)
        else:
            for ichan in xrange(len(noise)):
                prof = data[ichan]
                FFT = fft.rfft(prof)
                #!!!CHECK NORMALIZATION BELOW
                pows = np.real(FFT * np.conj(FFT)) / len(prof)
                noise[ichan] = np.sqrt(np.mean(pows[-len(pows)/frac:]))
            if chans:
                if tau:
                    return noise**-2
                else:
                    return noise
            else:
                if tau:
                    #not statistically rigorous
                    return np.median(noise)**-2
                else:
                    return np.median(noise)

def get_scales(data, model, phase, DM, P, freqs, nu_ref=np.inf):
    """
    """
    scales = np.zeros(len(freqs))
    dFFT = fft.rfft(data, axis=1)
    mFFT = fft.rfft(model, axis=1)
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

    ***Need to change this convention***
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

def phase_transform(phi, DM, freq1=np.inf, freq2=np.inf, P=None, mod=True):
    """
    """
    if P is None: P= 1.0
    phi_prime =  phi + (Dconst * DM * P**-1 * (freq2**-2.0 - freq1**-2.0))
    if mod:
        phi_prime %= 1
        if phi_prime >= 0.5:
            phi_prime -= 1.0
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

def load_data(filenm, dedisperse=False, dededisperse=False, tscrunch=False,
        pscrunch=False, fscrunch=False, rm_baseline=True, flux_prof=False,
        norm_weights=True, quiet=False):
    """
    Will read and return data using PSRCHIVE.
    The returned archive is 'refreshed'.
    Perhaps should have default to not returning arch (poten. memory problem)
    """
    #Load archive
    arch = pr.Archive_load(filenm)
    source = arch.get_source()
    if not quiet:
        print "\nReading data from %s on source %s..."%(filenm, source)
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
    #By-hand frequency calculation, equivalent to above from PSRCHIVE
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
    #The rest is now ignoring npol...
    arch.pscrunch()
    #Estimate noise -- FIX needs improvement
    noise_std = np.array([get_noise(subints[isub,0]) for isub in xrange(nsub)])
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
    #Number unzapped channels, subints
    nchanx = int(round(np.mean([subintsxs[isub].shape[1] for isub in
        xrange(nsub)])))
    nsubx = int(np.compress([subintsxs[isub].shape[1] for isub in
        xrange(nsub)], np.ones(nsub)).sum())
    if not quiet:
        P = arch.get_Integration(0).get_folding_period()*1000.0
        print "\tP [ms]             = %.1f\n\
        DM [cm**-3 pc]     = %.4f\n\
        center freq. [MHz] = %.4f\n\
        bandwidth [MHz]    = %.1f\n\
        # bins in prof     = %d\n\
        # channels         = %d\n\
        # chan (mean)      = %d\n\
        # subints          = %d\n\
        # unzapped subint  = %d\n\
        pol'n state        = %s\n"%(P, DM, nu0, bw, nbin, nchan, nchanx, nsub,
                nsubx, state)
    #Returns refreshed arch; could be changed...
    arch.refresh()
    if norm_weights:
        weights = weights_norm
    #Return getitem/attribute-accessible class!
    data = DataBunch(arch=arch, bw=bw, DM=DM, epochs=epochs,
            flux_prof=flux_prof, flux_profx=flux_profx, freqs=freqs,
            freqsxs=freqsxs, masks=masks, nbin=nbin, nchan=nchan,
            nchanx=nchanx, noise_std=noise_std, nsub=nsub, nsubx=nsubx,
            nu0=nu0, okichan=okichan, okisub=okisub, phases=phases, prof=prof,
            Ps=Ps, source=source, state=state, subints=subints,
            subintsxs=subintsxs, weights=weights)
    return data

def unpack_dict(data):
    """
    This does not work yet; just for reference...
    Dictionary has to be named 'data'...
    """
    for key in data.keys():
        exec(key + " = data['" + key + "']")

def write_model(filenm, name, nu_ref, model_params, fit_flags):
    """
    """
    outfile = open(filenm, "a")
    outfile.write("MODEL  %s\n"%name)
    outfile.write("FREQ   %.4f\n"%nu_ref)
    outfile.write("DC     %.8f  %d\n"%(model_params[0], fit_flags[0]))
    ngauss = (len(model_params) - 1) / 6
    for igauss in xrange(ngauss):
        comp = model_params[(1 + igauss*6):(7 + igauss*6)]
        fit_comp = fit_flags[(1 + igauss*6):(7 + igauss*6)]
        line = (igauss+1,) + tuple(np.array(zip(comp, fit_comp)).ravel())
        outfile.write("COMP%02d %1.8f  %d  % 10.8f  %d  % 10.8f  %d  % 10.8f  %d  % 12.8f  %d  % 12.8f  %d\n"%line)
    outfile.close()
    print "%s written."%filenm

def read_model(modelfile, phases=None, freqs=None, quiet=False):
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
            elif info[0][:4] == "COMP":
                comps.append(line)
                ngauss += 1
            else:
                pass
        except IndexError:
            pass
    #ngauss = len(modeldata) - 3
    params = np.zeros(ngauss*6 + 1)
    fit_flags = np.zeros(len(params))
    #name = modeldata.pop(0)[:-1]
    #nu_ref = float(modeldata.pop(0))
    #dc_line = modeldata.pop(0)
    #dc = float(dc_line.split()[0])
    #fit_dc = int(dc_line.split()[1])
    params[0] = dc
    fit_flags[0] = fit_dc
    for igauss in xrange(ngauss):
        #comp = map(float, modeldata[igauss].split()[::2])
        comp = map(float, comps[igauss].split()[1::2])
        #fit_comp = map(int, modeldata[igauss].split()[1::2])
        fit_comp = map(int, comps[igauss].split()[2::2])
        params[1 + igauss*6 : 7 + (igauss*6)] = comp
        fit_flags[1 + igauss*6 : 7 + (igauss*6)] = fit_comp
    if not read_only:
        nbin = len(phases)
        nchan = len(freqs)
        model = gen_gaussian_portrait(params, phases, freqs, nu_ref)
    if not quiet and not read_only:
        print "Model Name: %s"%name
        print "Made %d component model with %d profile bins,"%(ngauss, nbin)
        if len(freqs) != 1:
            bw = (freqs[-1] - freqs[0]) + ((freqs[-1] - freqs[-2]))
        else:
            bw = 0.0
        print "%d frequency channels, %.0f MHz bandwidth, centered near %.3f MHz,"%(nchan, bw, freqs.mean())
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
        dedispersed=True, state="Coherence", obs="GBT", quiet=False):
    """
    Mostly written by PBD.
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
            if param[0] == ("PSR" or "PSRJ"):
                PSR = param[1]
            elif param[0] == "RAJ":
                RAJ = param[1]
            elif param[0] == "DECJ":
                DECJ = param[1]
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
        scale=1.0, dedisperse=False, t_scat=None, bw_scint=None,
        state="Coherence", obs="GBT", quiet=False):
    """
    Mostly written by PBD.
    'phase' [rot] is an arbitrary rotation to all subints.
    'dDM' [cm**-3 pc] is an additional DM to what is given in ephemeris.
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
    name, ngauss, model = read_model(modelfile, phases, freqs, quiet=quiet)
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
    if dDM != 0.0:
        print obs + " %13s %8.3f %s %8.3f              %9.5f"%(name, nu_ref,
                TOA, TOA_err, dDM)
    else:
        print obs + " %13s %8.3f %s %8.3f"%(name, nu_ref, TOA, TOA_err)

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
    text =  "Residuals mean ~ %.3f\nResiduals std ~  %.3f\nData std ~       %.3f"%(residsx.mean(), residsx.std(), get_noise(portx))
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
