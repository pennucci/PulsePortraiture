#!/usr/bin/env python

#######
#pplib#
#######

#pplib contains all necessary functions and definitions for the fitting
#programs ppgauss and pptoas, as well as some additional functions used in our
#wideband timing analysis.

#Written by Timothy T. Pennucci (TTP; pennucci@virginia.edu).
#Contributions by Scott M. Ransom (SMR) and Paul B. Demorest (PBD) 

#########
#imports#
#########

import sys
import time
import numpy as np
import numpy.fft as fft
import scipy.optimize as opt
import scipy.signal as ss
import lmfit as lm
import psrchive as pr
nodes = False  #Used when needing parallelized operation
if nodes:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt

##########
#settings#
##########

#Exact dispersion constant (e**2/(2*pi*m_e*c)) (used by PRESTO).
Dconst_exact = 4.148808e3  #[MHz**2 cm**3 pc**-1 s]

#"Traditional" dispersion constant (used by PSRCHIVE).
Dconst_trad = 0.000241**-1 #[MHz**2 cm**3 pc**-1 s]

#Fitted DM values will depend on this choice.  Choose wisely.
Dconst = Dconst_trad

#Power-law index for scattering law
scattering_alpha = -4.0

#Default get_noise method (see functions get_noise_*).
#However, PSRCHIVE's baseline_stats is used in most cases (see load_data). 
#_To_be_improved_.
default_noise_method = 'PS'

#Ignore DC component in Fourier fit if DC_fact == 0, else set DC_fact == 1.
DC_fact = 0

#Upper limit on the width of a Gaussian component to "help" in fitting.
#Should be either None or > 0.0.
wid_max = 0.25

#If PL_model == True, the wid and loc of the gaussian parameters will be
#modeled with power-law functions instead of linear ones.
PL_model = True

#cfitsio defines a maximum number of files (NMAXFILES) that can be opened in
#the header file fitsio2.h.  Without calling unload() with PSRCHIVE, which
#touches the archive, I am not sure how to close the files.  So, to avoid the
#loop crashing, set a maximum number of archives for pptoas.  Modern machines
#should be able to handle almost 1000.
max_nfile = 999

#########
#display#
#########

#Set colormap preference
#Decent monocolor: pink, gist_heat, copper, hot, gray, Blues_r
#Decent multicolor: cubehelix, terrain, spectral, seismic
#see plt.cm for list of available colormaps.
default_colormap = 'gist_heat'
if hasattr(plt.cm, default_colormap):
    plt.rc('image', cmap=default_colormap)
else:
    plt.rc('image', cmap='pink')

#List of colors; can do this better...
cols = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'brown', 'purple', 'pink',
        'b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'brown', 'purple', 'pink',
        'b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'brown', 'purple', 'pink',
        'b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'brown', 'purple', 'pink']

######
#misc#
######

#Dictionary of observatory codes; not sure what "0" corresponds to.
#These are used for Princeton formatted TOAs, which may be incorrect...
obs_codes = {'bary':'@', '???':'0', 'gbt':'1', 'atca':'2', 'ao':'3',
             'arecibo':'3', 'nanshan':'5', 'tid43':'6', 'pks':'7', 'jb':'8',
             'vla':'c', 'ncy':'f', 'eff':'g', 'jbdfb':'q', 'wsrt':'i',
             'lofar':'t'}

#Dictionary of two-character observatory codes recognized by tempo/2.
#Taken and lowered from $TEMPO/obsys.dat
tempo_codes = {'arecibo':'ao', 'chime':'ch', 'effelsberg':'ef', 'gbt':'gb',
               'gmrt':'gm', 'jodrell':'jb', 'lofar':'lf', 'lwa':'lw',
               'nancay':'nc', 'parkes':'pk', 'shao':'sh', 'vla':'v2',
               'wsrt':'wb'}

#RCSTRINGS dictionary, for the return codes given by scipy.optimize.fmin_tnc.
#These are only needed for debugging.
RCSTRINGS = {'-1':'INFEASIBLE: Infeasible (low > up).',
             '0':'LOCALMINIMUM: Local minima reach (|pg| ~= 0).',
             '1':'FCONVERGED: Converged (|f_n-f_(n-1)| ~= 0.)',
             '2':'XCONVERGED: Converged (|x_n-x_(n-1)| ~= 0.)',
             '3':'MAXFUN: Max. number of function evaluations reach.',
             '4':'LSFAIL: Linear search failed.',
             '5':'CONSTANT: All lower bounds are equal to the upper bounds.',
             '6':'NOPROGRESS: Unable to progress.',
             '7':'USERABORT: User requested end of minimization.'}

#########
#classes#
#########

class DataBunch(dict):

    """
    Create a simple class instance of DataBunch.

    db = DataBunch(a=1, b=2,....) has attributes a and b, which are callable
    and update-able using either syntax db.a or db['a'].
    """

    def __init__(self, **kwds):
        dict.__init__(self, kwds)
        self.__dict__ = self

###########
#functions#
###########

def set_colormap(colormap):
    """
    Set the default colormap to colormap and apply to current image if any.

    See help(colormaps) for more information

    Stolen from matplotlib.pyplot: plt.pink().
    """
    plt.rc("image", cmap=colormap)
    im = plt.gci()

    if im is not None:
        exec("im.set_cmap(plt.cm.%s)"%colormap)
    plt.draw_if_interactive()

def gaussian_function(xs, loc, wid, norm=False):
    """
    Evaluates a gaussian function with parameters loc and wid at values xs.

    xs is the array of values that are evaluated in the function.
    loc is the pulse phase location (0-1) [rot].
    wid is the gaussian pulse's full width at half-max (FWHM) [rot].
    norm=True returns the profile such that the integrated density = 1.
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
    Return a gaussian pulse profile with nbin bins and peak amplitude of 1.

    nbin is the number of bins in the profile.
    loc is the pulse phase location (0-1) [rot].
    wid is the gaussian pulse's full width at half-max (FWHM) [rot].
    norm=True returns the profile such that the integrated density = 1.
    abs_wid=True, will use abs(wid).
    zeroout=True and wid <= 0, return a zero array.

    Note: The FWHM of a gaussian is approx 2.35482 "sigma", or exactly
          2*sqrt(2*ln(2)).

    Taken and tweaked from SMR's pygaussfit.py
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
    locval = np.arange(nbin, dtype='d') / np.float64(nbin)
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
    Return a model-profile with ngauss gaussian components.

    params is a sequence of 2 + (ngauss*3) values where the first value is
        the DC component, the second value is the scattering timescale [bin]
        and each remaining group of three represents the gaussians' loc (0-1),
        wid (i.e. FWHM) (0-1), and amplitude (>0.0).
    nbin is the number of bins in the model.

    Taken and tweaked from SMR's pygaussfit.py
    """
    ngauss = (len(params) - 2) / 3
    model = np.zeros(nbin, dtype='d') + params[0]
    for igauss in xrange(ngauss):
        loc, wid, amp = params[(2 + igauss*3):(5 + igauss*3)]
        model += amp * gaussian_profile(nbin, loc, wid)
    if params[1] != 0.0:
        bins = np.arange(nbin)
        sk = scattering_kernel(params[1], 1.0, np.array([1.0]), bins, P=1.0,
                alpha=scattering_alpha)[0]
        model = add_scattering(model, sk, repeat=3)
    return model

def gen_gaussian_portrait(params, phases, freqs, nu_ref, join_ichans=[],
        P=None, scattering_index=scattering_alpha):
    """
    Return a gaussian-component model portrait based on input parameters.

    join_ichans is used only in ppgauss, in which case the period P [sec] needs
        to be provided.

    params is an array of 2 + (ngauss*6) + 2*len(join_ichans) values.
        The first value is the DC component, and the second value is the
        scattering timescale [bin].  The next ngauss*6 values represent the
        gaussians' loc (0-1), evolution parameter in loc, wid (i.e. FWHM)
        (0-1), evolution parameter in wid, amplitude (>0,0), and spectral
        index alpha (no implicit negative).  The remaining 2*len(join_ichans)
        parameters are pairs of phase and DM.  The iith list of channels in
        join_ichans gets rotated in the generated model by the iith pair of
        phase and DM.
    phases is the array of phase values (will pass nbin to
        gen_gaussian_profile).
    freqs in the array of frequencies at which to calculate the model.
    nu_ref is the frequency to which the locs, wids, and amps reference.
    scattering_index is the power-law index of the scattering law; the default
        is set in the header lines of pplib.py.

    The evolution parameters will either be linear slopes, or a power-law
    indices, depending on the global settings in the header lines of pplib.py.

    The units of the evolution parameters and the frequencies need to match
    appropriately.
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
    if PL_model:
        gparams[:,2::3] = np.exp(np.outer(np.log(freqs) - np.log(nu_ref),
            locparams) + np.outer(np.ones(nchan), np.log(refparams[2::3])))
    else:
        gparams[:,2::3] = np.outer(freqs - nu_ref, locparams) + \
                np.outer(np.ones(nchan), refparams[2::3])
    #Wids
    if PL_model:
        gparams[:,3::3] = np.exp(np.outer(np.log(freqs) - np.log(nu_ref),
            widparams) + np.outer(np.ones(nchan), np.log(refparams[3::3])))
    else:
        gparams[:,3::3] = np.outer(freqs - nu_ref, widparams) + \
                np.outer(np.ones(nchan), refparams[3::3])
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
                alpha=scattering_index)
        gport = add_scattering(gport, sk, repeat=3)
    if njoin:
        for ij in xrange(njoin):
            join_ichan = join_ichans[ij]
            phi = join_params[0::2][ij]
            DM =  join_params[1::2][ij]
            gport[join_ichan] = rotate_data(gport[join_ichan], phi,
                    DM, P, freqs[join_ichan], nu_ref)
    return gport

def powlaw(nu, nu_ref, A, alpha):
    """
    Return a power-law 'spectrum' given by F(nu) = A*(nu/nu_ref)**alpha
    """
    return A * (nu/nu_ref)**alpha

def powlaw_integral(nu2, nu1, nu_ref, A, alpha):
    """
    Return the definite integral of a powerlaw from nu1 to nu2.

    The powerlaw is of the form A*(nu/nu_ref)**alpha.
    """
    alpha = np.float64(alpha)
    if alpha == -1.0:
        return A * nu_ref * np.log(nu2/nu1)
    else:
        C = A * (nu_ref**-alpha) / (1 + alpha)
        diff = ((nu2**(1+alpha)) - (nu1**(1+alpha)))
        return C * diff

def powlaw_freqs(lo, hi, N, alpha, mid=False):
    """
    Return frequencies spaced such that each channel has equal flux.

    Given a bandwidth from lo to hi frequencies, split into N channels, and a
    power-law index alpha, this function finds the frequencies such that each
    channel contains the same amount of flux.

    mid=True, returns N frequencies, corresponding to the center frequency in
        each channel. Default behavior returns N+1 frequencies (includes both
        lo and hi freqs).
    """
    alpha = np.float64(alpha)
    nus = np.zeros(N + 1)
    if alpha == -1.0:
        nus = np.exp(np.linspace(np.log(lo), np.log(hi), N+1))
    else:
        nus = np.power(np.linspace(lo**(1+alpha), hi**(1+alpha), N+1),
                (1+alpha)**-1)
        #Equivalently:
        #for ii in xrange(N+1):
        #    nus[ii] = ((ii / np.float64(N)) * (hi**(1+alpha)) + (1 - (ii /
        #        np.float64(N))) * (lo**(1+alpha)))**(1 / (1+alpha))
    if mid:
        midnus = np.zeros(N)
        for ii in xrange(N):
            midnus[ii] = 0.5 * (nus[ii] + nus[ii+1])
        nus = midnus
    return nus

def scattering_kernel(tau, nu_ref, freqs, phases, P, alpha=scattering_alpha):
    """
    Return a scattering kernel based on input parameters.

    tau is the scattering timescale in [sec] or [bin].
    nu_ref is the reference frequency for tau.
    freqs is the array of center frequencies in the nchan x nbin kernel
    phases gives the phase-bin centers of the nchan x nbin kernel
    P is the period [sec]; use P = 1.0 if tau is in units of [bin].
    alpha is the power-law index for the scattering evolution.
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

def add_scattering(port, kernel, repeat=3):
    """
    Add scattering into a portrait.

    port is the nchan x nbin portrait.
    kernel is the scattering kernel to be used in the convolution with port.
    repeat attempts to rid the convolution of edge effects by repeating the
        port and kernel repeat times before convolution; the center portion is
        returned.
    """
    mid = repeat/2
    d = np.array(list(port.transpose()) * repeat).transpose()
    k = np.array(list(kernel.transpose()) * repeat).transpose()
    if len(port.shape) == 1:
        nbin = port.shape[0]
        norm_kernel = kernel / kernel.sum()
        scattered_port = ss.convolve(norm_kernel, d)[mid * nbin : (mid+1) *
                nbin]
    else:
        nbin = port.shape[1]
        norm_kernel = np.transpose(np.transpose(k) * k.sum(axis=1)**-1)
        scattered_port = np.fft.irfft(np.fft.rfft(norm_kernel) *
                np.fft.rfft(d))[:, mid * nbin : (mid+1) * nbin]
    return scattered_port

def add_scintillation(port, params=None, random=True, nsin=2, amax=1.0,
        wmax=3.0):
    """
    Add totally fake scintillation to a portrait based on sinusoids.

    port is the nchan x nbin array of data values.
    params are triplets of sinusoidal parameters: "amps", "freqs" [cycles],
        and "phases" [cycles].
    if params is None and random is True, random params for nsin sinusoids are
        chosen from a uniform distribution on [0,amax], a chi-squared
        distribution with wmax dof, and a uniform distribution on [0,1].
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

def mean_C2N(nu, D, bw_scint):
    """
    Return mean_C2N [m**(-20/3)]

    For use with scattering measure.
    nu is the frequency [MHz]
    D is the distance [kpc]
    bw_scint is the scintillation bandwidth [MHz]

    Reference: Foster, Fairhead, and Backer (1991)
    """
    return 2e-14 * nu**(11/3.0) * D**(-11/6.0) * bw_scint**(-5/6.0)

def dDM(D, D_screen, nu, bw_scint):
    """
    Return the delta-DM [cm**-3 pc] predicted for a frequency dependent DM.

    D is the distance to the pulsar [kpc]
    D_screen is the distance from the Earth to the scattering screen [kpc]
    nu is the frequency [MHz]
    bw_scint is the scintillation bandwidth at nu [MHz]

    References: Cordes & Shannon (2010); Foster, Fairhead, and Backer (1991)
    """
    #SM is the scattering measure [m**(-20/3) kpc]
    SM = mean_C2N(nu, D, bw_scint) * D
    return 10**4.45 * SM * D_screen**(5/6.0) * nu**(-11/6.0)

def fit_powlaw_function(params, freqs, nu_ref, data=None, errs=None):
    """
    Return the weighted residuals from a power-law model and data.

    params is an array = [amplitude at reference frequency, spectral index].
    freqs is an nchan array of frequencies.
    nu_ref is the frequency at which the amplitude is referenced.
    data is the array of the data values.
    errs is the array of uncertainties on the data values.
    """
    prms = np.array([param.value for param in params.itervalues()])
    A = prms[0]
    alpha = prms[1]
    return (data - powlaw(freqs, nu_ref, A, alpha)) / errs

def fit_gaussian_profile_function(params, data=None, errs=None):
    """
    Return the weighted residuals from a gaussian profile model and data.

    See gen_gaussian_profile for form of input params.
    data is the array of data values.
    errs is the array of uncertainties on the data values.
    """
    prms = np.array([param.value for param in params.itervalues()])
    return (data - gen_gaussian_profile(prms, len(data))) / errs

def fit_gaussian_portrait_function(params, phases, freqs, nu_ref, data=None,
        errs=None, join_ichans=None, P=None):
    """
    Return the weighted residuals from a gaussian-component model and data.

    See gen_gaussian_portrait for form of input.
    data is the 2D array of data values.
    errs is the 2D array of the uncertainties on the data values.
    """
    prms = np.array([param.value for param in params.itervalues()])
    deviates = np.ravel((data - gen_gaussian_portrait(prms[:-1], phases, freqs,
        nu_ref, join_ichans, P, scattering_index=prms[-1])) / errs)
    return deviates

def fit_phase_shift_function(phase, model=None, data=None, err=None):
    """
    Return the negative, weighted inverse DFT at phase of model and data.

    phase is the input phase [rot]
    model is the array of model values (Fourier domain)
    data is the array of data values (Fourier domain)
    err is the noise level (Fourier domain)
    """
    harmind = np.arange(len(model))
    phasor = np.exp(harmind * 2.0j * np.pi * phase)
    C = -np.real((data * np.conj(model) * phasor).sum()) / err**2.0
    return C

def fit_phase_shift_function_deriv(phase, model=None, data=None, err=None):
    """
    Return the first derivative of fit_phase_shift_function at phase.

    See fit_phase_shift_function for form of input.
    """
    harmind = np.arange(len(model))
    phasor = np.exp(harmind * 2.0j * np.pi * phase)
    dC = -np.real((2.0j * np.pi * harmind * data * np.conj(model) *
        phasor).sum()) / err**2.0
    return dC

def fit_phase_shift_function_2deriv(phase, model=None, data=None, err=None):
    """
    Return the second derivative of fit_phase_shift_function at phase.

    See fit_phase_shift_function for form of input.
    """
    harmind = np.arange(len(model))
    phasor = np.exp(harmind * 2.0j * np.pi * phase)
    d2C = -np.real((-4.0 * (np.pi**2.0) * (harmind**2.0) * data *
        np.conj(model) * phasor).sum()) / err**2.0
    return d2C

def fit_portrait_function(params, model=None, p_n=None, data=None, errs=None,
        P=None, freqs=None, nu_ref=np.inf):
    """
    Return the function to be minimized by fit_portrait, evaluated at params.

    The returned value is equivalent to the chi-squared value of the model and
    data, given the input parameters, differing only by a constant depending
    on a weighted, quadratic sum of the data (see 'd' in fit_portrait).
    The quantity is related to the inverse DFT of model and data (i.e. the
    cross-correlation of the time-domain quantities).

    NB: both model and data must already be in the Fourier domain.

    params is an array = [phase, DM], with phase in [rot] and DM in
        [cm**-3 pc].
    model is the nchan x nbin phase-frequency model portrait that has been
        DFT'd along the phase axis.
    p_n is an nchan array containing a weighted, quadratic sum of the model
        (see 'p_n' in fit_portrait).
    data is the nchan x nbin phase-frequency data portrait that has been DFT'd
        along the phase axis.
    err is the nchan array of noise level estimates (in the Fourier domain).
    P is the period [s] of the pulsar at the data epoch.
    freqs is an nchan array of frequencies [MHz].
    nu_ref is the frequency [MHz] that is defined to have zero delay from a
        non-zero dispersion measure.
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
    Return the two first-derivatives of fit_portrait_function.

    See fit_portrait_function for form of input.
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
    Return the three second-derivatives of fit_portrait_function.

    The three unique values in the Hessian, which is a 2x2 symmetric matrix of
    the second-derivatives of fit_portrait_function, are returned, evaluated at
    params, as well as the estimate of the zero-covariance reference
    frequency.

    NB: The curvature matrix is one-half the second-derivative of the
    chi-squared function (this function).  The covariance matrix is the
    inverse of the curvature matrix.

    See fit_portrait_function for form of input.
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

def estimate_portrait(phase, DM, scales, data, errs, P, freqs, nu_ref=np.inf):
    #############
    #MOTH-BALLED#
    #############
    #here, all vars have additional epoch-index except nu_ref
    #i.e. all have to be arrays of at least len 1; errs are precision
    """
    Return an average over all data portraits.

    An early attempt to make a '2-D' version of PBD's autotoa.  That is, to
    iterate over all epochs of data portraits to build a non-gaussian model
    that can be smoothed.

    <UNDER CONSTRUCTION>

    References: PBD's autotoa, PhDT
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
    Return the 'optimal' Wiener filter given a noisy pulse profile.

    <UNDER CONSTRUCTION>

    prof is a noisy pulse profile.
    noise is standard error of the profile.

    Reference: PBD's PhDT
    """
    FFT = fft.rfft(prof)
    pows = np.real(FFT * np.conj(FFT)) / len(prof)
    return pows / (pows + (noise**2))
    #return (pows - (noise**2)) / pows

def brickwall_filter(N, kc):
    """
    Return a 'brickwall' filter with N points.

    The brickwall filter has the first kc as ones and the remainder as zeros.
    """
    fk = np.zeros(N)
    fk[:kc] = 1.0
    return fk

def fit_brickwall(prof, noise):
    #FIX this is obviously wrong
    """
    Return the index kc for the best-fit brickwall.

    See brickwall_filter and wiener_filter.

    <UNDER CONSTRUCTION>
    """
    wf = wiener_filter(prof, noise)
    N = len(wf)
    X2 = np.zeros(N)
    for ii in xrange(N):
        X2[ii] = np.sum((wf - brickwall_filter(N, ii))**2)
    return X2.argmin()

def half_triangle_function(a, b, dc, N):
    """
    Return a half-triangle function with base a and height b.

    dc is an overall baseline level.
    N is the length of the returned function.
    """
    fn = np.zeros(N) + dc
    a = int(np.floor(a))
    fn[:a] += -(np.float64(b)/a)*np.arange(a) + b
    return fn

def find_kc_function(params, data, errs=1.0):
    """
    Return the (weighted) chi-squared statistic for a half-triangle model.

    params are the input parameters for half_triangle_function: (a, b, dc).
    data is the array of data values.
    errs is the array of uncertainties on the data.
    """
    a, b, dc = params[0], params[1], params[2]
    return np.sum((data - half_triangle_function(a, b, dc, len(data)) /
        errs)**2.0)

def find_kc(pows):
    """
    Return the critical cutoff index kc based on a half-triangle function fit.

    The function attempts to find where the noise-floor in a power-spectrum
    begins.

    pows is the input power-spectrum values.
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
    Fit a power-law function to data.

    lmfit is used for the minimization.
    Returns an object containing the fitted parameter values, the parameter
    errors, an array of the residuals, the chi-squared value, and the number of
    degrees of freedom.

    data is the input array of data values used in the fit.
    init_params is a list of initial parameter guesses = [amplitude at nu_ref,
        spectral index].
    errs is the array of uncertainties on the data values.
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
    #fitted_params = np.array([param.value for param in
    #    results.params.itervalues()])
    dof = results.nfree
    chi2 = results.chisqr
    red_chi2 = results.redchi
    residuals = results.residual
    #fit_errs = np.array([param.stderr for param in
    #    results.params.itervalues()])
    results = DataBunch(alpha=results.params['alpha'].value,
            alpha_err=results.params['alpha'].stderr,
            amp=results.params['amp'].value,
            amp_err=results.params['amp'].stderr, residuals=residuals,
            chi2=chi2, dof=dof)
    return results

def fit_gaussian_profile(data, init_params, errs, fit_flags=None,
        fit_scattering=False, quiet=True):
    """
    Fit gaussian functions to a profile.

    lmfit is used for the minimization.
    Returns an object containing an array of fitted parameter values, an array
    of parameter errors, an array of the residuals, the chi-squared value, and
    the number of degrees of freedom.

    data is the pulse profile array of length nbin used in the fit.
    init_params is a list of initial guesses for the 2 + (ngauss*3) values;
        the first value is the DC component, the second value is the
        scattering timescale [bin] and each remaining group of three represents
        the gaussians' loc (0-1), wid (i.e. FWHM) (0-1), and amplitude (>0.0).
    errs is the array of uncertainties on the data values.
    fit_flags is an array specifying which of the non-scattering parameters to
        fit; defaults to fitting all.
    fit_scattering=True fits a scattering timescale parameter via convolution
        with a one-sided exponential function.
    quiet=True suppresses output [default].
    """
    nparam = len(init_params)
    ngauss = (len(init_params) - 2) / 3
    if fit_flags is None:
        fit_flags = [True for t in xrange(nparam)]
        fit_flags[1] = fit_scattering
    else:
        fit_flags = [np.bool(fit_flags[0]), fit_scattering] + \
                [np.bool(fit_flags[xx]) for xx in xrange(1, nparam-1)]
    #Generate the parameter structure
    params = lm.Parameters()
    for ii in xrange(nparam):
        if ii == 0:
            params.add('dc', init_params[ii], vary=fit_flags[ii], min=None,
                    max=None, expr=None)
        elif ii ==1:
            params.add('tau', init_params[ii], vary=fit_flags[ii], min=0.0,
                    max=None, expr=None)
        elif ii in range(nparam)[2::3]:
            params.add('loc%s'%str((ii-2)/3 + 1), init_params[ii],
                    vary=fit_flags[ii], min=None, max=None, expr=None)
        elif ii in range(nparam)[3::3]:
            params.add('wid%s'%str((ii-3)/3 + 1), init_params[ii],
                    vary=fit_flags[ii], min=0.0, max=wid_max, expr=None)
        elif ii in range(nparam)[4::3]:
            params.add('amp%s'%str((ii-4)/3 + 1), init_params[ii],
                    vary=fit_flags[ii], min=0.0, max=None, expr=None)
        else:
            print "Undefined index %d."%ii
            sys.exit()
    other_args = {'data':data, 'errs':errs}
    #Now fit it
    results = lm.minimize(fit_gaussian_profile_function, params,
            kws=other_args)
    fitted_params = np.array([param.value for param in
        results.params.itervalues()])
    fit_errs = np.array([param.stderr for param in
        results.params.itervalues()])
    dof = results.nfree
    chi2 = results.chisqr
    red_chi2 = results.redchi
    residuals = results.residual
    if not quiet:
        print "---------------------------------------------------------------"
        print "Multi-Gaussian Profile Fit Results"
        print "---------------------------------------------------------------"
        print "lmfit status:", results.message
        print "gaussians:", ngauss
        print "DOF:", dof
        print "reduced chi-sq: %.2f" % red_chi2
        print "residuals mean: %.3g" % np.mean(residuals)
        print "residuals std.: %.3g" % np.std(residuals)
        print "---------------------------------------------------------------"
    results = DataBunch(fitted_params=fitted_params, fit_errs=fit_errs,
            residuals=residuals, chi2=chi2, dof=dof)
    return results

def fit_gaussian_portrait(data, init_params, errs, fit_flags, phases, freqs,
        nu_ref, join_params=[], P=None, fit_scattering_index=False,
        quiet=True):
    """
    Fit evolving gaussian components to a portrait.

    lmfit is used for the minimization.
    Returns an object containing an array of fitted parameter values, an array
    of parameter errors, the chi-squared value, and the number of degrees of
    freedom.

    data is the nchan x nbin phase-frequency data portrait used in the fit.
    init_params is a list of initial guesses for the 1 + (ngauss*6)
        parameters in the model; the first value is the DC component.  Each
        remaining group of six represent the gaussians loc (0-1), linear slope
        in loc, wid (i.e. FWHM) (0-1), linear slope in wid, amplitude (>0,0),
        and spectral index alpha (no implicit negative).
    errs is the array of uncertainties on the data values.
    fit_flags is an array of 1 + (ngauss*6) values, where non-zero entries
        signify that the parameter should be fit.
    phases is the array of phase values.
    freqs in the array of frequencies at which to calculate the model.
    nu_ref [MHz] is the frequency to which the locs, wids, and amps reference.
    join_params specifies how to simultaneously fit several portraits; see
        ppgauss.
    P is the pulse period [sec].
    fit_scattering_index will also fit for the power-law index of the
        scattering law, with the initial guess set as the default in the header
        lines of pplib.py.
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
                    vary=bool(fit_flags[ii]), min=0.0, max=wid_max, expr=None)
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
    #Fit scattering index?  Not recommended.
    params.add('scattering_index', scattering_alpha, vary=fit_scattering_index,
            min=None, max=None, expr=None)
    #Now fit it
    results = lm.minimize(fit_gaussian_portrait_function, params,
            kws=other_args)
    fitted_params = np.array([param.value for param in
        results.params.itervalues()])
    scattering_index = fitted_params[-1]
    fitted_params = fitted_params[:-1]
    fit_errs = np.array([param.stderr for param in
        results.params.itervalues()])
    scattering_index_err = fit_errs[-1]
    fit_errs = fit_errs[:-1]
    dof = results.nfree
    chi2 = results.chisqr
    red_chi2 = results.redchi
    residuals = results.residual
    if not quiet:
        print "---------------------------------------------------------------"
        print "Gaussian Portrait Fit"
        print "---------------------------------------------------------------"
        print "lmfit status:", results.message
        print "gaussians:", ngauss
        print "DOF:", dof
        print "reduced chi-sq: %.2g" %red_chi2
        print "residuals mean: %.3g" %np.mean(residuals)
        print "residuals std.: %.3g" %np.std(residuals)
        print "data std.: %.3g" %get_noise(data)
        print "---------------------------------------------------------------"
    results = DataBunch(fitted_params=fitted_params, fit_errs=fit_errs,
            scattering_index=scattering_index,
            scattering_index_err=scattering_index_err, chi2=chi2, dof=dof)
    return results

def fit_phase_shift(data, model, noise=None, bounds=[-0.5, 0.5]):
    """
    Fit a phase shift between data and model.

    This is a simple implementation of FFTFIT using a brute-force algorithm
    Returns an object containing the fitted parameter values (phase and scale),
    the parameter errors, and the reduced chi-squared value.

    The returned phase is the phase of the data with respect to the model.
    NB: the provided rotation functions rotate to earlier phases, given a
    positive phase.

    data is the array of data profile values.
    model is the array of model profile values.
    noise is time-domain noise-level; it is measured if None.
    bounds is the list containing the bounds on the phase.
    """
    dFFT = fft.rfft(data)
    dFFT[0] *= DC_fact
    mFFT = fft.rfft(model)
    mFFT[0] *= DC_fact
    if noise is None:
        #err = np.real(dFFT[-len(dFFT)/4:]).std()
        err = get_noise(data) * np.sqrt(len(data)/2.0)
    else:
        err = noise * np.sqrt(len(data)/2.0)
    d = np.real(np.sum(dFFT * np.conj(dFFT))) / err**2.0
    p = np.real(np.sum(mFFT * np.conj(mFFT))) / err**2.0
    other_args = (mFFT, dFFT, err)
    start = time.time()
    results = opt.brute(fit_phase_shift_function, [tuple(bounds)],
            args=other_args, Ns=100, full_output=True)
    duration = time.time() - start
    phase = results[0][0]
    fmin = results[1]
    scale = -fmin / p
    #In the next two error equations, consult fit_portrait for factors of 2
    phase_error = (scale * fit_phase_shift_function_2deriv(phase, mFFT, dFFT,
        err))**-0.5
    scale_error = p**-0.5
    red_chi2 = (d - ((fmin**2) / p)) / (len(data) - 2)
    #SNR of the fit, based on PDB's notes
    snr = pow(scale**2 * p / err**2, 0.5)
    return DataBunch(phase=phase, phase_err=phase_error, scale=scale,
            scale_error=scale_error, snr=snr, red_chi2=red_chi2,
            duration=duration)

def fit_portrait(data, model, init_params, P, freqs, nu_fit=None, nu_out=None,
        errs=None, bounds=[(None, None), (None, None)], id=None, quiet=True):
    """
    Fit a phase offset and DM between a data portrait and model portrait.

    A truncated Newtonian algorithm is used.
    Returns an object containing the fitted parameter values, the parameter
    errors, and other attributes.

    data is the nchan x nbin phase-frequency data portrait.
    model is the nchan x nbin phase-frequency model portrait.
    init_params is a list of initial parameter guesses = [phase, DM], with
        phase in [rot] and DM in [cm**-3 pc].
    P is the period [s] of the pulsar at the data epoch.
    freqs is an nchan array of frequencies [MHz].
    nu_fit is the frequency [MHz] used as nu_ref in the fit.  Defaults to the
        mean value of freqs.
    nu_out is the desired output reference frequency [MHz].  Defaults to the
        zero-covariance frequency calculated in the fit.
    errs is the array of uncertainties on the data values (time-domain); they
        are measured if None.
    bounds is the list of 2 tuples containing the bounds on the phase and DM.
    id provides a label for the TOA.
    quiet = False produces more diagnostic output.
    """
    dFFT = fft.rfft(data, axis=1)
    dFFT[:, 0] *= DC_fact
    mFFT = fft.rfft(model, axis=1)
    mFFT[:, 0] *= DC_fact
    if errs is None:
        #errs = np.real(dFFT[:, -len(dFFT[0])/4:]).std(axis=1)
        errs = get_noise(data, chans=True) * np.sqrt(len(data[0])/2.0)
    else:
        errs = np.copy(errs) * np.sqrt(len(data[0])/2.0)
    d = np.real(np.sum(np.transpose(errs**-2.0 * np.transpose(dFFT *
        np.conj(dFFT)))))
    p_n = np.real(np.sum(mFFT * np.conj(mFFT), axis=1))
    if nu_fit is None: nu_fit = freqs.mean()
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
            isub = id[-ii:]
            filename = id[:-ii-1]
            sys.stderr.write(
                    "Fit failed with return code %d: %s -- %s subint %s\n"%(
                        results.status, rcstring, filename, isub))
        else:
            sys.stderr.write(
                    "Fit failed with return code %d -- %s"%(results.status,
                        rcstring))
    if not quiet and results.success is True and 0:
        sys.stderr.write("Fit succeeded with return code %d -- %s\n"
                %(results.status, rcstring))
    #Curvature matrix = 1/2 2deriv of chi2 (cf. Gregory sect 11.5)
    #Parameter errors are related to curvature matrix by **-0.5 
    #Calculate nu_zero
    nu_zero = fit_portrait_function_2deriv(np.array([phi, DM]), mFFT,
            p_n, dFFT, errs, P, freqs, nu_fit)[1]
    if nu_out is None:
        nu_out = nu_zero
    phi_out = phase_transform(phi, DM, nu_fit, nu_out, P, mod=False)
    #Calculate Hessian
    hessian = fit_portrait_function_2deriv(np.array([phi_out, DM]),
            mFFT, p_n, dFFT, errs, P, freqs, nu_out)[0]
    hessian = np.array([[hessian[0], hessian[2]], [hessian[2], hessian[1]]])
    covariance_matrix = np.linalg.inv(0.5*hessian)
    covariance = covariance_matrix[0,1]
    #These are true 1-sigma errors iff covariance == 0
    param_errs = list(covariance_matrix.diagonal()**0.5)
    DoF = len(data.ravel()) - (len(freqs) + 2)
    red_chi2 = (d + results.fun) / DoF
    #Calculate scales
    scales = get_scales(data, model, phi, DM, P, freqs, nu_fit)
    #Errors on scales, if ever needed (these may be wrong b/c of covariances)
    scale_errs = pow(p_n / errs**2.0, -0.5)
    #SNR of the fit, based on PDB's notes
    snr = pow(np.sum(scales**2 * p_n / errs**2), 0.5)
    results = DataBunch(phase=phi_out, phase_err=param_errs[0], DM=DM,
            DM_err=param_errs[1], scales=scales, scale_errs=scale_errs,
            nu_ref=nu_out, covariance=covariance, red_chi2=red_chi2, snr=snr,
            duration=duration, nfeval=nfeval, return_code=return_code)
    return results

def get_noise(data, method=default_noise_method, **kwargs):
    """
    Estimate the off-pulse noise.

    data is a 1- or 2-D array of input values.
    method is either "PS" or "fit" where:
        "PS" uses the mean of the last quarter if the power spectrum, and
        "fit" attempts to find the noise floor by fitting a half-triangle
        function to the power spectrum.
    **kwargs are passed to the selected noise-measuring function.

    Other noise-measuring methods to come.
    """
    if method == "PS":
        return get_noise_PS(data, **kwargs)
    elif method == "fit":
        return get_noise_fit(data, **kwargs)
    else:
        print "Unknown get_noise method."
        return 0

def get_noise_PS(data, frac=4, chans=False):
    """
    Estimate the off-pulse noise.

    This function will estimate the noise based on the mean of the highest
    1/frac harmonics in the power spectrum of data.

    data is a 1- or 2-D array of values.
    frac is a value specifying the inverse of the fraction of the
        power-spectrum to be examined.
    chans=True will return an estimate of the noise in each input channel.
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
    Estimate the off-pulse noise.

    This function will estimate the noise based on the mean of the highest
    harmonics, where critical cutoff harmonic is found by a fit of a
    half-triangle function to the power-spectrum of the data.

    data is a 1- or 2-D array of values.
    fact is a value to scale the fitted cutoff harmonic.
    chans=True will return an estimate of the noise in each input channel.
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
    Return an estimate of the signal-to-noise ratio (SNR) of the data.

    Assumes that the baseline is removed!

    prof is an input array of data.
    fudge is a scale factor that attempts to match (poorly) PSRCHIVE's SNRs.

    Reference: Lorimer & Kramer (2005).
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
    Return the best-fit, per-channel scaling amplitudes.

    data is the nchan x nbin phase-frequency data portrait.
    model is the nchan x nbin phase-frequency model portrait.
    phase is the best-fit phase [rot]
    DM is the best-fit dispersion measure in [cm**-3 pc].
    P is the period [s] of the pulsar at the data epoch.
    freqs is an nchan array of frequencies [MHz].
    nu_ref is the reference frequency of the input phase [MHz].

    Reference: Equation 11 of Pennucci, Demorest, & Ransom (2014).
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

def rotate_data(data, phase=0.0, DM=0.0, Ps=None, freqs=None, nu_ref=np.inf):
    """
    Rotate and/or dedisperse data.

    Positive values of phase and DM rotate the data to earlier phases
    (i.e. it "dedisperses").

    Simpler functions are rotate_portrait or rotate_profile.

    data is a 1-, 2-, or 4-D array of data -- i.e. either an array of nbin (a
        profile), nchan x nbin (a portrait), or nsub x npol x nchan x nbin
        (a subint) values.
    phase is a value specifying the amount of achromatic rotation [rot].
    DM is a value specifying the amount of rotation based on the cold-plasma
        dispersion law [cm**-3 pc].
    Ps is a single float or an array of nsub periods [sec] needed if DM != 0.0.
    freqs is a single float or an array of either nchan or nsub x nchan
        frequencies [MHz].
    nu_ref is the reference frequency [MHz] that has zero dispersive delay.
    """
    shape = data.shape
    ndim = data.ndim
    if DM == 0.0:
        idim = 'ijkl'
        idim = idim[:ndim]
        iaxis = range(ndim)
        baxis = iaxis[-1]
        bdim = list(idim)[baxis]
        dFFT = fft.rfft(data, axis=baxis)
        nharm = dFFT.shape[baxis]
        harmind = np.arange(nharm)
        baxis = iaxis.pop(baxis)
        othershape = np.take(shape, iaxis)
        ones = np.ones(othershape)
        order = np.take(list(idim), iaxis)
        order = ''.join([order[xx] for xx in xrange(len(order))])
        phasor = np.exp(harmind * 2.0j * np.pi * phase)
        phasor = np.einsum(order + ',' + bdim, ones, phasor)
        dFFT *= phasor
        return fft.irfft(dFFT, axis=baxis)
    else:
        datacopy = np.copy(data)
        while(datacopy.ndim != 4):
            datacopy = np.array([datacopy])
        baxis = 3
        nsub = datacopy.shape[0]
        npol = datacopy.shape[1]
        nchan = datacopy.shape[2]
        dFFT = fft.rfft(datacopy, axis=baxis)
        nharm = dFFT.shape[baxis]
        harmind = np.arange(nharm)
        D = Dconst * DM / (np.ones(nsub)*Ps)
        if len(D) != nsub:
            print "Wrong shape for array of periods."
            sys.exit()
        try:
            test = float(nu_ref)
        except TypeError:
            print "Only one nu_ref permitted."
            sys.exit()
        if not hasattr(freqs, 'ndim'):
            freqs = np.ones(nchan)*freqs
        if freqs.ndim == 0:
            freqs = np.ones(nchan)*float(freqs)
        if freqs.ndim == 1:
            if nchan != len(freqs):
                print "Wrong number of frequencies."
                sys.exit()
            fterm = np.tile(freqs, nsub).reshape(nsub, nchan)**-2.0 - \
                    nu_ref**-2.0
        else:
            fterm = freqs**-2.0 - nu_ref**-2.0
        if fterm.shape[1] != nchan or fterm.shape[0] != nsub:
            print "Wrong shape for frequency array."
            sys.exit()
        phase += np.array([D[isub]*fterm[isub] for isub in range(nsub)])
        phase = np.einsum('ij,k', phase, harmind)
        phasor = np.exp(2.0j * np.pi * phase)
        dFFT = np.array([dFFT[:,ipol,:,:]*phasor for ipol in xrange(npol)])
        dFFT = np.einsum('jikl', dFFT)
        if ndim == 1:
            return fft.irfft(dFFT, axis=baxis)[0,0,0]
        elif ndim == 2:
            return fft.irfft(dFFT, axis=baxis)[0,0]
        elif ndim == 4:
            return fft.irfft(dFFT, axis=baxis)
        else:
            print "Wrong number of dimensions."
            sys.exit()

def rotate_portrait(port, phase=0.0, DM=None, P=None, freqs=None,
        nu_ref=np.inf):
    """
    Rotate and/or dedisperse a portrait.

    Positive values of phase and DM rotate the data to earlier phases
    (i.e. it "dedisperses").

    When used to dediserpse, rotate_portrait is virtually identical to
    arch.dedisperse() in PSRCHIVE.

    port is a nchan x nbin array of data values.
    phase is a value specifying the amount of achromatic rotation [rot].
    DM is a value specifying the amount of rotation based on the cold-plasma
        dispersion law [cm**-3 pc].
    P is the pulsar period [sec] at the epoch of the data, needed if a DM is
        provided.
    freqs is an array of frequencies [MHz], needed if a DM is provided.
    nu_ref is the reference frequency [MHz] that has zero delay for any value
        of DM.
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
    Rotate a profile by phase.

    Positive values of phase rotate to earlier phase.

    profile is an input array of data.
    phase is a value specifying the amount of rotation [rot].
    """
    pFFT = fft.rfft(profile)
    pFFT *= np.exp(np.arange(len(pFFT)) * 2.0j * np.pi * phase)
    return fft.irfft(pFFT)

def fft_rotate(arr, bins):
    """
    Return array 'arr' rotated by 'bins' places to the left.

    The rotation is done in the Fourier domain using the Shift Theorem.
    'bins' can be fractional.
    The resulting vector will have the same length as the original.

    Taken and tweaked from SMR's PRESTO.  Used for testing.
    """
    arr = np.asarray(arr)
    freqs = np.arange(arr.size/2 + 1, dtype=np.float64)
    phasor = np.exp(complex(0.0, 2*np.pi) * freqs * bins /
            np.float64(arr.size))
    return np.fft.irfft(phasor * np.fft.rfft(arr), arr.size)

def DM_delay(DM, freq, freq_ref=np.inf, P=None):
    """
    Return the amount of dispersive delay [sec] between two frequencies.

    DM is the dispersion measure [cm**-3 pc].
    freq is the delayed frequency [MHz].
    freq_ref is the frequency [MHz] against which the delay is measured.
    P is a period [sec]; if provided, the return is in [rot].
    """
    delay = Dconst * DM * ((freq**-2.0) - (freq_ref**-2.0))
    if P:
        return delay / P
    else:
        return delay

def phase_transform(phi, DM, nu_ref1=np.inf, nu_ref2=np.inf, P=None,
        mod=False):
    """
    Transform an input delay at nu_ref1 to a delay at nu_ref2.

    phi is an input delay [rot] or [sec].
    DM is the dispersion measure [cm**-3 pc].
    nu_ref1 is the reference frequency [MHz] for phi.
    nu_ref2 is the reference frequency [MHz] of the ouput delay.
    P is the pulsar period; if not provided, assumes phi is in [sec].
    mod=True ensures the output delay in [rot] is on the interval [-0.5, 0.5).

    Default behavior is for P=1.0 [sec], i.e. transform delays [sec]
    """
    if P is None:
        P = 1.0
        mod = False
    phi_prime =  phi + (Dconst * DM * P**-1 * (nu_ref2**-2.0 - nu_ref1**-2.0))
    if mod:
        #phi_prime %= 1
        phi_prime = np.where(abs(phi_prime) >= 0.5, phi_prime % 1, phi_prime)
        phi_prime = np.where(phi_prime >= 0.5, phi_prime - 1.0, phi_prime)
        if not phi_prime.shape:
            phi_prime = np.float64(phi_prime)
    return phi_prime

def guess_fit_freq(freqs, SNRs=None):
    """
    Estimate a zero-covariance frequency.

    Returns a "center of mass" frequency, where the weights are given by
    SNR*(freq**-2).

    freqs is an array of frequencies.
    SNRs is an array of signal-to-noise ratios (defaults to 1s).
    """
    nu0 = (freqs.min() + freqs.max()) * 0.5
    if SNRs is None:
        SNRs = np.ones(len(freqs))
    diff = np.sum((freqs - nu0) * SNRs * freqs**-2) / np.sum(SNRs * freqs**-2)
    return nu0 + diff

def doppler_correct_freqs(freqs, doppler_factor):
    """
    Correct frequencies for doppler motion.

    Returns barycentric frequencies.  Untested.

    freqs is an array of topocentric frequencies.
    doppler_factor is the Doppler factor:
        doppler_factor = nu_source / nu_observed = sqrt( (1+beta) / (1-beta)),
        for beta = v/c, and v is positive for increasing source distance.
        NB: PSRCHIVE defines doppler_factor as the inverse of the above.

    Reference: Equations 13, 14, & 15 of Pennucci, Demorest, & Ransom (2014).
    """
    return doppler_factor * freqs

def calculate_TOA(epoch, P, phi, DM=0.0, nu_ref1=np.inf, nu_ref2=np.inf):
    """
    Calculate a TOA [PSRCHIVE MJD] for given input.

    epoch is a PSRCHIVE MJD.
    P is the pulsar period [sec].
    phi is the phase offset [rot].
    DM is the dispersion measure [cm**-3 pc], for transforming to nu_ref2.
    nu_ref1 is the reference frequency [MHz] of phi.
    nu_ref2 is the reference frequency [MHz] of the output TOA.
    """
    #The pre-Doppler corrected DM must be used
    phi_prime = phase_transform(phi, DM, nu_ref1, nu_ref2, P, mod=False)
    TOA = epoch + pr.MJD((phi_prime * P) / (3600 * 24.))
    return TOA

def load_data(filename, dedisperse=False, dededisperse=False, tscrunch=False,
        pscrunch=False, fscrunch=False, rm_baseline=True, flux_prof=False,
        refresh_arch=True, return_arch=True, quiet=False):
    """
    Load data from a PSRCHIVE archive.

    Returns an object containing a large number of useful archive attributes.

    filename is the input PSRCHIVE archive.
    Most of the options should be self-evident; archives are manipulated by
        PSRCHIVE only.
    flux_prof=True will include an array with the phase-averaged flux profile.
    refresh_arch=True refreshes the returned archive to its original state.
    return_arch=False will not return the archive, which may be smart at times.
    quiet=True suppresses output.
    """
    #Load archive
    arch = pr.Archive_load(filename)
    source = arch.get_source()
    if not quiet:
        print "\nReading data from %s on source %s..."%(filename, source)
    #Basic info used in TOA output
    telescope = arch.get_telescope()
    try:
        tempo_code = tempo_codes[telescope.lower()]
    except KeyError:
        tempo_code = '??'
    frontend = arch.get_receiver_name()
    backend = arch.get_backend_name()
    backend_delay = arch.get_backend_delay()
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
    freqs = np.array([[sub.get_centre_frequency(ichan) for ichan in \
            xrange(nchan)] for sub in arch])
    nbin = arch.get_nbin()
    #Centers of phase bins
    phases = np.linspace(0.0 + (nbin*2)**-1, 1.0 - (nbin*2)**-1, nbin)
    #These are NOT the bin centers...
    #phases = np.arange(nbin, dtype='d') / nbin
    #Get data
    #PSRCHIVE indices are [isub:ipol:ichan:ibin]
    subints = arch.get_data()
    Ps = np.array([sub.get_folding_period() for sub in arch], dtype=np.double)
    epochs = [sub.get_epoch() for sub in arch]
    subtimes = [sub.get_duration() for sub in arch]
    #Get weights
    weights = arch.get_weights()
    weights_norm = np.where(weights == 0.0, np.zeros(weights.shape),
            np.ones(weights.shape))
    #Get off-pulse noise
    noise_stds = np.array([sub.baseline_stats()[1]**0.5 for sub in arch])
    ok_isubs = np.compress(weights_norm.mean(axis=1), xrange(nsub))
    ok_ichans = [np.compress(weights_norm[isub], xrange(nchan)) \
            for isub in xrange(nsub)]
    #np.einsum is AWESOME
    masks = np.einsum('ij,k', weights_norm, np.ones(nbin))
    masks = np.einsum('j,ikl', np.ones(npol), masks)
    SNRs = np.zeros([nsub, npol, nchan])
    for isub in xrange(nsub):
        for ipol in xrange(npol):
            for ichan in xrange(nchan):
                SNRs[isub, ipol, ichan] = \
                        arch.get_Integration(
                                isub).get_Profile(ipol, ichan).snr()
    #The rest is now ignoring npol...
    arch.pscrunch()
    if flux_prof:
        #Flux profile
        #The below is about equal to bscrunch to ~6 places
        arch.dedisperse()
        arch.tscrunch()
        flux_prof = arch.get_data().mean(axis=3)[0][0]
    else:
        flux_prof = np.array([])
    #Get pulse profile
    arch.tscrunch()
    arch.fscrunch()
    prof = arch.get_data()[0,0,0]
    prof_noise = arch.get_Integration(0).baseline_stats()[1][0,0]**0.5
    prof_SNR = arch.get_Integration(0).get_Profile(0,0).snr()
    #Number unzapped channels (mean), subints
    nchanx = np.array(map(len, ok_ichans)).mean()
    nsubx = len(ok_isubs)
    if not quiet:
        P = arch.get_Integration(0).get_folding_period()*1000.0
        print "\tP [ms]             = %.3f\n\
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
    if refresh_arch: arch.refresh()
    if not return_arch: arch = None
    #Return getitem/attribute-accessible class!
    data = DataBunch(arch=arch, backend=backend, backend_delay=backend_delay,
            bw=bw, DM=DM, epochs=epochs, filename=filename,
            flux_prof=flux_prof, freqs=freqs, frontend=frontend, masks=masks,
            nbin=nbin, nchan=nchan, noise_stds=noise_stds, nsub=nsub, nu0=nu0,
            ok_ichans=ok_ichans, ok_isubs=ok_isubs, phases=phases, prof=prof,
            prof_noise=prof_noise, prof_SNR=prof_SNR, Ps=Ps, SNRs=SNRs,
            source=source, state=state, subints=subints, subtimes=subtimes,
            telescope=telescope, tempo_code=tempo_code, weights=weights)
    return data

def unpack_dict(data):
    """
    unpack a DataBunch/dictionary

    <UNDER CONSTRUCTION>

    This does not work yet; just for reference...
    Dictionary has to be named 'data'...
    """
    for key in data.keys():
        exec(key + " = data['" + key + "']")

def write_model(filename, name, nu_ref, model_params, fit_flags, append=False,
        quiet=False):
    """
    Write a gaussian-component model.

    filename is the output file name.
    name is the name of the model.
    nu_ref is the reference frequency [MHz] of the model.
    model_params is the list of 2 + 6*ngauss model parameters, where index 1 is
        the scattering timescale [sec].
    fit_flags is the list of 2 + 6*ngauss flags (1 or 0) designating a fit.
    append=True will append to a file named filename.
    quiet=True suppresses output.
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
        outfile.write("COMP%02d % .8f %d  % .8f %d  % .8f %d  % .8f %d  % .8f %d  % .8f %d\n"%line)
    outfile.close()
    if not quiet: print "%s written."%filename

def read_model(modelfile, phases=None, freqs=None, P=None, quiet=False):
    """
    Read-in a gaussian-component model.

    If only modelfile is specified, returns the contents of the modelfile:
        (model name, model reference frequency, number of gaussian components,
        list of parameters, list of fit flags).
    Otherwise, builds a model based on the input phases, frequencies, and
    period (if scattering timescale != 0.0).

    modelfile is the name of the write_model(...)-type of model file.
    phases is an array of phase-bin centers [rot].
    freqs in an array of center-frequencies [MHz].
    P is the pulsar period [sec]; needed if the scattering timescale != 0.0.
    quiet=True suppresses output.
    """
    if phases is None and freqs is None:
        read_only = True
    else:
        read_only = False
    ngauss = 0
    comps = []
    if not quiet:
        print "Reading model from %s..."%modelfile
    modeldata = open(modelfile, "r").readlines()
    for line in modeldata:
        info = line.split()
        try:
            if info[0] == "MODEL":
                name = info[1]
            elif info[0] == "FREQ":
                nu_ref = np.float64(info[1])
            elif info[0] == "DC":
                dc = np.float64(info[1])
                fit_dc = int(info[2])
            elif info[0] == "TAU":
                tau = np.float64(info[1])
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
        comp = map(np.float64, comps[igauss].split()[1::2])
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
        model = gen_gaussian_portrait(params, phases, freqs, nu_ref,
                scattering_index=scattering_alpha)
    if not quiet and not read_only:
        print "Model Name: %s"%name
        print "Made %d component model with %d profile bins,"%(ngauss, nbin)
        if len(freqs) != 1:
            bw = (freqs[-1] - freqs[0]) + ((freqs[-1] - freqs[-2]))
        else:
            bw = 0.0
        print "%d frequency channels, ~%.0f MHz bandwidth, centered near ~%.0f MHz,"%(nchan, abs(bw), freqs.mean())
        print "with model parameters referenced at %.3f MHz."%nu_ref
    #This could be changed to a DataBunch
    if read_only:
        return name, nu_ref, ngauss, params, fit_flags
    else:
        return name, ngauss, model

def file_is_ASCII(filename):
    """
    Checks if a file is ASCII.
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
        dedispersed=False, state="Stokes", obs="GBT", quiet=False):
    """
    Write data to a PSRCHIVE psrfits file (using a hack).

    Not guaranteed to work perfectly.

    Takes dedispersed data, please.

    data is a nsub x npol x nchan x nbin array of values.
    ephemeris is the timing ephemeris to be installed.
    freqs is an array of the channel center-frequencies [MHz].
    nu0 is the center frequency [MHz]; if None, defaults to the mean of freqs.
    bw is the bandwidth; if None, is calculated from freqs.
    outfile is the desired output file name.
    tsub is the duration of each subintegration [sec].
    start_MJD is the starting epoch of the data in PSRCHIVE MJD format.
    weights is a nsub x nchan array of weights.
    dedispersed=True, will save the archive as dedispered.
    state is the polarization state of the data ("Coherence" or "Stokes" for
        npol == 4, or "Intensity" for npol == 1).
    obs is the telescope code.
    quiet=True suppresses output.

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
            if len(param) == 0:
                pass
            elif param[0] == ("PSR" or "PSRJ"):
                PSR = param[1]
            elif param[0] == "RAJ":
                RAJ = param[1]
            elif param[0] == "DECJ":
                DECJ = param[1]
            elif param[0] == "DM":
                DM = np.float64(param[1])
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
        arch.set_state(state)
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
        scale=1.0, dedispersed=False, t_scat=0.0, alpha=scattering_alpha,
        scint=False, state="Stokes", obs="GBT", quiet=False):
    """
    Generate fake pulsar data written to a PSRCHIVE psrfits archive.

    Not guaranteed to work perfectly.

    modelfile is the write_model(...)-type of file specifying the
        gaussian-component model to use.
    ephemeris is the timing ephemeris to be installed.
    outfile is the desired output file name.
    nsub is the number of subintegrations in the data.
    npol is the number of polarizations in the data.
    nchan is the number of frequency channels in the data.
    nbin is the number of phase bins in the data.
    nu0 is the center frequency [MHz] of the data.
    bw is the bandwidth of the data.
    tsub is the duration of each subintegration [sec].
    phase is an arbitrary rotation [rot] to all subints, with respect to nu0.
    dDM is a dispersion measure [cm**-3 pc] added to what is given by the
        ephemeris. Dispersion occurs at infinite frequency. (??)
    start_MJD is the starting epoch of the data in PSRCHIVE MJD format.
    weights is a nsub x nchan array of weights.
    noise_std is the frequency-independent noise-level added to the data.
    scale is an arbitrary scaling to the amplitude parameters in modelfile.
    dedispersed=True, will save the archive as dedispered.
    t_scat != 0.0 convolves the data with a scattering timscale t_scat [sec].
    alpha is the scattering index.
    scint=True adds random scintillation, based on default parameters. scint
        can also be a list of parameters taken by add_scintillation.
    state is the polarization state of the data ("Coherence" or "Stokes" for
        npol == 4, or "Intensity" for npol == 1).
    obs is the telescope code.
    quiet=True suppresses output.

    Mostly written by PBD.
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
        PEPOCH = par.PEPOCH
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
                P0 = np.float64(param[1])**-1
            elif param[0] == "P0":
                P0 = np.float64(param[1])
            elif param[0] == "PEPOCH":
                PEPOCH = np.float64(param[1])
            elif param[0] == "DM":
                DM = np.float64(param[1])
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
        arch.set_state(state)
    #Fill in some subintegration attributes
    if start_MJD is None:
        start_MJD = pr.MJD(PEPOCH)
    else:
        start_MJD = pr.MJD(start_MJD)
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
    #If wanting to use PSRCHIVE's rotation scheme, uncomment the dedisperse and
    #dededisperse lines, and set rotmodel = model.
    arch.set_dedispersed(False)
    arch.dededisperse()
    if weights is None: weights = np.ones([nsub, nchan])
    isub = 0
    for subint in arch:
        P = subint.get_folding_period()
        for ipol in xrange(npol):
            name, ngauss, model = read_model(modelfile, phases, freqs, P,
                    quiet=True)
            rotmodel = rotate_data(model, -phase, -(DM+dDM), P, freqs, nu0)
            #rotmodel = model
            if t_scat:
                sk = scattering_kernel(t_scat, nu0, freqs, phases, P,
                        alpha=alpha)
                rotmodel = add_scattering(rotmodel, sk, repeat=3)
            if scint is not False:
                if scint is True:
                    rotmodel = add_scintillation(rotmodel, random=True, nsin=3,
                            amax=1.0, wmax=5.0)
                else:
                    rotmodel = add_scintillation(rotmodel, scint)
            for ichan in xrange(nchan):
                subint.set_weight(ichan, weights[isub, ichan])
                prof = subint.get_Profile(ipol, ichan)
                noise = noise_std[ichan]
                if noise:
                    prof.get_amps()[:] = scale*rotmodel[ichan] + \
                            np.random.normal(0.0, noise, nbin)
                else:
                    prof.get_amps()[:] = scale*rotmodel[ichan]
        isub += 1
    if dedispersed: arch.dedisperse()
    arch.unload(outfile)
    if not quiet: print "\nUnloaded %s.\n"%outfile

def filter_TOAs(TOAs, flag, cutoff, criterion=">=", pass_unflagged=False):
    """
    Filter TOAs based on a flag and cutoff value.

    TOAs is a TOA list from pptoas.
    flag is a string specifying what attribute of the toa is filtered.
    cutoff is the cutoff value for the flag.
    criterion is a string specifying the condition e.g. '>', '<=', etc.
    pass_unflagged=True will pass TOAs if they do not have the flag.
    """
    new_toas = []
    for toa in TOAs:
        if hasattr(toa, flag):
            exec("if toa.%s %s cutoff: new_toas.append(toa)"%(flag, criterion))
        else:
            if pass_unflagged: new_toas.append(toa)
    return new_toas

def write_princeton_TOA(TOA_MJDi, TOA_MJDf, TOA_err, nu_ref, dDM, obs='@',
        name=' ' * 13):
    """
    Write Princeton-style TOAs.

    Additional formats coming soon...

    Princeton Format

    columns     item
    1-1     Observatory (one-character code) '@' is barycenter
    2-2     must be blank
    16-24   Reference (not necessarily observing) frequency [MHz]
    25-44   TOA [MJD] (decimal point must be in column 30 or column 31)
    45-53   TOA uncertainty [us]
    69-78   DM correction [cm**-3 pc]

    TOA_MJDi is the integer part of the TOA's MJD epoch.
    TOA_MJDf is the fractional part of the TOA's MJD epoch.
    TOA_err is the TOA's "1-sigma" uncertainty [us].
    nu_ref is the TOA's reference frequency.
    dDM is this TOA's DM "correction" (cf. tempo's NDDM).
    obs is the observatory code.
    name is whitespace.

    Taken and tweaked from SMR's PRESTO.
    """
    if nu_ref == np.inf: nu_ref = 0.0
    #Splice together the fractional and integer MJDs
    TOA = "%5d"%int(TOA_MJDi) + ("%.13f"%TOA_MJDf)[1:]
    #if dDM != 0.0:
    print obs + " %13s %8.3f %s %8.3f              %9.5f"%(name, nu_ref, TOA,
            TOA_err, dDM)
    #else:
    #    print obs + " %13s %8.3f %s %8.3f"%(name, nu_ref, TOA, TOA_err)

def write_TOAs(TOAs, format="tempo2", SNR_cutoff=0.0, outfile=None,
        append=True):
    """
    Write formatted TOAs to file.

    TOAs is a single TOA of the TOA class from pptoas, or a list of them.
    format is one of 'tempo2', ... others coming ...
    SNR_cutoff is a value specifying which TOAs are written based on the flag
        pp_snr.
    outfile is the output file name; if None, will print to standard output.
    append=False will overwrite a file with the same name as outfile.
    """
    if format != "tempo2":
        print "Only tempo2-formatted TOAs are provided for now..."
        return 0
    if not hasattr(TOAs, "__len__"): toas = [TOAs]
    else: toas = TOAs
    toas = filter_TOAs(toas, "pp_snr", SNR_cutoff, ">=", pass_unflagged=False)
    if outfile is not None:
        if append: mode = 'a'
        else: mode = 'w'
        of = open(outfile, mode)
    for toa in toas:
        if format == "tempo2":
            toa_string = "%s %.3f %d"%(toa.archive, toa.frequency,
                    toa.MJD.intday()) + ("%.15f   %.3f  %s"%(toa.MJD.fracday(),
                            toa.TOA_error,
                            tempo_codes[toa.telescope.lower()]))[1:]
            if toa.DM is not None:
                toa_string += " -pp_dm %.7f"%toa.DM
            if toa.DM_error is not None:
                toa_string += " -pp_dme %.7f"%toa.DM_error
            for flag,value in toa.flags.iteritems():
                if hasattr(value, "lower"):
                    exec("toa_string += ' -%s %s'"%(flag, value))
                elif hasattr(value, "bit_length"):
                    exec("toa_string += ' -%s %d'"%(flag, value))
                elif flag == "pp_cov":
                    exec("toa_string += ' -%s %.1e'"%(flag, toa.flags[flag]))
                else:
                    exec("toa_string += ' -%s %.3f'"%(flag, toa.flags[flag]))
            if outfile is not None:
                toa_string += "\n"
                of.write(toa_string)
            else:
                print toa_string
    if outfile is not None: of.close()

def show_portrait(port, phases=None, freqs=None, title=None, prof=True,
        fluxprof=True, rvrsd=False, colorbar=True, savefig=False,
        aspect="auto", interpolation="none", origin="lower", extent=None,
        **kwargs):
    """
    Show a pulsar portrait.

    To be improved.

    port is the nchan x nbin pulsar portrait array.
    phases is the nbin array with phase-bin centers [rot].  Defaults to
        phase-bin indices.
    freqs is the nchan array with frequency-channel centers [MHz]. Defaults to
        frequency-channel indices.
    title is a string to be displayed.
    prof=True adds a panel showing the average pulse profile.
    fluxprof=True adds a panel showing the phase-averaged spectrum.
    rvrsd=True flips the frequency axis.
    colorbar=True adds the color bar.
    savefig specifies a string for a saved figure; will not show the plot.
    aspect sets the aspect ratio
    interpolation sets the interpolation scheme
    origin tells pyplot where to put the (0,0) point (?).
    extent is a 4-element tuple setting the (lo_phase, hi_phase, lo_freq,
        hi_freq) limits on the plot.
    **kwargs get passed to imshow.  e.g. vmin, vmax...
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
    Show a portrait, model, and residuals in a single plot.

    To be improved.

    port is the nchan x nbin pulsar portrait array.
    model is the nchan x nbin model portrait array.
    resids=None tells the function to calculate the residuals in situ.
    phases is the nbin array with phase-bin centers [rot].  Defaults to
        phase-bin indices.
    freqs is the nchan array with frequency-channel centers [MHz]. Defaults to
        frequency-channel indices.
    titles is a three-element tuple for the titles on each plot.
    rvrsd=True flips the frequency axis.
    colorbar=True adds the color bar.
    savefig specifies a string for a saved figure; will not show the plot.
    aspect sets the aspect ratio
    interpolation sets the interpolation scheme
    origin tells pyplot where to put the (0,0) point (?).
    extent is a 4-element tuple setting the (lo_phase, hi_phase, lo_freq,
        hi_freq) limits on the plot.
    **kwargs get passed to imshow.  e.g. vmin, vmax...
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
            interpolation=interpolation, vmin=im.properties()['clim'], **kwargs)
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
