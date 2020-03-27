from __future__ import division
from __future__ import print_function

from builtins import map
from builtins import range
from builtins import str
from builtins import zip

from past.utils import old_div
from pplib import *
from scipy.special import erf


#############
# pptoaslib #
#############
# pptoaslib contains all necessary functions and definitions for the new pptoas
#    capabilities: fitting for scattering time and index, nu**-4 delays, and
#    flux measurement.  Future development will merge this with pplib.
# Written by Timothy T. Pennucci (TTP; tim.pennucci@nanograv.org).

def gaussian_profile_FT(nbin, loc, wid, amp):
    """
    Return the Fourier transform of a gaussian profile with nbin phase bins.

    nbin is the number of phase bins in the profile.
    loc is the pulse phase location (0-1) [rot].
    wid is the Gaussian pulse's full width at half-max (FWHM) [rot].
    amp is the Gaussian amplitude.

    Makes use of the Fourier shift theorem and the fact that the FT of a
        Gaussian is also a Gaussian.  Accounts for windowing by using an
        analytic formula for the convolution of a Gaussian and sinc function.
        This formula cannot be evaluated far from the peak, and so is still an
        approximation.

    Note that the function returns the analytic FT sampled nbin/2 + 1 times.

    Reference: http://herschel.esac.esa.int/hcss-doc-12.0/load/hcss_urm/html \\
               /herschel.ia.numeric.toolbox.fit.SincGaussModel.html
    """
    nharm = old_div(nbin, 2) + 1
    if wid <= 0.0:
        return np.zeros(nharm, 'd')
    sigma = old_div(wid, (2 * np.sqrt(2 * np.log(2))))
    amp *= (2 * np.pi * sigma ** 2) ** 0.5
    sigma *= 2 * np.pi
    sigma = old_div(1, sigma)
    harmind = np.arange(nharm)
    snc = 1.0 / np.pi  # Distance between first two zero crossings of sinc / 2pi
    a = old_div(sigma, (snc * 2 ** 0.5))
    b = old_div(harmind, (sigma * 2 ** 0.5))
    retvals = old_div(np.exp(-b ** 2) * (erf(a - b * 1j) + erf(a + b * 1j)), 2)
    retvals *= amp * nbin
    if loc != 0.0:
        phasor = np.exp(-harmind * 2.0j * np.pi * loc)
        retvals *= phasor
    return np.nan_to_num(retvals)


def rotate_portrait_full(port, phi, DM, GM, freqs, nu_DM=np.inf,
                         nu_GM=np.inf, P=None):
    """
    Rotate and/or dedisperse a portrait.

    Positive values of phi, DM, and GM rotate the data to earlier phases
        (i.e. it "dedisperses") for freqs < nu_DM.

    When used to dediserpse, with GM=0, rotate_portrait_full is virtually
        identical to arch.dedisperse() in PSRCHIVE.

    port is a nchan x nbin array of data values.
    phi is a value specifying the amount of achromatic rotation [rot].
    DM is a value specifying the amount of rotation based on the cold-plasma
        dispersion law [cm**-3 pc].
    GM [cm**-6 pc**2 s**-1] is proportional to the square of the additional
        dispersion measure from a refracting body contributing delays
        proportional to freqs**-4.
    freqs is an array of frequencies [MHz], needed if DM or GM is provided.
    nu_DM is the reference frequency [MHz] where dispersive delay is 0.
    nu_GM is the reference frequency [MHz] where the freqs**-4 delay is 0.
    P is the pulsar period [sec]; if not provided, assumes phi is in [sec].
    """
    if P is None: P = 1.0
    port_FT = fft.rfft(port, axis=-1)
    nharm = port_FT.shape[-1]
    phis = phase_shifts(phi, DM, GM, freqs, nu_DM, nu_GM, P, False)
    phsr = phasor(phis, nharm)
    rotated_port = fft.irfft(port_FT * phsr)
    return rotated_port


def GM_from_DMc(DMc, D, a_perp):
    """
    Return the geometric delay factor GM from a 'discrete cloud' having DMc.

    DMc is the dispersion measure of the discrete cloud [cm**-3 pc].
    D is the Earth-cloud (or pulsar) distance [kpc].
    a_perp is the characteristic transverse length scale of the cloud [AU].

    e.g., see Lam et al. (2016):
    """
    c = old_div(3e10, 3.1e21)  # speed of light [cm/s / cm/kpc]
    GM = old_div(DMc ** 2 * (c * D), (2.0 * (a_perp * 4.8e-9) ** 2))
    return GM


def DMc_from_GM(GM, D, a_perp):
    """
    Return the 'discrete cloud' DM arising from a geometric delay factor GM.

    GM is the geometric delay factor resulting in a pulse delay:
        Dconst**2 * GM * nu**-4 [cm**-6 pc s**-1].
    D is the Earth-cloud (or pulsar) distance [kpc].
    a_perp is the characteristic transverse length scale of the cloud [AU].

    e.g., see Lam et al. (2016):
    """
    c = old_div(3e10, 3.1e21)  # speed of light [cm/s / cm/kpc]
    DMc = (old_div(GM * (2.0 * a_perp * (4.8e-9) ** 2), (c * D))) ** 0.5
    return DMc


def instrumental_response_FT(nbin, wid=0.0, irf_type='rect'):
    """
    Return the Fourier transform of the instrumental response.

    The choice of width and function type should reflect the combined effect of
        e.g., dispersive smearing from incoherent / incorrect dedispersion,
        profile binning, back-end time averaging, and additional postdetection
        time averaging.  See Bhat et al. (2003).

    nbin is the number of phase bins in the profile.
    wid is the width [rot] of the time-domain response; it is the width of the
        rectangle if type='rect' or the FWHM if type='gauss'.
    irf_type is the instrumental response shape, either 'rect' (rectangular) or
        'gauss' (Gaussian).

    Default (wid=0.0) is that the combined effects are negligible and this
        function will have no effect.
    """
    nharm = old_div(nbin, 2) + 1
    if wid == 0.0:
        inst_resp_FT = np.ones(nharm)
    else:
        if irf_type == 'rect':
            return np.sinc(np.arange(nharm) * wid)
        elif irf_type == 'gauss':
            gp_FT = gaussian_profile_FT(nbin, 0.0, wid, 1.0)
            gp_FT /= gp_FT[0]
            return gp_FT
        else:
            print("Unrecognized instrumental response function type '%s'." \
                  % irf_type)
            return 0


def instrumental_response_port_FT(nbin, freqs, DM=0.0, P=1.0, wids=[],
                                  irf_types=[]):
    """
    Return the Fourier transform of the combined instrumental responses.

    nbin is the number of phase bins in the profile.
    freqs in the array of frequencies [MHz] with length nchan.
    DM is the estimate of the dispersion measure [cm**-3 pc] contributing to
        smearing due to incoherent / incorrect dedispersion.
    P is the period [sec].
    wids is a list of widths [rot] of constant time-domain responses; it is the
        width of a rectangle function if type='rect' or the FWHM of a Gaussian
        if type='gauss'.
    irf_types is a list of instrumental response shapes; each entry is either
        'rect' (rectangular) or 'gauss' (Gaussian).

    Default is that the combined effects are negligible and this function will
        have no effect.
    """
    nharm = old_div(nbin, 2) + 1
    nchan = len(freqs)
    if DM == len(wids) == 0.0:
        return np.ones([nchan, nharm])
    else:
        inst_resp_port_FT = np.ones([nchan, nharm], dtype='complex_')
        for wid, irf_type in zip(wids, irf_types):
            inst_resp_port_FT *= np.tile(instrumental_response_FT(nbin, wid,
                                                                  irf_type), nchan).reshape(nchan, nharm)
        if DM:
            chan_bw = abs(freqs[1] - freqs[0])
            for ichan, freq in enumerate(freqs):
                wid = old_div(old_div(8.3e-6 * chan_bw, (old_div(freq, 1e3)) ** 3), P)
                inst_resp_port_FT[ichan] *= instrumental_response_FT(nbin, wid,
                                                                     'rect')
        return inst_resp_port_FT


def phase_shifts(phi, DM, GM, freqs, nu_DM=np.inf, nu_GM=np.inf,
                 P=None, mod=False):
    """
    Return phase delay and frequencies freqs given other parameters.

    phi is an input delay [rot] or [sec].
    DM is the dispersion measure [cm**-3 pc].
    GM [cm**-6 pc**2 s**-1] is proportional to the square of the additional
        dispersion measure from a refracting body contributing delays
        proportional to freqs**-4.
    freqs are the frequencies [MHz] at which to calculate the delays.
    nu_DM is the reference frequency [MHz] where dispersive delay is 0.
    nu_GM is the reference frequency [MHz] where the freqs**-4 delay is 0.
    P is the pulsar period [sec]; if not provided, assumes phi is in [sec].
    mod=True ensures the output delay in [rot] is on the interval [-0.5, 0.5).

    e.g., see Lam et al. (2016):
        GM = cD / (2*a_perp**2) * DMc**2

    Default behavior is for P=1.0 [sec], i.e. return delays [sec]
    """
    if P is None:
        P = 1.0
        mod = False
    const_delay = phi
    dispersive_delays = old_div(Dconst * DM * (freqs ** -2 - nu_DM ** -2), P)
    refractive_delays = old_div(Dconst ** 2 * GM * (freqs ** -4 - nu_GM ** -4), P)
    delays = const_delay + dispersive_delays + refractive_delays
    if mod:
        delays = np.where(abs(delays) >= 0.5, delays % 1, delays)
        delays = np.where(delays >= 0.5, delays - 1.0, delays)
        if not delays.shape:
            delays = np.float64(delays)
    return delays


def phase_shifts_deriv(freqs, nu_DM=np.inf, nu_GM=np.inf, P=None):
    """
    """
    if P is None: P = 1.0
    if hasattr(freqs, 'shape'):
        dphi = np.ones(len(freqs))
    else:
        dphi = 1.0
    dDM = old_div(Dconst * (freqs ** -2 - nu_DM ** -2), P)
    dGM = old_div(Dconst ** 2 * (freqs ** -4 - nu_GM ** -4), P)
    gradient = np.array([dphi, dDM, dGM])
    return gradient


def phase_shifts_2deriv(freqs, nu_GM=np.inf, P=None):
    """
    """
    hessian = np.zeros([3, 3, len(freqs)])
    return hessian


def phasor(phase_shifts, nharm):
    """
    """
    iharm = np.arange(nharm)
    phasor = np.exp(2.0j * np.pi * np.outer(phase_shifts, iharm))
    return phasor


# def scattering_times(tau, alpha, freqs, nu_tau):
#    """
#    """
#    taus = tau * (freqs/nu_tau)**alpha
#    return taus

def scattering_times_deriv(tau, freqs, nu_tau, log10_tau, scattering_times):
    """
    """
    taus = scattering_times
    if not log10_tau:
        if taus.sum():
            dtau = old_div(taus, tau)  # = (freqs/nu_tau)**alpha
        else:
            dtau = np.zeros(len(freqs))
    else:
        dtau = np.log(10.) * taus
    dalpha = np.log(old_div(freqs, nu_tau)) * taus
    gradient = np.array([dtau, dalpha])
    return gradient


def scattering_times_2deriv(tau, freqs, nu_tau, log10_tau, scattering_times,
                            scattering_times_deriv):
    """
    """
    taus = scattering_times
    dtau, dalpha = scattering_times_deriv
    if not log10_tau:
        d2tau = np.zeros(len(freqs))
        if taus.sum():
            dtaudalpha = old_div(dalpha, tau)
        else:
            dtaudalpha = np.zeros(len(freqs))
    else:
        d2tau = np.log(10.) * dtau
        dtaudalpha = np.log(10.) * dalpha
    d2alpha = np.log(old_div(freqs, nu_tau)) * dalpha
    hessian = np.array([[d2tau, dtaudalpha], [dtaudalpha, d2alpha]])
    return hessian


# def scattering_profile_FT(tau, nbin, binshift=binshift):
#    """
#    Return the Fourier transform of the scattering_profile() function.
#
#    tau is the scattering timescale [rot].
#    nbin is the number of phase bins in the profile.
#    binshift is a fudge-factor; currently has no effect.
#
#    Makes use of the analytic formulation of the FT of a one-sided exponential
#        function.  There is no windowing, since the whole scattering kernel is
#        convolved with the pulse profile.
#
#    Note that the function returns the analytic FT sampled nbin/2 + 1 times.
#    """
#    nharm = nbin/2 + 1
#    if tau == 0.0:
#        scat_prof_FT = np.ones(nharm)
#    else:
#        harmind = np.arange(nharm)
#        #harmind = np.arange(-(nharm-1), (nharm-1))
#        #scat_prof_FT = tau**-1 * (tau**-1 + 2*np.pi*1.0j*harmind)**-1
#        scat_prof_FT = (1.0 + 2*np.pi*1.0j*harmind*tau)**-1
#        #scat_prof_FT *= np.exp(-harmind * 2.0j * np.pi * binshift / nbin)
#    return scat_prof_FT

# def scattering_portrait_FT(taus, nbin, binshift=binshift):
#    """
#    """
#    nchan = len(taus)
#    nharm = nbin/2 + 1
#    if not np.any(taus):
#        scat_port_FT = np.ones([nchan, nharm])
#    else:
#        scat_port_FT = np.zeros([nchan, nharm], dtype='complex_')
#        for ichan in range(nchan):
#            scat_port_FT[ichan] = scattering_profile_FT(taus[ichan], nbin,
#                    binshift)
#    # Not sure this is needed;
#    # probably has no effect since it is multiplied with other ports w/ 0 mean
#    #scat_port_FT[:, 0] *= F0_fact
#    return scat_port_FT

def scattering_portrait_FT_deriv(scattering_times, scattering_times_deriv,
                                 scattering_portrait_FT):
    """
    """
    taus = scattering_times
    dtau, dalpha = scattering_times_deriv
    scat_port_FT = scattering_portrait_FT
    if taus.sum():
        f = (old_div((scat_port_FT * (scat_port_FT - 1.0)).T, taus)).T
        gradient = np.array([(f.T * dtau).T, (f.T * dalpha).T])
    else:
        gradient = np.zeros([2, scat_port_FT.shape[0], scat_port_FT.shape[1]])
    return gradient


def scattering_portrait_FT_2deriv(scattering_times, scattering_times_deriv,
                                  scattering_times_2deriv, scattering_portrait_FT):
    """
    """
    taus = scattering_times
    dtau, dalpha = scattering_times_deriv
    scat_port_FT = scattering_portrait_FT
    d2tau, dtaudalpha, d2alpha = scattering_times_2deriv[0, 0], \
                                 scattering_times_2deriv[0, 1], scattering_times_2deriv[1, 1]
    if taus.sum():
        H = (old_div((scat_port_FT * (scat_port_FT - 1)).T, taus ** 2)).T
        H11 = (H.T * (dtau ** 2)).T
        if dtau.sum():  # else = 0.0
            H11 *= ((2 * (scat_port_FT - 1)).T + old_div((d2tau * taus), (dtau ** 2))).T
        H22 = (H.T * (dalpha ** 2)).T
        if dalpha.sum():  # else = 0.0
            H22 *= ((2 * (scat_port_FT - 1)).T + old_div((d2alpha * taus), (dalpha ** 2))).T
        H12 = (H.T * (dtau * dalpha)).T
        if dalpha.sum() and dtau.sum():  # else = 0.0
            H12 *= ((2 * (scat_port_FT - 1)).T + \
                    old_div((dtaudalpha * taus), (dtau * dalpha))).T
        hessian = np.array([[H11, H12], [H12, H22]])
    else:
        hessian = np.zeros([2, 2, scat_port_FT.shape[0], scat_port_FT.shape[1]])
    return hessian


def abs_scattering_portrait_FT(scattering_portrait_FT):
    """
    """
    scat_port_FT = scattering_portrait_FT
    abs_scat_port_FT = np.abs(scat_port_FT) ** 2
    return abs_scat_port_FT


def abs_scattering_portrait_FT_deriv(scattering_portrait_FT,
                                     scattering_portrait_FT_deriv):
    """
    """
    scat_port_FT = scattering_portrait_FT
    scat_port_FT_deriv = scattering_portrait_FT_deriv
    gradient = 2 * np.real(scat_port_FT * np.conj(scat_port_FT_deriv))
    return gradient


def abs_scattering_portrait_FT_2deriv(scattering_portrait_FT,
                                      scattering_portrait_FT_deriv, scattering_portrait_FT_2deriv):
    """
    """
    scat_port_FT = scattering_portrait_FT
    dtau, dalpha = scattering_portrait_FT_deriv
    d2tau, dtaudalpha, d2alpha = scattering_portrait_FT_2deriv[0, 0], \
                                 scattering_portrait_FT_2deriv[0, 1], \
                                 scattering_portrait_FT_2deriv[1, 1]
    H11 = 2 * (np.abs(dtau) ** 2 + np.real(scat_port_FT * np.conj(d2tau)))
    H22 = 2 * (np.abs(dalpha) ** 2 + np.real(scat_port_FT * np.conj(d2alpha)))
    H12 = 2 * np.real((dtau * np.conj(dalpha)) + \
                      (scat_port_FT * np.conj(dtaudalpha)))
    hessian = np.array([[H11, H12], [H12, H22]])
    return hessian


def Sbp(scattering_portrait_FT, model_portrait_FT, errs_FT=None):
    """
    """
    scat_port_FT = scattering_portrait_FT
    model_port_FT = model_portrait_FT
    Sbp = np.sum(np.abs(scat_port_FT) ** 2 * np.abs(model_port_FT) ** 2, axis=-1)
    if errs_FT is not None: Sbp /= errs_FT ** 2
    return Sbp


def Sbp_deriv(abs_scattering_portrait_FT_deriv, model_portrait_FT,
              errs_FT=None):
    """
    """
    abs_scat_port_FT_deriv = abs_scattering_portrait_FT_deriv
    model_port_FT = model_portrait_FT
    gradient = np.sum(abs_scat_port_FT_deriv * np.abs(model_port_FT) ** 2,
                      axis=-1)
    gradient = np.insert(gradient, 0, np.zeros([3, gradient.shape[1]]), 0)
    if errs_FT is not None: gradient /= errs_FT ** 2
    return gradient


def Sbp_2deriv(abs_scattering_portrait_FT_2deriv, model_portrait_FT,
               errs_FT=None):
    """
    """
    abs_scat_port_FT_2deriv = abs_scattering_portrait_FT_2deriv
    model_port_FT = model_portrait_FT
    d2S = np.sum(abs_scat_port_FT_2deriv * np.abs(model_port_FT) ** 2,
                 axis=-1)
    hessian = np.zeros([5, 5, len(model_port_FT)])  # not 100% sure about this
    hessian[3:, 3:] = d2S
    if errs_FT is not None: hessian /= errs_FT ** 2
    return hessian


def Cdbp(data_portrait_FT, model_portrait_FT, scattering_portrait_FT, phasor,
         errs_FT=None):
    """
    """
    data_port_FT = data_portrait_FT
    model_port_FT = model_portrait_FT
    scat_port_FT = scattering_portrait_FT
    Cdbp = np.real(np.sum(data_port_FT * np.conj(model_port_FT) * \
                          np.conj(scat_port_FT) * phasor, axis=-1))
    if errs_FT is not None:
        Cdbp = (old_div(Cdbp.T, errs_FT ** 2)).T
    return Cdbp


def Cdbp_deriv_phase_shifts(data_portrait_FT, model_portrait_FT,
                            scattering_portrait_FT, phasor):
    """
    """
    data_port_FT = data_portrait_FT
    model_port_FT = model_portrait_FT
    scat_port_FT = scattering_portrait_FT
    iharm = np.arange(phasor.shape[1])
    Cdbp_deriv_phase_shifts = np.real(np.sum(2.0j * np.pi * iharm * \
                                             data_port_FT * np.conj(model_port_FT) * np.conj(scat_port_FT) * \
                                             phasor, axis=-1))
    return Cdbp_deriv_phase_shifts


def Cdbp_2deriv_phase_shifts(data_portrait_FT, model_portrait_FT,
                             scattering_portrait_FT, phasor):
    """
    """
    data_port_FT = data_portrait_FT
    model_port_FT = model_portrait_FT
    scat_port_FT = scattering_portrait_FT
    iharm = np.arange(phasor.shape[1])
    Cdbp_2deriv_phase_shifts = np.real(np.sum(pow(2.0j * np.pi * iharm, 2.0) * \
                                              data_port_FT * np.conj(model_port_FT) * np.conj(scat_port_FT) * \
                                              phasor, axis=-1))
    return Cdbp_2deriv_phase_shifts


def Cdbp_deriv(data_portrait_FT, model_portrait_FT,
               scattering_portrait_FT_deriv, phasor, phase_shifts_deriv,
               Cdbp_deriv_phase_shifts, errs_FT=None):
    """
    """
    data_port_FT = data_portrait_FT
    model_port_FT = model_portrait_FT
    scat_port_FT_deriv = scattering_portrait_FT_deriv
    phis_deriv = phase_shifts_deriv
    iharm = np.arange(phasor.shape[1])
    Cdbp_deriv_phis = Cdbp_deriv_phase_shifts
    dphi, dDM, dGM = Cdbp_deriv_phis * phis_deriv
    dtau, dalpha = np.real(np.sum(data_port_FT * np.conj(model_port_FT) * \
                                  np.conj(scat_port_FT_deriv) * phasor, axis=-1))
    gradient = np.array([dphi, dDM, dGM, dtau, dalpha])
    if errs_FT is not None:
        gradient = old_div(gradient, errs_FT ** 2)
    return gradient


def Cdbp_2deriv(data_portrait_FT, model_portrait_FT,
                scattering_portrait_FT_deriv, scattering_portrait_FT_2deriv, phasor,
                phase_shifts_deriv, phase_shifts_2deriv, Cdbp_deriv_phase_shifts,
                Cdbp_2deriv_phase_shifts, errs_FT=None):
    """
    """
    data_port_FT = data_portrait_FT
    nchan, nharm = data_port_FT.shape
    model_port_FT = model_portrait_FT
    scat_port_FT_deriv = scattering_portrait_FT_deriv
    nscat = scat_port_FT_deriv.shape[0]
    scat_port_FT_2deriv = scattering_portrait_FT_2deriv
    phis_deriv = phase_shifts_deriv
    nphase = phis_deriv.shape[0]
    phis_deriv_matrix = np.zeros([nphase, nphase, nchan])
    for iparam in range(nphase):
        for jparam in range(nphase):
            phis_deriv_matrix[iparam, jparam] = \
                phis_deriv[iparam] * phis_deriv[jparam]
    phis_2deriv = phase_shifts_2deriv
    iharm = np.arange(nharm)
    Cdbp_deriv_phis = Cdbp_deriv_phase_shifts
    Cdbp_2deriv_phis = Cdbp_2deriv_phase_shifts
    Cdbp_hess_phis = (Cdbp_2deriv_phis * phis_deriv_matrix) + \
                     (Cdbp_deriv_phis * phis_2deriv)
    Cdbp_hess_scat = np.real(np.sum(data_port_FT * np.conj(model_port_FT) * \
                                    np.conj(scat_port_FT_2deriv) * phasor, axis=-1))
    Cdbp_hess_cross = np.zeros([nphase, nscat, nchan])
    Cdbp_cross = np.real(np.sum(2.0j * np.pi * iharm * data_port_FT * \
                                np.conj(model_port_FT) * np.conj(scat_port_FT_deriv) * phasor,
                                axis=-1))
    for iparam in range(nphase):
        for jparam in range(nscat):
            Cdbp_hess_cross[iparam, jparam] = phis_deriv[iparam] * \
                                              Cdbp_cross[jparam]
    hessian = np.zeros([nphase + nscat, nphase + nscat, nchan])
    hessian[:nphase, :nphase] = Cdbp_hess_phis
    hessian[nphase:, nphase:] = Cdbp_hess_scat
    hessian[:nphase, nphase:] = Cdbp_hess_cross
    hessian[nphase:, :nphase] = Cdbp_hess_cross.transpose(1, 0, 2)
    if errs_FT is not None: hessian /= errs_FT ** 2
    return hessian


def fit_portrait_full_function(params, data_portrait_FT, model_portrait_FT,
                               errs_FT, P, freqs, nu_DM, nu_GM, nu_tau, fit_flags, log10_tau):
    """
    """
    phi, DM, GM, tau, alpha = params
    if log10_tau: tau = 10 ** tau
    data_port_FT = data_portrait_FT
    model_port_FT = model_portrait_FT
    nharm = data_portrait_FT.shape[-1]
    nbin = 2 * (nharm - 1)
    phis = phase_shifts(phi, DM, GM, freqs, nu_DM, nu_GM, P, mod=False)
    phsr = phasor(phis, nharm)
    taus = scattering_times(tau, alpha, freqs, nu_tau)
    scat_port_FT = scattering_portrait_FT(taus, nbin, binshift=binshift)
    S = Sbp(scat_port_FT, model_port_FT, errs_FT)
    C = Cdbp(data_port_FT, model_port_FT, scat_port_FT, phsr, errs_FT)
    chi2_prime = -(old_div(C ** 2, S)).sum()  # without data term Sd
    return chi2_prime


def fit_portrait_full_function_deriv(params, data_portrait_FT,
                                     model_portrait_FT, errs_FT, P, freqs, nu_DM, nu_GM, nu_tau,
                                     fit_flags, log10_tau):
    """
    """
    phi, DM, GM, tau, alpha = params
    if log10_tau: tau = 10 ** tau
    data_port_FT = data_portrait_FT
    model_port_FT = model_portrait_FT
    nharm = data_portrait_FT.shape[-1]
    nbin = 2 * (nharm - 1)
    phis = phase_shifts(phi, DM, GM, freqs, nu_DM, nu_GM, P, mod=False)
    phis_deriv = phase_shifts_deriv(freqs, nu_DM, nu_GM, P)
    phsr = phasor(phis, nharm)
    taus = scattering_times(tau, alpha, freqs, nu_tau)
    taus_deriv = scattering_times_deriv(tau, freqs, nu_tau, log10_tau, taus)
    scat_port_FT = scattering_portrait_FT(taus, nbin, binshift=binshift)
    scat_port_FT_deriv = scattering_portrait_FT_deriv(taus, taus_deriv,
                                                      scat_port_FT)
    abs_scat_port_FT_deriv = abs_scattering_portrait_FT_deriv(scat_port_FT,
                                                              scat_port_FT_deriv)
    S = Sbp(scat_port_FT, model_port_FT, errs_FT)
    dS = Sbp_deriv(abs_scat_port_FT_deriv, model_port_FT, errs_FT)
    C = Cdbp(data_port_FT, model_port_FT, scat_port_FT, phsr, errs_FT)
    dCdphi = Cdbp_deriv_phase_shifts(data_port_FT, model_port_FT, scat_port_FT,
                                     phsr)
    dC = Cdbp_deriv(data_port_FT, model_port_FT, scat_port_FT_deriv, phsr,
                    phis_deriv, dCdphi, errs_FT)
    gradient = -((old_div(C ** 2, S)) * ((old_div(2 * dC, C) - old_div(dS, S)))).sum(axis=-1)
    gradient *= fit_flags  # not sure about this
    return gradient


def fit_portrait_full_function_2deriv(params, data_portrait_FT,
                                      model_portrait_FT, errs_FT, P, freqs, nu_DM, nu_GM, nu_tau,
                                      fit_flags, log10_tau, per_channel=False,
                                      return_covariance_matrix=False, return_scales=False):
    """
    """
    returns = []
    phi, DM, GM, tau, alpha = params
    if log10_tau: tau = 10 ** tau
    data_port_FT = data_portrait_FT
    model_port_FT = model_portrait_FT
    nharm = data_portrait_FT.shape[-1]
    nbin = 2 * (nharm - 1)
    phis = phase_shifts(phi, DM, GM, freqs, nu_DM, nu_GM, P, mod=False)
    phis_deriv = phase_shifts_deriv(freqs, nu_DM, nu_GM, P)
    phis_2deriv = phase_shifts_2deriv(freqs, nu_GM, P)
    phsr = phasor(phis, nharm)
    taus = scattering_times(tau, alpha, freqs, nu_tau)
    taus_deriv = scattering_times_deriv(tau, freqs, nu_tau, log10_tau, taus)
    taus_2deriv = scattering_times_2deriv(tau, freqs, nu_tau, log10_tau, taus,
                                          taus_deriv)
    scat_port_FT = scattering_portrait_FT(taus, nbin, binshift=binshift)
    scat_port_FT_deriv = scattering_portrait_FT_deriv(taus, taus_deriv,
                                                      scat_port_FT)
    scat_port_FT_2deriv = scattering_portrait_FT_2deriv(taus, taus_deriv,
                                                        taus_2deriv, scat_port_FT)
    abs_scat_port_FT_deriv = abs_scattering_portrait_FT_deriv(scat_port_FT,
                                                              scat_port_FT_deriv)
    abs_scat_port_FT_2deriv = abs_scattering_portrait_FT_2deriv(scat_port_FT,
                                                                scat_port_FT_deriv, scat_port_FT_2deriv)
    S = Sbp(scat_port_FT, model_port_FT, errs_FT)
    dS = Sbp_deriv(abs_scat_port_FT_deriv, model_port_FT, errs_FT)
    d2S = Sbp_2deriv(abs_scat_port_FT_2deriv, model_port_FT, errs_FT)
    C = Cdbp(data_port_FT, model_port_FT, scat_port_FT, phsr, errs_FT)
    dCdphi = Cdbp_deriv_phase_shifts(data_port_FT, model_port_FT, scat_port_FT,
                                     phsr)
    d2Cdphi = Cdbp_2deriv_phase_shifts(data_port_FT, model_port_FT,
                                       scat_port_FT, phsr)
    dC = Cdbp_deriv(data_port_FT, model_port_FT, scat_port_FT_deriv, phsr,
                    phis_deriv, dCdphi, errs_FT)
    d2C = Cdbp_2deriv(data_port_FT, model_port_FT, scat_port_FT_deriv,
                      scat_port_FT_2deriv, phsr, phis_deriv, phis_2deriv, dCdphi,
                      d2Cdphi, errs_FT)
    scales = old_div(C, S)  # maximum-likelihood values for amplitude parameters
    hessian = np.zeros([5, 5, len(freqs)])
    for iparam in range(5):
        for jparam in range(5):  # repeating calculations even though symmetric
            # covariances with a_n are accounted for in the below entries
            Hij_n = -2 * ((old_div(C ** 2, S)) * \
                          ((old_div(d2C[iparam, jparam], C)) - (old_div(0.5 * d2S[iparam, jparam], S)) + \
                           (old_div(dC[iparam] * dC[jparam], C ** 2)) + \
                           (old_div(dS[iparam] * dS[jparam], S ** 2)) - \
                           old_div(((dC[iparam] * dS[jparam]) + (dS[iparam] * dC[jparam])), (C * S))))
            hessian[iparam, jparam] = Hij_n * \
                                      fit_flags[iparam] * fit_flags[jparam]
    if not per_channel: hessian = hessian.sum(axis=-1)
    returns.append(hessian)
    if return_covariance_matrix:
        if per_channel: hessian = hessian.sum(axis=-1)
        ifit = np.where(fit_flags)[0]  # which parameters are fit
        covariance_matrix = np.linalg.inv(0.5 * hessian[ifit].T[ifit].T)
        returns.append(covariance_matrix)
    if return_scales:
        returns.append(scales)
    if len(returns) == 1:
        return hessian
    else:
        return tuple(returns)


def fit_portrait_full_function_2deriv_with_scales(params, data_portrait_FT,
                                                  model_portrait_FT, errs_FT, P, freqs, nu_DM, nu_GM, nu_tau,
                                                  fit_flags, log10_tau, per_channel=False,
                                                  return_covariance_matrix=False, return_scales=False):
    """
    """
    returns = []
    phi, DM, GM, tau, alpha = params
    if log10_tau: tau = 10 ** tau
    data_port_FT = data_portrait_FT
    model_port_FT = model_portrait_FT
    nharm = data_portrait_FT.shape[-1]
    nbin = 2 * (nharm - 1)
    phis = phase_shifts(phi, DM, GM, freqs, nu_DM, nu_GM, P, mod=False)
    phis_deriv = phase_shifts_deriv(freqs, nu_DM, nu_GM, P)
    phis_2deriv = phase_shifts_2deriv(freqs, nu_GM, P)
    phsr = phasor(phis, nharm)
    taus = scattering_times(tau, alpha, freqs, nu_tau)
    taus_deriv = scattering_times_deriv(tau, freqs, nu_tau, log10_tau, taus)
    taus_2deriv = scattering_times_2deriv(tau, freqs, nu_tau, log10_tau, taus,
                                          taus_deriv)
    scat_port_FT = scattering_portrait_FT(taus, nbin, binshift=binshift)
    scat_port_FT_deriv = scattering_portrait_FT_deriv(taus, taus_deriv,
                                                      scat_port_FT)
    scat_port_FT_2deriv = scattering_portrait_FT_2deriv(taus, taus_deriv,
                                                        taus_2deriv, scat_port_FT)
    abs_scat_port_FT_deriv = abs_scattering_portrait_FT_deriv(scat_port_FT,
                                                              scat_port_FT_deriv)
    abs_scat_port_FT_2deriv = abs_scattering_portrait_FT_2deriv(scat_port_FT,
                                                                scat_port_FT_deriv, scat_port_FT_2deriv)
    S = Sbp(scat_port_FT, model_port_FT, errs_FT)
    dS = Sbp_deriv(abs_scat_port_FT_deriv, model_port_FT, errs_FT)
    d2S = Sbp_2deriv(abs_scat_port_FT_2deriv, model_port_FT, errs_FT)
    C = Cdbp(data_port_FT, model_port_FT, scat_port_FT, phsr, errs_FT)
    dCdphi = Cdbp_deriv_phase_shifts(data_port_FT, model_port_FT, scat_port_FT,
                                     phsr)
    d2Cdphi = Cdbp_2deriv_phase_shifts(data_port_FT, model_port_FT,
                                       scat_port_FT, phsr)
    dC = Cdbp_deriv(data_port_FT, model_port_FT, scat_port_FT_deriv, phsr,
                    phis_deriv, dCdphi, errs_FT)
    d2C = Cdbp_2deriv(data_port_FT, model_port_FT, scat_port_FT_deriv,
                      scat_port_FT_2deriv, phsr, phis_deriv, phis_2deriv, dCdphi,
                      d2Cdphi, errs_FT)
    scales = old_div(C, S)  # maximum-likelihood values for amplitude parameters
    hessian = np.zeros([5 + len(freqs), 5 + len(freqs), len(freqs)])
    cross_hess = -2 * (dC - scales * dS)
    for iparam in range(5):
        for jparam in range(5):  # repeating calculations even though symmetric
            # a_n terms calculated
            Hij_n = -2 * ((old_div(C ** 2, S)) * \
                          ((old_div(d2C[iparam, jparam], C)) - (old_div(0.5 * d2S[iparam, jparam], S))))
            hessian[iparam, jparam] = Hij_n * \
                                      fit_flags[iparam] * fit_flags[jparam]
    for ichan in range(len(freqs)):  # diagonal a_n matrix and cross terms
        iparam = 5 + ichan  # a_n params
        Hij_n = np.zeros(len(freqs))
        Hij_n[ichan] = 2 * S[ichan]
        hessian[iparam, iparam] = Hij_n
        for jparam in range(5):  # fit params
            hessian[iparam, jparam, ichan] = hessian[jparam, iparam, ichan] = \
                cross_hess[jparam, ichan] * fit_flags[jparam]
    if not per_channel: hessian = hessian.sum(axis=-1)
    returns.append(hessian)
    if return_covariance_matrix:
        # see Woodbury Matrix Identity for below;
        # use block-wise inversion via LDU decomp.
        if per_channel: hessian = hessian.sum(axis=-1)
        ifit = np.where(fit_flags)[0]  # which parameters are fit
        A = hessian[ifit].T[ifit].T
        C_inv = np.identity(len(scales)) * (2 * S) ** -1
        U = cross_hess[ifit]
        V = U.T
        X_inv = np.linalg.inv(A - np.dot(np.dot(U, C_inv), V))
        UL = X_inv
        UR = -np.dot(np.dot(X_inv, U), C_inv)
        LL = -np.dot(np.dot(C_inv, V), X_inv)
        LR = -np.dot(np.dot(LL, U), C_inv) + C_inv
        covariance_matrix = np.append(np.append(UL, UR, 1),
                                      np.append(LL, LR, 1), 0)
        covariance_matrix *= 2.0  # (0.5 * hessian)**-1
        returns.append(covariance_matrix)
    if return_scales:
        returns.append(scales)
    if len(returns) == 1:
        return hessian
    else:
        return tuple(returns)


def get_nu_zeros(params, data_portrait_FT, model_portrait_FT, errs_FT, P,
                 freqs, nu_DM, nu_GM, nu_tau, fit_flags, log10_tau, option=0):
    """
    """
    phi, DM, GM, tau, alpha = params
    if log10_tau: tau = 10 ** tau
    Hij_n = fit_portrait_full_function_2deriv(params, data_portrait_FT,
                                              model_portrait_FT, errs_FT, P, freqs, nu_DM, nu_GM, nu_tau,
                                              fit_flags, log10_tau, per_channel=True, return_covariance_matrix=False,
                                              return_scales=False)
    phis_deriv = phase_shifts_deriv(freqs, nu_DM, nu_GM, P)
    taus = scattering_times(tau, alpha, freqs, nu_tau)
    taus_deriv = scattering_times_deriv(tau, freqs, nu_tau, log10_tau, taus)
    if np.all(fit_flags == [1, 1, 0, 0, 0]):  # only phi and DM, 'original'
        Hij_n = Hij_n[:2, :2]
        H21_n = old_div(Hij_n[0, 1], phis_deriv[1])  # (freqs**-2 - nu_DM**-2)
        numer = (freqs ** -2 * H21_n).sum()
        denom = H21_n.sum()
        nu_zero_DM = (old_div(numer, denom)) ** -0.5
        nu_zero_GM, nu_zero_tau = nu_GM, nu_tau
    elif np.all(fit_flags == [1, 0, 1, 0, 0]):  # only phi and GM, not likely
        Hij_n = np.delete(np.delete(Hij_n, 1, 0), 1, 1)
        Hij_n = Hij_n[:2, :2]
        H21_n = old_div(Hij_n[0, 1], phis_deriv[2])  # (freqs**-4 - nu_GM**-4)
        numer = (freqs ** -4 * H21_n).sum()
        denom = H21_n.sum()
        nu_zero_GM = (old_div(numer, denom)) ** -0.25
        nu_zero_DM, nu_zero_tau = nu_DM, nu_tau
    elif np.all(fit_flags == [0, 0, 0, 1, 1]):  # only tau and alpha, not likely
        nu_zero_DM, nu_zero_GM = nu_DM, nu_GM
        Hij_n = Hij_n[3:, 3:]
        H21_n = old_div(Hij_n[0, 1], (old_div(taus_deriv[1], taus)))  # ln(freqs/nu_tau)
        numer = (np.log(freqs) * H21_n).sum()
        denom = H21_n.sum()
        nu_zero_tau = np.exp(old_div(numer, denom))
    elif np.all(fit_flags == [1, 1, 0, 1, 0]):  # phi, DM, and tau
        Hij_n = np.delete(np.delete(Hij_n, 2, 0), 2, 1)
        Hij_n = Hij_n[:3, :3]
        H21_n, H23_n = old_div(Hij_n[1, [0, 2]], phis_deriv[1])
        Hij = Hij_n.sum(axis=-1)
        H13, H33 = Hij[2, [0, 2]]
        numer = (H13 * (freqs ** -2 * H23_n).sum()) - \
                (H33 * (freqs ** -2 * H21_n).sum())
        denom = (H13 * H23_n.sum()) - (H33 * H21_n.sum())
        nu_zero_DM = (old_div(numer, denom)) ** -0.5
        nu_zero_GM, nu_zero_tau = nu_GM, nu_tau
    elif np.all(fit_flags == [1, 1, 1, 0, 0]):  # phi, DM, and GM, no scattering
        Hij_n = Hij_n[:3, :3]
        Hij = Hij_n.sum(axis=-1)
        if option == 0:  # zero covariance b/w phi & DM
            H21_n, H23_n = old_div(Hij_n[1, [0, 2]], phis_deriv[1])
            H31_n, H33_n = old_div(Hij_n[2, [0, 2]], phis_deriv[2])
            A, B = (H31_n * freqs ** -4).sum(), H31_n.sum()
            C, D = (H23_n * freqs ** -2).sum(), H23_n.sum()
            E, F = (H33_n * freqs ** -4).sum(), H33_n.sum()
            G, H = (H21_n * freqs ** -2).sum(), H21_n.sum()
            coeffs = [(A * C - E * G), 0.0, (E * H - A * D), 0.0, (F * G - B * C), 0.0,
                      (B * D - F * H)]
            roots = np.roots(coeffs)
            roots = np.real(roots[np.where(np.imag(roots) == 0.0)[0]])
            roots = roots[np.where(roots > 0.0)[0]]
            nu_zero_DM = roots[np.argmin(abs(freqs.mean() - roots))]
            nu_zero_GM = nu_zero_DM
        elif option == 1:  # zero covariance b/w phi & GM
            H21_n, H22_n = old_div(Hij_n[1, [0, 1]], phis_deriv[1])
            H31_n, H32_n = old_div(Hij_n[2, [0, 1]], phis_deriv[2])
            A, B = (H21_n * freqs ** -4).sum(), H21_n.sum()
            C, D = (H32_n * freqs ** -2).sum(), H32_n.sum()
            E, F = (H22_n * freqs ** -4).sum(), H22_n.sum()
            G, H = (H31_n * freqs ** -2).sum(), H31_n.sum()
            coeffs = [(A * C - E * G), 0.0, (E * H - A * D), 0.0, (F * G - B * C), 0.0,
                      (B * D - F * H)]
            roots = np.roots(coeffs)
            roots = np.real(roots[np.where(np.imag(roots) == 0.0)[0]])
            roots = roots[np.where(roots > 0.0)[0]]
            nu_zero_DM = roots[np.argmin(abs(freqs.mean() - roots))]
            nu_zero_GM = nu_zero_DM
        else:
            nu_zero_DM, nu_zero_GM = nu_DM, nu_GM
        nu_zero_tau = nu_tau
    elif np.all(fit_flags == [1, 1, 0, 1, 1]):  # no GM fit
        Hij_n = np.delete(np.delete(Hij_n, 2, 0), 2, 1)
        H21_n, H23_n, H24_n = old_div(Hij_n[1, [0, 2, 3]], phis_deriv[1])
        H41_n, H42_n, H43_n = old_div(Hij_n[3, [0, 1, 2]], (old_div(taus_deriv[1], taus)))
        Hij = Hij_n.sum(axis=-1)
        H11, H22, H33, H44 = np.diag(Hij)
        H12, H13, H14 = Hij[0, 1:]
        H23, H24 = Hij[1, 2:]
        H34 = Hij[2, 3]
        numer = (H34 * H34 - H33 * H44) * ((freqs ** -2 * H21_n).sum()) + \
                (H13 * H44 - H14 * H34) * ((freqs ** -2 * H23_n).sum()) + \
                (H14 * H33 - H13 * H34) * ((freqs ** -2 * H24_n).sum())
        denom = (H34 * H34 - H33 * H44) * (H21_n.sum()) + \
                (H13 * H44 - H14 * H34) * (H23_n.sum()) + \
                (H14 * H33 - H13 * H34) * (H24_n.sum())
        nu_zero_DM = (old_div(numer, denom)) ** -0.5
        nu_zero_GM = nu_GM
        numer = (H13 * H22 - H12 * H23) * ((np.log(freqs) * H41_n).sum()) + \
                (H11 * H23 - H12 * H13) * ((np.log(freqs) * H42_n).sum()) + \
                (H12 * H12 - H11 * H22) * ((np.log(freqs) * H43_n).sum())
        denom = (H13 * H22 - H12 * H23) * (H41_n.sum()) + \
                (H11 * H23 - H12 * H13) * (H42_n.sum()) + \
                (H12 * H12 - H11 * H22) * (H43_n.sum())
        nu_zero_tau = np.exp(old_div(numer, denom))
    elif np.all(fit_flags == [1, 1, 1, 1, 0]):  # no alpha fit; maybe not right
        Hij_n = Hij_n[:4, :4]
        Hij = Hij_n.sum(axis=-1)
        if option == 0:  # zero covariance b/w phi & DM
            H21_n, H23_n, H24_n = old_div(Hij_n[1, [0, 2, 3]], (freqs ** -2 - nu_DM ** -2))
            H31_n, H33_n, H34_n = old_div(Hij_n[2, [0, 2, 3]], (freqs ** -4 - nu_GM ** -4))
            H14, H44 = Hij[3, [0, 3]]
            A, a = (freqs ** -4 * H34_n).sum(), H34_n.sum()
            B, b = (freqs ** -2 * H21_n).sum(), H21_n.sum()
            C, c = (freqs ** -4 * H31_n).sum(), H31_n.sum()
            D, d = (freqs ** -2 * H23_n).sum(), H23_n.sum()
            E, e = (freqs ** -4 * H33_n).sum(), H33_n.sum()
            F, f = (freqs ** -2 * H24_n).sum(), H24_n.sum()
            P5 = (A ** 2) * B + H44 * C * D + H14 * E * F - H44 * B * E - A * C * F - H14 * A * D
            P4 = -(A ** 2) * b - H44 * C * d - H14 * E * f + H44 * b * E + A * C * f + H14 * A * d
            P3 = -2 * A * a * B - H44 * c * D - H14 * e * F + H44 * B * e + (A * c + a * C) * F + \
                 H14 * a * D
            P2 = 2 * A * a * b + H44 * c * d + H14 * e * f - H44 * b * e - (A * c + a * C) * f - \
                 H14 * a * d
            P1 = (a ** 2) * B - a * c * F
            P0 = -(a ** 2) * b + a * c * f
            coeffs = [P5, P4, P3, P2, P1, P0]
            roots = np.roots(coeffs)
            roots = np.real(roots[np.where(np.imag(roots) == 0.0)[0]])
            roots = roots[np.where(roots > 0.0)[0]]
            roots = roots ** 0.5
            nu_zero_DM = roots[np.argmin(abs(freqs.mean() - roots))]
            nu_zero_GM = nu_zero_DM
        elif option == 1:  # zero covariance b/w phi & GM
            H21_n, H22_n, H24_n = old_div(Hij_n[1, [0, 1, 3]], (freqs ** -2 - nu_DM ** -2))
            H31_n, H32_n, H34_n = old_div(Hij_n[2, [0, 1, 3]], (freqs ** -4 - nu_GM ** -4))
            H14, H44 = Hij[3, [0, 3]]
            A, a = (freqs ** -2 * H24_n).sum(), H24_n.sum()
            B, b = (freqs ** -4 * H31_n).sum(), H31_n.sum()
            C, c = (freqs ** -2 * H21_n).sum(), H21_n.sum()
            D, d = (freqs ** -4 * H32_n).sum(), H32_n.sum()
            E, e = (freqs ** -2 * H22_n).sum(), H22_n.sum()
            F, f = (freqs ** -4 * H34_n).sum(), H34_n.sum()
            P4 = (A ** 2) * B + H44 * C * D + H14 * E * F - H44 * B * E - A * C * F - H14 * A * D
            P3 = -2 * A * a * B - H44 * c * D - H14 * e * F + H44 * B * e + (A * c + a * C) * F + \
                 H14 * a * D
            P2 = -((A ** 2) * b - (a ** 2) * B) - H44 * C * d - H14 * E * f + H44 * b * E + \
                 (A * C * f - a * c * F) + H14 * A * d
            P1 = 2 * A * a * b + H44 * c * d + H14 * e * f - H44 * b * e - (A * c + a * C) * f - \
                 H14 * a * d
            P0 = -(a ** 2) * b + a * c * f
            coeffs = [P4, P3, P2, P1, P0]
            roots = np.roots(coeffs)
            roots = np.real(roots[np.where(np.imag(roots) == 0.0)[0]])
            roots = roots[np.where(roots > 0.0)[0]]
            roots = roots ** 0.5
            nu_zero_DM = roots[np.argmin(abs(freqs.mean() - roots))]
            nu_zero_GM = nu_zero_DM
        else:
            nu_zero_DM, nu_zero_GM = nu_DM, nu_GM
        nu_zero_tau = nu_tau
    elif np.all(fit_flags == [1, 1, 1, 1, 1]):  # all fit, but using [1,1,0,1,1]
        # In principle, we can work this out, but I haven't found any resource
        # willing to write down the explicit formulation of the inverse of a
        # 5x5 matrix as a function of the individual matrix entries.  Plus,
        # the algebra to find the nu_zeros would be even more laborious...
        print("Approximating zero-covariance frequencies...")
        nu_zero_DM, nu_zero_GM, nu_zero_tau = get_nu_zeros(params,
                                                           data_portrait_FT, model_portrait_FT, errs_FT, P, freqs,
                                                           nu_DM,
                                                           nu_GM, nu_tau, [1, 1, 0, 1, 1], log10_tau, option)
    else:
        if np.sum(fit_flags) > 1:
            print("No zero-covariance frequencies found.")
        nu_zero_DM, nu_zero_GM, nu_zero_tau = nu_DM, nu_GM, nu_tau
    return [nu_zero_DM, nu_zero_GM, nu_zero_tau]


def get_scales_full(params, data_portrait_FT, model_portrait_FT, errs_FT, P,
                    freqs, nu_DM, nu_GM, nu_tau, log10_tau):
    """
    Return the maximum-likelihood, per-channel scaling amplitudes.
    """
    phi, DM, GM, tau, alpha = params
    if log10_tau: tau = 10 ** tau
    data_port_FT = data_portrait_FT
    model_port_FT = model_portrait_FT
    nharm = data_portrait_FT.shape[-1]
    nbin = 2 * (nharm - 1)
    phis = phase_shifts(phi, DM, GM, freqs, nu_DM, nu_GM, P, mod=False)
    phsr = phasor(phis, nharm)
    taus = scattering_times(tau, alpha, freqs, nu_tau)
    scat_port_FT = scattering_portrait_FT(taus, nbin, binshift=binshift)
    S = Sbp(scat_port_FT, model_port_FT, errs_FT)
    C = Cdbp(data_port_FT, model_port_FT, scat_port_FT, phsr, errs_FT)
    scales = old_div(C, S)
    return scales


def fit_portrait_full(data_port, model_port, init_params, P, freqs,
                      nu_fits=[None, None, None], nu_outs=[None, None, None], errs=None,
                      fit_flags=[1, 1, 1, 1, 1], bounds=[(None, None), (None, None),
                                                         (None, None), (None, None), (None, None)], log10_tau=True,
                      option=0, sub_id=None, method='trust-ncg', is_toa=True, quiet=True):
    """
    Fit a phase offset, DM, GM, tau, & alpha between data and model portraits.

    A truncated Newtonian algorithm is used, with FFT's of input data.
    Returns an object containing the fitted parameter values, the parameter
    errors, and other attributes.

    data is the nchan x nbin phase-frequency data portrait.
    model is the nchan x nbin phase-frequency model portrait.
    init_params is a list of initial parameter guesses =
        [phase, DM, GM, tau, alpha], with phase in [rot], DM in [cm**-3 pc],
        GM in [cm**-6 pc**2 s**-1], tau in [rot] or [log10(rot)] (dependeing
        on log10_tau), and alpha dimensionless.
    P is the period [s] of the pulsar at the data epoch.
    freqs is an nchan array of frequencies [MHz].
    nu_fits = [nu_fit_DM, nu_fit_GM, nu_fit_tau] are the frequencies [MHz]
        used as nu_DM, nu_GM, and nu_tau in the fit.  Defaults to the mean
        value of freqs.
    nu_outs = [nu_out_DM, nu_out_GM, nu_out_tau] are the desired output
        reference frequencies [MHz]. Defaults to the zero-covariance
        frequencies.
    errs is the array of uncertainties on the data values (time-domain); they
        are measured if None.
    fit_flags is a list; any non-zero entries are interpreted as True.
    bounds is the list of 2 tuples containing the bounds on phi, DM, GM, tau,
        and alpha; only used if method='TNC'.
    log10_tau = True fits for the log10 of the scattering timescale.
    option = 0 is for zero covariance between phi and DM, and option = 1 is for
        zero covariance between phi and GM, when appropriate.
    sub_id provides a label for the subintegration being fit.
    method is the scipy.optimize.minimize method; currently can be 'TNC',
        'Newton-CG', or 'trust-cng', which are all Newton Conjugate-Gradient
        algorithms.  Only for 'TNC' are the bounds applied, and only for the
        other two is the second derivative function used in the fit.  Old
        pptoas used 'TNC', but 'trust-cng' seems fastest.
    is_toa=True makes sure the phi parameter corresponds to a phase shift at
        some specific frequency.
    quiet = False produces more diagnostic output.
    """
    ifit = np.where(fit_flags)[0]  # which parameters are fit
    nfit = len(ifit)  # not including the scale parameters a_n
    dof = len(data_port.ravel()) - (nfit + len(freqs))
    nbin = data_port.shape[-1]
    data_port_FT = fft.rfft(data_port, axis=-1)
    data_port_FT[:, 0] *= F0_fact
    model_port_FT = fft.rfft(model_port, axis=-1)
    model_port_FT[:, 0] *= F0_fact
    if errs is None:
        errs_FT = get_noise(data_port, chans=True) * \
                  np.sqrt(len(data_port[0]) / 2.0)
    else:
        errs_FT = errs * np.sqrt(len(data_port[0]) / 2.0)
    Sd = (old_div((np.abs(data_port_FT) ** 2).T, errs_FT ** 2.0)).T.sum()
    nu_fit_DM, nu_fit_GM, nu_fit_tau = nu_fits
    if nu_fit_DM is None: nu_fit_DM = freqs.mean()
    if nu_fit_GM is None: nu_fit_GM = freqs.mean()
    if nu_fit_tau is None: nu_fit_tau = freqs.mean()
    # BEWARE BELOW! Order matters!
    other_args = (data_port_FT, model_port_FT, errs_FT, P, freqs, nu_fit_DM,
                  nu_fit_GM, nu_fit_tau, list(map(bool, fit_flags)), log10_tau)
    minimize = opt.minimize
    jac = fit_portrait_full_function_deriv
    if method != 'TNC':
        bounds = None
        hess = fit_portrait_full_function_2deriv
    else:
        bounds = bounds
        hess = None
    if method == 'trust-ncg':
        options = {'gtol': -1}  # will result mostly in return_code = 2
    elif method == 'Newton-CG':
        options = {'maxiter': 2000, 'disp': False, 'xtol': -1}
    elif method == 'TNC':
        minfev = dof - Sd  # minimum function value estimate
        options = {'maxiter': 2000, 'disp': False, 'xtol': 1e-10, 'minfev': minfev}
    else:
        print("Method '%s' is not implemented." % method)
        sys.exit()
    start = time.time()
    results = minimize(fit_portrait_full_function, init_params,
                       args=other_args, method=method, jac=jac, hess=hess, bounds=bounds,
                       options=options)
    duration = time.time() - start
    phi_fit, DM_fit, GM_fit, tau_fit, alpha_fit = results.x
    nfeval = results.nfev
    return_code = results.status
    rcstring = RCSTRINGS["%s" % str(return_code)]
    # If the fit fails...????  These don't seem to be great indicators of the
    # fit failing
    if results.success is not True and results.status not in [1, 2, 4]:
        if sub_id is not None:
            ii = sub_id[::-1].index("_")
            isub = sub_id[-ii:]
            filename = sub_id[:-ii - 1]
            sys.stderr.write(
                "Fit 'failed' with return code %d: %s -- %s subint %s\n" % (
                    results.status, rcstring, filename, isub))
        else:
            sys.stderr.write(
                "Fit 'failed' with return code %d -- %s" % (results.status,
                                                            rcstring))
    if not quiet and results.success is True and 0:
        sys.stderr.write("Fit 'succeeded' with return code %d -- %s\n"
                         % (results.status, rcstring))
    # Curvature matrix = 1/2 2deriv of chi2 (cf. Gregory sect 11.5)
    # Parameter errors are related to curvature matrix by **-0.5
    # Calculate nu_zeros
    nu_out_DM, nu_out_GM, nu_out_tau = nu_outs
    if not bool(np.all(nu_outs)):
        nu_zero_DM, nu_zero_GM, nu_zero_tau = get_nu_zeros(results.x,
                                                           data_port_FT, model_port_FT, errs_FT, P, freqs, nu_fit_DM,
                                                           nu_fit_GM, nu_fit_tau, fit_flags, log10_tau, option=option)
        if nu_out_DM is None: nu_out_DM = nu_zero_DM
        if nu_out_GM is None: nu_out_GM = nu_zero_GM
        if nu_out_tau is None: nu_out_tau = nu_zero_tau
    if is_toa:  # Necesssary for phi to be interpreted as TOA if both DM&GM fit
        if fit_flags[1]:
            nu_out_GM = nu_out_DM
        elif fit_flags[2]:
            nu_out_DM = nu_out_GM

    phi_inf = phase_shifts(phi_fit, DM_fit, GM_fit, np.inf, nu_fit_DM,
                           nu_fit_GM, P, False)
    phi_out = phi_inf + ((old_div(Dconst, P)) * DM_fit * nu_out_DM ** -2) + \
              ((old_div(Dconst ** 2, P)) * GM_fit * nu_out_GM ** -4)
    if abs(phi_out) >= 0.5: phi_out %= 1
    if phi_out >= 0.5: phi_out -= 1.0

    if log10_tau:
        tau_fit = 10 ** tau_fit
    tau_out = scattering_times(tau_fit, alpha_fit, nu_out_tau, nu_fit_tau)
    taus = scattering_times(tau_out, alpha_fit, freqs, nu_out_tau)
    if log10_tau:
        tau_fit = np.log10(tau_fit)
        tau_out = np.log10(tau_out)
    params = [phi_out, DM_fit, GM_fit, tau_out, alpha_fit]
    param_errs = np.zeros(len(params))
    # Calculate Hessian
    Hij, covariance_matrix, scales = \
        fit_portrait_full_function_2deriv_with_scales(params,
                                                      data_port_FT, model_port_FT, errs_FT, P, freqs, nu_out_DM,
                                                      nu_out_GM, nu_out_tau, fit_flags, log10_tau,
                                                      per_channel=False, return_covariance_matrix=True,
                                                      return_scales=True)
    all_param_errs = np.diag(covariance_matrix) ** 0.5
    param_errs[ifit], scale_errs = all_param_errs[:nfit], all_param_errs[nfit:]
    covariance_matrix = covariance_matrix[:nfit].T[:nfit].T
    # SNR of the fit, based on PDB's notes
    scat_port_FT = scattering_portrait_FT(taus, nbin, binshift=binshift)
    S = Sbp(scat_port_FT, model_port_FT, errs_FT)
    channel_snrs = scales * np.sqrt(S)
    snr = pow(np.sum(channel_snrs ** 2), 0.5)
    # snr = pow(np.sum(scales**2.0 * S), 0.5)  # same as above
    chi2 = Sd + results.fun
    red_chi2 = old_div(chi2, dof)
    fit_port_results = DataBunch(params=params, param_errs=param_errs,
                                 # phase=phi_out, phase_err=param_errs[0], DM=DM_fit,
                                 phi=phi_out, phi_err=param_errs[0], DM=DM_fit,
                                 DM_err=param_errs[1], GM=GM_fit, GM_err=param_errs[2],
                                 tau=tau_out, tau_err=param_errs[3], alpha=alpha_fit,
                                 alpha_err=param_errs[4], scales=scales, scale_errs=scale_errs,
                                 nu_DM=nu_out_DM, nu_GM=nu_out_GM, nu_tau=nu_out_tau,
                                 covariance_matrix=covariance_matrix, chi2=chi2, red_chi2=red_chi2,
                                 snr=snr, channel_snrs=channel_snrs, duration=duration,
                                 nfeval=nfeval, return_code=return_code)
    return fit_port_results
