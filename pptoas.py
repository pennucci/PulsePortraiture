#!/usr/bin/env python

##########
# pptoas #
##########

#pptoas is a command-line program used to simultaneously fit for phases
#    (phis/TOAs), dispersion measures (DMs), frequency**-4 delays (GMs),
#    scattering timescales (taus), and scattering indices (alphas).  Mean flux
#    densities can also be estimated from the fitted model.  Full-functionality
#    is obtained when using pptoas within an interactive python environment.

#Written by Timothy T. Pennucci (TTP; tim.pennucci@nanograv.org).
#Contributions by Scott M. Ransom (SMR) and Paul B. Demorest (PBD).

from pptoaslib import *

#cfitsio defines a maximum number of files (NMAXFILES) that can be opened in
#the header file fitsio2.h.  Without calling unload() with PSRCHIVE, which
#touches the archive, I am not sure how to close the files.  So, to avoid the
#loop crashing, set a maximum number of archives for pptoas.  Modern machines
#should be able to handle almost 1000.
max_nfile = 999

#See F0_fact in pplib.py
if F0_fact:
    rm_baseline = True
else:
    rm_baseline = False

class TOA:

    """
    TOA class bundles common TOA attributes together with useful functions.
    """

    def __init__(self, archive, frequency, MJD, TOA_error, telescope,
            telescope_code, DM=None, DM_error=None, flags={}):
        """
        Form a TOA.

        archive is the string name of the TOA's archive.
        frequency is the reference frequency [MHz] of the TOA.
        MJD is a PSRCHIVE MJD object (the TOA, topocentric).
        TOA_error is the TOA uncertainty [us].
        telescope is the name of the observatory.
        telescope_code is the string written on the TOA line.
        DM is the full DM [cm**-3 pc] associated with the TOA.
        DM_error is the DM uncertainty [cm**-3 pc].
        flags is a dictionary of arbitrary TOA flags
            (e.g. {'subint':0, 'be':'GUPPI'}).
        """
        self.archive = archive
        self.frequency = frequency
        self.MJD = MJD
        self.TOA_error = TOA_error
        self.telescope = telescope
        self.telescope_code = telescope_code
        self.DM = DM
        self.DM_error = DM_error
        self.flags = flags
        for flag in flags.keys():
            exec('self.%s = flags["%s"]'%(flag, flag))

    def write_TOA(self, inf_is_zero=True, outfile=None):
        """
        Print a loosely IPTA-formatted TOA to standard output or to file.
        inf_is_zero=True follows the TEMPO/2 convention of writing 0.0 MHz as
            the frequency for infinite-frequency TOAs.
        outfile is the output file name; if None, will print to standard
            output.
        """
        write_TOAs(self, inf_is_zero=inf_is_zero, outfile=outfile, append=True)

class GetTOAs:

    """
    GetTOAs is a class with methods to measure TOAs and DMs from data.
    """

    def __init__(self, datafiles, modelfile, quiet=False):
        """
        Unpack all of the data and set initial attributes.

        datafiles is either a single PSRCHIVE file name, or a name of a
            metafile containing a list of archive names.
        modelfile is a ppgauss or ppspline model file.  modelfile can also be
            an arbitrary PSRCHIVE archive, although this feature is
            *not*quite*implemented*yet*.
        quiet=True suppresses output.
        """
        if file_is_type(datafiles, "ASCII"):
            self.datafiles = [datafile[:-1] for datafile in \
                    open(datafiles, "r").readlines()]
        else:
            self.datafiles = [datafiles]
        if len(self.datafiles) > max_nfile:
            print "Too many archives.  See/change max_nfile(=%d) in pptoas.py."%max_nfile
            sys.exit()
        self.is_FITS_model = file_is_type(modelfile, "FITS")
        self.modelfile = modelfile  # the model file in use
        self.obs = []  # observatories from the observations
        self.doppler_fs = []  # PSRCHIVE Doppler factors from Earth's motion
        self.nu0s = []  # PSRCHIVE center frequency
        self.nu_fits = []  # reference frequencies for the fit
        self.nu_refs = []  # reference frequencies for the output
        self.ok_idatafiles = [] # list of indices for good/examined datafiles
        self.ok_isubs = [] # list of indices for the good subintegrations
        self.epochs = []  # PSRCHIVE midpoints of the subintegrations
        self.MJDs = []  # same as epochs, in days
        self.Ps = []  # PSRCHIVE spin period at each epoch
        self.phis = []  # the fitted phase shifts / phi parameter
        self.phi_errs = [] # their uncertainties
        self.TOAs = []  # the fitted TOA
        self.TOA_errs = []  # their uncertainties
        self.DM0s = []  # the stored PSRCHIVE header DMs
        self.DMs = []  # the fitted DMs (may include the Doppler correction)
        self.DM_errs = []  # their uncertainties
        self.DeltaDM_means = []  # fitted single mean DM-DM0
        self.DeltaDM_errs = []  # their uncertainties
        self.GMs = []  # fitted "GM" parameter, from delays that go as nu**-4
        self.GM_errs = []  # their uncertainties
        self.taus = []  # fitted scattering timescales
        self.tau_errs = []  # their uncertainties
        self.alphas = []  # fitted scattering indices
        self.alpha_errs = []  # their uncertainties
        self.scales = []  # fitted per-channel scaling parameters
        self.scale_errs = []  # their uncertainties
        self.snrs = []  # signal-to-noise ratios (S/N)
        self.channel_snrs = []  # per-channel S/Ns
        self.profile_fluxes = []  # estimated per-channel fluxes
        self.profile_flux_errs = []  # their uncertainties
        self.fluxes = []  # estimated overall fluxes
        self.flux_errs = []  # their uncertainties
        self.flux_freqs = []  # their reference frequencies
        self.red_chi2s = []  # reduced chi2 values of the fit
        self.channel_red_chi2s = []  # per-channel reduced chi2 values
        self.covariances = []  # full covariance matrices
        self.nfevals = []  # number of likelihood function evaluations
        self.rcs = []  # return codes from the fit
        self.fit_durations = []  # durations of the fit
        self.order = []  # order that datafiles are examined (deprecated)
        self.TOA_list = []  # complete, single list of TOAs
        self.zap_channels = []  # proposed channels to be zapped
        # dictionary of instrumental response characteristics
        self.instrumental_response_dict = self.ird = \
                {'DM':0.0, 'wids':[], 'irf_types':[]}
        self.quiet = quiet  # be quiet?

    def get_TOAs(self, datafile=None, tscrunch=False, nu_refs=None, DM0=None,
            bary=True, fit_DM=True, fit_GM=False, fit_scat=False,
            log10_tau=True, scat_guess=None, fix_alpha=False,
            print_phase=False, print_flux=False, print_parangle=False,
            add_instrumental_response=False, addtnl_toa_flags={},
            method='trust-ncg', bounds=None, nu_fits=None, show_plot=False,
            quiet=None):
        """
        Measure TOAs from wideband data accounting for numerous ISM effects.

        datafile defaults to self.datafiles, otherwise it is a single
            PSRCHIVE archive name
        tscrunch=True tscrunches archive before fitting (i.e. make one set of
            measurements per archive)
        nu_refs is a tuple containing two output reference frequencies [MHz],
            one for the TOAs, and the other for the scattering timescales;
            defaults to the zero-covariance frequency between the TOA and DM,
            and the scattering timescale and index, respectively.
        DM0 is the baseline dispersion measure [cm**-3 pc]; defaults to what is
            stored in each datafile.
        bary=True corrects the measured DMs, GMs, taus, and nu_ref_taus based
            on the Doppler motion of the observatory with respect to the solar
            system barycenter.
        fit_DM=False will not fit for DM; if this is the case, you might want
            to set bary to False.
        fit_GM=True will fit for a parameter ('GM') characterizing a delay term
            for each TOA that scales as nu**-4.  Will be highly covariant with
            DM.
        fit_scat=True will fit the scattering timescale and index for each TOA.
        log10_tau=True does the scattering fit with log10(scattering timescale)
            as the parameter.
        scat_guess can be a list of three numbers: a guess of the scattering
            timescale tau [s], its reference frequency [MHz], and a guess of
            the scattering index alpha.  Will be used for all archives;
            supercedes other initial values.
        fix_alpha=True will hold the scattering index fixed, in the case that
            fit_scat==True.  alpha is fixed to the value specified in the
            .gmodel file, or scattering_alpha in pplib.py if no .gmodel is
            provided.
        print_phase=True will print the fitted parameter phi and its
            uncertainty on the TOA line with the flags -phs and -phs_err.
        print_flux=True will print an estimate of the overall flux density and
            its uncertainty on the TOA line.
        print_parangle=True will print the parallactic angle on the TOA line.
        add_instrumental_response=True will account for the instrumental
            response according to the dictionary instrumental_response_dict.
        addtnl_toa_flags are pairs making up TOA flags to be written uniformly
            to all IPTA-formatted TOAs. e.g. ('pta','NANOGrav','version',0.1)
        method is the scipy.optimize.minimize method; currently can be 'TNC',
            'Newton-CG', or 'trust-cng', which are all Newton
            Conjugate-Gradient algorithms.
        bounds is a list of five 2-tuples, giving the lower and upper bounds on
            the phase, dispersion measure, GM, tau, and alpha parameters,
            respectively.  NB: this is only used if method=='TNC'.
        nu_fits is a tuple, analogous to nu_ref, where these reference
            frequencies [MHz] are used in the fit; defaults to a guess at the
            zero-covariance frequency based on signal-to-noise ratios.
        show_plot=True will show a plot of the fitted model, data, and
            residuals at the end of the fitting.
        quiet=True suppresses output.
        """
        if quiet is None: quiet = self.quiet
        already_warned = False
        warning_message = \
                "You are using an experimental functionality of pptoas!"
        self.nfit = 1
        if fit_DM: self.nfit += 1
        if fit_GM: self.nfit += 1
        if fit_scat: self.nfit += 2
        if fix_alpha: self.nfit -= 1
        self.fit_phi = True
        self.fit_DM = fit_DM
        self.fit_GM = fit_GM
        self.fit_tau = self.fit_alpha = fit_scat
        if fit_scat: self.fit_alpha = not fix_alpha
        self.fit_flags = [int(self.fit_phi), int(self.fit_DM),
                int(self.fit_GM), int(self.fit_tau), int(self.fit_alpha)]
        self.log10_tau = log10_tau
        if not fit_scat:
            self.log10_tau = log10_tau = False
        if self.fit_GM or fit_scat or self.fit_tau or self.fit_alpha:
            print warning_message
            already_warned = True
        self.scat_guess = scat_guess
        nu_ref_tuple = nu_refs
        nu_fit_tuple = nu_fits
        self.DM0 = DM0
        self.bary = bary
        start = time.time()
        tot_duration = 0.0
        if datafile is None:
            datafiles = self.datafiles
        else:
            datafiles = [datafile]
        self.tscrunch = tscrunch
        self.add_instrumental_response = add_instrumental_response
        for iarch, datafile in enumerate(datafiles):
            fit_duration = 0.0
            #Load data
            try:
                data = load_data(datafile, dedisperse=False,
                        dededisperse=False, tscrunch=tscrunch,
                        pscrunch=True, fscrunch=False, rm_baseline=rm_baseline,
                        flux_prof=False, refresh_arch=False, return_arch=False,
                        quiet=quiet)
                if not len(data.ok_isubs):
                    if not quiet:
                        print "No subints to fit for %s.  Skipping it."%\
                                datafile
                    continue
                else: self.ok_idatafiles.append(iarch)
            except RuntimeError:
                if not quiet:
                    print "Cannot load_data(%s).  Skipping it."%datafile
                continue
            #Unpack the data dictionary into the local namespace; see load_data
            #for dictionary keys.
            for key in data.keys():
                exec(key + " = data['" + key + "']")
            if source is None: source = "noname"
            #Observation info
            obs = DataBunch(telescope=telescope, backend=backend,
                    frontend=frontend)
            nu_fits = list(np.zeros([nsub, 3], dtype=np.float64))
            nu_refs = list(np.zeros([nsub, 3], dtype=np.float64))
            phis = np.zeros(nsub, dtype=np.double)
            phi_errs = np.zeros(nsub, dtype=np.double)
            TOAs = np.zeros(nsub, dtype="object")
            TOA_errs = np.zeros(nsub, dtype="object")
            DMs = np.zeros(nsub, dtype=np.float64)
            DM_errs = np.zeros(nsub, dtype=np.float64)
            GMs = np.zeros(nsub, dtype=np.float64)
            GM_errs = np.zeros(nsub, dtype=np.float64)
            taus = np.zeros(nsub, dtype=np.float64)
            tau_errs = np.zeros(nsub, dtype=np.float64)
            alphas = np.zeros(nsub, dtype=np.float64)
            alpha_errs = np.zeros(nsub, dtype=np.float64)
            scales = np.zeros([nsub, nchan], dtype=np.float64)
            scale_errs = np.zeros([nsub, nchan], dtype=np.float64)
            snrs = np.zeros(nsub, dtype=np.float64)
            channel_snrs = np.zeros([nsub, nchan], dtype=np.float64)
            profile_fluxes = np.zeros([nsub, nchan], dtype=np.float64)
            profile_flux_errs = np.zeros([nsub, nchan], dtype=np.float64)
            fluxes = np.zeros(nsub, dtype=np.float64)
            flux_errs = np.zeros(nsub, dtype=np.float64)
            flux_freqs = np.zeros(nsub, dtype=np.float64)
            red_chi2s = np.zeros(nsub, dtype=np.float64)
            covariances = np.zeros([nsub, self.nfit, self.nfit],
                    dtype=np.float64)
            nfevals = np.zeros(nsub, dtype="int")
            rcs = np.zeros(nsub, dtype="int")
            #PSRCHIVE epochs are *midpoint* of the integration
            MJDs = np.array([epochs[isub].in_days() \
                    for isub in range(nsub)], dtype=np.double)
            DM_stored = DM # same as = arch.get_dispersion_measure()
            if self.DM0 is None:
                DM0 = DM_stored
            else:
                DM0 = self.DM0
            if self.is_FITS_model:
                if not already_warned:
                    print warning_message
                    already_warned = True
                model_data = load_data(self.modelfile, dedisperse=False,
                    dededisperse=False, tscrunch=True, pscrunch=True,
                    fscrunch=False, rm_baseline=True, flux_prof=False,
                    #fscrunch=False, rm_baseline=False, flux_prof=False,
                    refresh_arch=False, return_arch=False, quiet=True)
                model = (model_data.masks * model_data.subints)[0,0]
                if model.shape[-1] != nbin:
                    print "Model nbin %d != data nbin %d for %s; skipping it."\
                            %(model.shape[-1], nbin, datafile)
                    continue
                if model_data.nchan == 1:
                    model = np.tile(model[0], len(freqs[0])).reshape(
                            len(freqs[0]), nbin)
            if not quiet:
                print "\nEach of the %d TOAs is approximately %.2f s"%(
                        len(ok_isubs), integration_length / nsub)
                print "Doing Fourier-domain least-squares fit..."
            itoa = 1
            for isub in ok_isubs:
                sub_id = datafile + "_%d"%isub
                epoch = epochs[isub]
                MJD = MJDs[isub]
                P = Ps[isub]
                if not self.is_FITS_model:
                    #Read model
                    try:
                        if not fit_scat:
                            self.model_name, self.ngauss, model = read_model(
                                    self.modelfile, phases, freqs[isub],
                                    Ps[isub],
                                    quiet=bool(quiet+(itoa-1)))
                        else:
                            self.model_name, self.ngauss, full_model = \
                                    read_model(self.modelfile, phases,
                                            freqs[isub], Ps[isub],
                                            quiet=bool(quiet+(itoa-1)))
                            self.model_name, self.model_code, \
                                    self.model_nu_ref, self.ngauss, \
                                    self.gparams, model_fit_flags, self.alpha,\
                                    model_fit_alpha = read_model(
                                            self.modelfile,
                                            quiet=bool(quiet+(itoa-1)))
                            unscat_params = np.copy(self.gparams)
                            unscat_params[1] = 0.0
                            model = unscat_model = gen_gaussian_portrait(
                                    self.model_code, unscat_params, 0.0,
                                    phases, freqs[isub], self.model_nu_ref)
                    except UnboundLocalError:
                        self.model_name, model = read_spline_model(
                                self.modelfile, freqs[isub], nbin,
                                quiet=True) #bool(quiet+(itoa-1)))
                #else:
                ##THESE FREQUENCIES WILL BE OFF IF AVERAGED CHANNELS##
                #    print model_data.freqs[0, ok_ichans[isub]] - \
                #            freqs[isub,ok_ichans[isub]]
                freqsx = freqs[isub,ok_ichans[isub]]
                weightsx = weights[isub,ok_ichans[isub]]
                portx = subints[isub,0,ok_ichans[isub]]
                modelx = model[ok_ichans[isub]]
                if add_instrumental_response and \
                        (self.ird['DM'] or len(self.ird['wids'])):
                            inst_resp_port_FT = instrumental_response_port_FT(
                                    nbin, freqsx, self.ird['DM'], P,
                                    self.ird['wids'], self.ird['irf_types'])
                            modelx = fft.irfft(inst_resp_port_FT * \
                                    fft.rfft(modelx, axis=-1), axis=-1)
                SNRsx = SNRs[isub,0,ok_ichans[isub]]
                #NB: Time-domain uncertainties below
                errs = noise_stds[isub,0,ok_ichans[isub]]
                #nu_fit is the reference frequency for parameters in the fit
                nu_mean = freqsx.mean()
                if nu_fit_tuple is None:
                    #NB: the subints are dedispersed at different nu_fit.
                    nu_fit = guess_fit_freq(freqsx, SNRsx)
                    nu_fit_DM = nu_fit_GM = nu_fit_tau = nu_fit
                else:
                    nu_fit_DM = nu_fit_GM = nu_fit_tuple[0]
                    nu_fit_tau = nu_fit_tuple[-1]
                nu_fits[isub] = [nu_fit_DM, nu_fit_GM, nu_fit_tau]
                if nu_ref_tuple is None:
                    nu_ref = None
                    nu_ref_DM = nu_ref_GM = nu_ref_tau = nu_ref
                else:
                    nu_ref_DM = nu_ref_GM = nu_ref_tuple[0]
                    nu_ref_tau = nu_ref_tuple[-1]
                    if bary and nu_ref_tau:  # from bary to topo below
                        nu_ref_tau /= doppler_factors[isub]
                nu_refs[isub] = [nu_ref_DM, nu_ref_GM, nu_ref_tau]

                ###################
                # INITIAL GUESSES #
                ###################
                DM_guess = DM_stored
                rot_port = rotate_data(portx, 0.0, DM_guess, P, freqsx,
                        nu_mean)  # why not nu_fit?
                rot_prof = np.average(rot_port, axis=0, weights=weightsx)
                GM_guess = 0.0
                tau_guess = 0.0
                alpha_guess = 0.0
                if fit_scat:
                    if self.scat_guess is not None:
                        tau_guess_s,tau_guess_ref,alpha_guess = self.scat_guess
                        tau_guess = (tau_guess_s / P) * \
                                (nu_fit_tau / tau_guess_ref)**alpha_guess
                    else:
                        if hasattr(self, 'alpha'): alpha_guess = self.alpha
                        else: alpha_guess = scattering_alpha
                        if hasattr(self, 'gparams'):
                            tau_guess = (self.gparams[1] / P) * \
                                    (nu_fit_tau/self.model_nu_ref)**alpha_guess
                        else:
                            tau_guess = 0.0  # nbin**-1?
                            #tau_guess = guess_tau(...)
                    model_prof_scat = fft.irfft(scattering_portrait_FT(
                        np.array([scattering_times(tau_guess, alpha_guess,
                            nu_fit_tau, nu_fit_tau)]), nbin)[0] * fft.rfft(
                                modelx.mean(axis=0)))
                    phi_guess = fit_phase_shift(rot_prof,
                            model_prof_scat, Ns=100).phase
                    if self.log10_tau:
                        if tau_guess == 0.0: tau_guess = nbin**-1
                        tau_guess = np.log10(tau_guess)
                else:
                    #NB: Ns should be larger than nbin for very low S/N data,
                    #especially in the case of noisy models...
                    phi_guess = fit_phase_shift(rot_prof, modelx.mean(axis=0),
                            Ns=100).phase
                phi_guess = phase_transform(phi_guess, DM_guess, nu_mean,
                        nu_fit_DM, P, mod=True) # why not use nu_fit at first?
                #Need a status bar?
                param_guesses = [phi_guess, DM_guess, GM_guess, tau_guess,
                        alpha_guess]
                if bounds is None and method == 'TNC':
                    phi_bounds = (None, None)
                    DM_bounds = (None, None)
                    GM_bounds = (None, None)
                    if not self.log10_tau: tau_bounds = (0.0, None)
                    else: tau_bounds = (np.log10((10*nbin)**-1), None)
                    alpha_bounds = (-10.0, 10.0)
                    bounds = [phi_bounds, DM_bounds, GM_bounds, tau_bounds,
                            alpha_bounds]

                ###########
                # THE FIT #
                ###########
                if not quiet: print "Fitting for TOA #%d"%(itoa)
                if len(freqsx) == 1:
                    fit_flags = [1,0,0,0,0]
                    if not quiet:
                        print "TOA #%d only has 1 frequency channel...fitting for phase only..."%(itoa)
                elif len(freqsx) == 2 and self.fit_DM and self.fit_GM:
                    # prioritize DM fit
                    fit_flags[2] = 0
                    if not quiet:
                        print "TOA #%d only has 2 frequency channels...fitting for phase and DM only..."%(itoa)
                else:
                    fit_flags = list(np.copy(self.fit_flags))
                results = fit_portrait_full(portx, modelx, param_guesses, P,
                        freqsx, nu_fits[isub], nu_refs[isub], errs, fit_flags,
                        bounds, self.log10_tau, option=0, sub_id=sub_id,
                        method=method, is_toa=True, quiet=quiet)
                # Old code
                #results = fit_portrait(portx, modelx,
                #        np.array([phi_guess, DM_guess]), P, freqsx,
                #        nu_fit_DM, nu_ref_DM, errs, bounds=bounds, id=sub_id,
                #        quiet=quiet)
                #results.phi = results.phase
                #results.phi_err = results.phase_err
                #results.GM = results.GM_err = None
                #results.tau = results.tau_err = None
                #results.alpha = results.alpha_err = None
                #results.covariance_matrix = np.zeros([2,2])
                #results.nu_DM = results.nu_GM = results.nu_tau =results.nu_ref
                # Old code for fitting just phase...
                #else:  #1-channel hack
                #    if not quiet:
                #        print "TOA only has %d frequency channel!..."%len(
                #                freqsx)
                #        print "...using Fourier phase gradient routine to fit phase only..."
                #    results = fit_phase_shift(portx[0], modelx[0], errs[0],
                #            Ns=nbin)
                #    results.phi = results.phase
                #    results.phi_err = results.phase_err
                #    results.DM = results.DM_err = None
                #    results.GM = results.GM_err = None
                #    results.tau = results.tau_err = None
                #    results.alpha = results.alpha_err = None
                #    results.nu_DM, results.nu_GM, results.nu_tau = \
                #            [freqsx[0], freqsx[0], freqsx[0]]
                #    results.nfeval = 0
                #    results.return_code = -2
                #    results.scales = np.array([results.scale])
                #    results.scale_errs = np.array([results.scale_error])
                #    results.covariance_matrix = np.identity(self.nfit)
                fit_duration += results.duration

                ####################
                #  CALCULATE  TOA  #
                ####################
                results.TOA = epoch + pr.MJD(
                        ((results.phi * P) + backend_delay) /
                        (3600 * 24.))
                results.TOA_err = results.phi_err * P * 1e6 # [us]

                ######################
                # DOPPLER CORRECTION #
                ######################
                # This correction should fix Doppler-induced annual variations
                # to DM(t), but will not fix Doppler-induced /orbital/
                # variations to DM(t).
                if self.bary: #Default is True
                    df = doppler_factors[isub]
                    if fit_flags[1]:
                        # NB: The following eqution was incorrectly reversed in
                        #     the original paper Pennucci et al. (2014),
                        #     printed as DM_bary = DM_topo / df.
                        results.DM *= df  #NB: No longer the *fitted* value!
                    if fit_flags[2]:
                        results.GM *= df**3  #NB: No longer the *fitted* value!
                else:
                    df = 1.0

                #################
                # ESTIMATE FLUX #
                #################
                if print_flux:
                    if results.tau != 0.0:
                        if self.log10_tau: tau = 10**results.tau
                        else: tau = results.tau
                        alpha = results.alpha
                        scat_model = fft.irfft(scattering_portrait_FT(
                            scattering_times(tau, alpha, freqsx,
                                results.nu_tau), data.nbin) * \
                                        fft.rfft(modelx, axis=1), axis=1)
                    else: scat_model = np.copy(modelx)
                    scat_model_means = scat_model.mean(axis=1)
                    profile_fluxes[isub, ok_ichans[isub]] = scat_model_means *\
                            results.scales
                    profile_flux_errs[isub, ok_ichans[isub]] = abs(
                            scat_model_means) * results.scale_errs
                    flux, flux_err = weighted_mean(profile_fluxes[isub,
                        ok_ichans[isub]], profile_flux_errs[isub,
                            ok_ichans[isub]])
                    flux_freq, flux_freq_err = weighted_mean(freqsx,
                            profile_flux_errs[isub, ok_ichans[isub]])
                    fluxes[isub] = flux
                    flux_errs[isub] = flux_err
                    flux_freqs[isub] = flux_freq

                nu_refs[isub] = [results.nu_DM, results.nu_GM, results.nu_tau]
                phis[isub] = results.phi
                phi_errs[isub] = results.phi_err
                TOAs[isub] = results.TOA
                TOA_errs[isub] = results.TOA_err
                DMs[isub] = results.DM
                DM_errs[isub] = results.DM_err
                GMs[isub] = results.GM
                GM_errs[isub] = results.GM_err
                taus[isub] = results.tau
                tau_errs[isub] = results.tau_err
                alphas[isub] = results.alpha
                alpha_errs[isub] = results.alpha_err
                nfevals[isub] = results.nfeval
                rcs[isub] = results.return_code
                scales[isub, ok_ichans[isub]] = results.scales
                scale_errs[isub, ok_ichans[isub]] = results.scale_errs
                snrs[isub] = results.snr
                channel_snrs[isub, ok_ichans[isub]] = results.channel_snrs
                try:
                    covariances[isub] = results.covariance_matrix
                except ValueError:
                    for ii,ifit in enumerate(np.where(fit_flags)[0]):
                        for jj,jfit in enumerate(np.where(fit_flags)[0]):
                            covariances[isub][ifit,jfit] = \
                                    results.covariance_matrix[ii,jj]
                red_chi2s[isub] = results.red_chi2
                #Compile useful TOA flags
                # Add doppler_factor?
                toa_flags = {}
                if not fit_flags[1]:
                    results.DM = None
                    results.DM_err = None
                if fit_flags[2]:
                    toa_flags['gm'] = results.GM
                    toa_flags['gm_err'] = results.GM_err
                if fit_flags[3]:
                    if self.log10_tau:
                        toa_flags['scat_time'] = 10**results.tau * P / df * 1e6
                                                 # usec, w/ df
                        toa_flags['log10_scat_time'] = results.tau + \
                                np.log10(P / df)  # w/ df
                        toa_flags['log10_scat_time_err'] = results.tau_err
                    else:
                        toa_flags['scat_time'] = results.tau * P / df * 1e6
                                                 # usec, w/ df
                        toa_flags['scat_time_err'] = results.tau_err * P / df \
                                * 1e6  # usec, w/ df
                    toa_flags['scat_ref_freq'] = results.nu_tau * df  # w/ df
                    toa_flags['scat_ind'] = results.alpha
                if fit_flags[4]:
                    toa_flags['scat_ind_err'] = results.alpha_err
                toa_flags['be'] = backend
                toa_flags['fe'] = frontend
                toa_flags['f'] = frontend + "_" + backend
                toa_flags['nbin'] = nbin
                toa_flags['nch'] = nchan
                toa_flags['nchx'] = len(freqsx)
                toa_flags['bw'] = freqsx.max() - freqsx.min()
                toa_flags['subint'] = isub
                toa_flags['tobs'] = subtimes[isub]
                toa_flags['fratio'] = freqsx.max() / freqsx.min()
                toa_flags['tmplt'] = self.modelfile
                toa_flags['snr'] = results.snr
                if (nu_ref_DM is not None and np.all(fit_flags[:2])):
                            toa_flags['phi_DM_cov'] = \
                                    results.covariance_matrix[0,1]
                toa_flags['gof'] = results.red_chi2
                if print_phase:
                    toa_flags['phs'] = results.phi
                    toa_flags['phs_err'] = results.phi_err
                if print_flux:
                    toa_flags['flux'] = fluxes[isub]
                    toa_flags['flux_err'] = flux_errs[isub] # consistent w/ pat
                    #toa_flags['fluxe'] = flux_errs[isub]  # consistent w/ pat
                    toa_flags['flux_ref_freq'] = flux_freqs[isub]
                if print_parangle:
                    toa_flags['par_angle'] = parallactic_angles[isub]
                for k,v in addtnl_toa_flags.iteritems():
                    toa_flags[k] = v
                self.TOA_list.append(TOA(datafile, results.nu_DM, results.TOA,
                    results.TOA_err, telescope, telescope_code, results.DM,
                    results.DM_err, toa_flags))
                itoa += 1

            DeltaDMs = DMs - DM0
            #The below returns the weighted mean and the sum of the weights,
            #but needs to do better in the case of small-error outliers from
            #RFI, etc.  Also, last TOA may mess things up...use median...?
            if np.all(DM_errs[ok_isubs]):
                DM_weights = DM_errs[ok_isubs]**-2
            else:
                DM_weights = np.ones(len(DM_errs[ok_isubs]))
            DeltaDM_mean, DeltaDM_var = np.average(DeltaDMs[ok_isubs],
                    weights=DM_weights, returned=True)
            DeltaDM_var = DeltaDM_var**-1
            if len(ok_isubs) > 1:
                #The below multiply by the red. chi-squared to inflate the
                #errors.
                DeltaDM_var *= np.sum(
                        ((DeltaDMs[ok_isubs] - DeltaDM_mean)**2) * DM_weights)\
                                / (len(DeltaDMs[ok_isubs]) - 1)
            DeltaDM_err = DeltaDM_var**0.5
            self.order.append(datafile)
            self.obs.append(obs)
            self.doppler_fs.append(doppler_factors)
            self.nu0s.append(nu0)
            self.nu_fits.append(nu_fits)
            self.nu_refs.append(nu_refs)
            self.ok_isubs.append(ok_isubs)
            self.epochs.append(epochs)
            self.MJDs.append(MJDs)
            self.Ps.append(Ps)
            self.phis.append(phis) #NB: phis are w.r.t. nu_ref_DM
            self.phi_errs.append(phi_errs)
            self.TOAs.append(TOAs) #NB: TOAs are w.r.t. nu_ref_DM
            self.TOA_errs.append(TOA_errs)
            self.DM0s.append(DM0)
            self.DMs.append(DMs)
            self.DM_errs.append(DM_errs)
            self.DeltaDM_means.append(DeltaDM_mean)
            self.DeltaDM_errs.append(DeltaDM_err)
            self.GMs.append(GMs)
            self.GM_errs.append(GM_errs)
            self.taus.append(taus)  #NB: taus are w.r.t. nu_ref_tau
            self.tau_errs.append(tau_errs)
            self.alphas.append(alphas)
            self.alpha_errs.append(alpha_errs)
            self.scales.append(scales)
            self.scale_errs.append(scale_errs)
            self.snrs.append(snrs)
            self.channel_snrs.append(channel_snrs)
            self.profile_fluxes.append(profile_fluxes)
            self.profile_flux_errs.append(profile_flux_errs)
            self.fluxes.append(fluxes)
            self.flux_errs.append(flux_errs)
            self.flux_freqs.append(flux_freqs)
            self.covariances.append(covariances)
            self.red_chi2s.append(red_chi2s)
            self.nfevals.append(nfevals)
            self.rcs.append(rcs)
            self.fit_durations.append(fit_duration)
            if not quiet:
                print "--------------------------"
                print datafile
                print "~%.4f sec/TOA"%(fit_duration / len(ok_isubs))
                print "Med. TOA error is %.3f us"%(np.median(
                    phi_errs[ok_isubs]) * Ps.mean() * 1e6)
            if show_plot:
                stop = time.time()
                tot_duration += stop - start
                for isub in ok_isubs:
                    self.show_fit(datafile, isub)
                start = time.time()
        if not show_plot:
            tot_duration = time.time() - start
        if not quiet and len(self.ok_isubs):
            print "--------------------------"
            print "Total time: %.2f sec, ~%.4f sec/TOA"%(tot_duration,
                    tot_duration / (np.array(map(len, self.ok_isubs)).sum()))

    def get_channels_to_zap(self, SNR_threshold=8.0, rchi2_threshold=1.3,
            iterate=True, show=False):
        """
        NB: get_TOAs(...) needs to have been called first.

        SNR_threshold is a signal-to-noise ratio value which is used to flag
            channels for zapping (cf. ppzap.py).  Channels that have a S/N
            values below (SNR_threshold**2 / nchx)**0.5, where nchx is the
            number of channels used in the fit, are added to self.zap_channels.
            NB: only operates if SNR_threshold != 0.0 (individual channels may
            have S/N < 0.0).
        rchi2_threshold is a reduced chi-squared value which is used to flag
            channels for zapping (cf. ppzap.py).  Channels that have a reduced
            chi-squared value above rchi2_threshold are added to
            self.zap_channels.
        iterate=True will iterate over the S/N cut by recalculating the
            effective single-channel S/N threshold and continuing cuts until
            no new channels are cut; this helps to ensure all TOAs will have a
            wideband TOA S/N above SNR_threshold.
        show=True will show the before/after portraits for each subint with
            proposed channels to zap.
        """
        for iarch,ok_idatafile in enumerate(self.ok_idatafiles):
            datafile = self.datafiles[ok_idatafile]
            channel_red_chi2s = []
            zap_channels = []
            for isub in self.ok_isubs[iarch]:
                red_chi2s = []
                bad_ichans = []
                port, model, ok_ichans, freqs, noise_stds = self.show_fit(
                        datafile=datafile, isub=isub, rotate=0.0, show=False,
                        return_fit=True, quiet=True)
                channel_snrs = self.channel_snrs[iarch][isub]
                channel_SNR_threshold = (SNR_threshold**2.0 / \
                        len(ok_ichans))**0.5
                for ichan,ok_ichan in enumerate(ok_ichans):
                    channel_red_chi2 = get_red_chi2(port[ok_ichan],
                            model[ok_ichan], errs=noise_stds[ok_ichan],
                            dof=len(port[ok_ichan])-2) #Not sure about dof
                    red_chi2s.append(channel_red_chi2)
                    if channel_red_chi2 > rchi2_threshold:
                        bad_ichans.append(ok_ichan)
                    elif np.isnan(channel_red_chi2):
                        bad_ichans.append(ok_ichan)
                    elif SNR_threshold and \
                            channel_snrs[ok_ichan] < channel_SNR_threshold:
                                bad_ichans.append(ok_ichan)
                    else:
                        pass
                channel_red_chi2s.append(red_chi2s)
                zap_channels.append(bad_ichans)
                if iterate and SNR_threshold and len(bad_ichans):
                    old_len = len(bad_ichans)
                    added_new = True
                    while(added_new and (len(ok_ichans)-len(bad_ichans))):
                        # recalculate threshold after removing channels
                        channel_SNR_threshold = (SNR_threshold**2.0 / \
                                (len(ok_ichans)-len(bad_ichans)))**0.5
                        for ichan,ok_ichan in enumerate(ok_ichans):
                            if ok_ichan in bad_ichans:
                                continue
                            elif channel_snrs[ok_ichan] < \
                                    channel_SNR_threshold:
                                bad_ichans.append(ok_ichan)
                            else:
                                pass
                        added_new = bool(len(bad_ichans) - old_len)
                        old_len = len(bad_ichans)
                if show and len(bad_ichans):
                    show_portrait(port, get_bin_centers(port.shape[1]),
                            title="%s, subint: %d\nbad chans: %s"%(datafile,
                                isub, bad_ichans), show=False)
                    port[bad_ichans] *= 0.0
                    show_portrait(port, get_bin_centers(port.shape[1]),
                            title="%s, subint: %d\nbad chans: %s"%(datafile,
                                isub, bad_ichans), show=True)
            self.channel_red_chi2s.append((channel_red_chi2s))
            self.zap_channels.append((zap_channels))

    def show_subint(self, datafile=None, isub=0, rotate=0.0, quiet=None):
        """
        Plot a phase-frequency portrait of a subintegration.

        datafile is a single PSRCHIVE archive name; defaults to the first one
            listed in self.datafiles.
        isub is the index of the subintegration to be displayed.
        rotate is a phase [rot] specifying the amount to rotate the portrait.
        quiet=True suppresses output.

        To be improved.
        (see show_portrait(...))
        """
        if quiet is None: quiet = self.quiet
        if datafile is None:
            datafile = self.datafiles[0]
        ifile = list(np.array(self.datafiles)[self.ok_idatafiles]).index(
                datafile)
        data = load_data(datafile, dedisperse=True,
                dededisperse=False, tscrunch=self.tscrunch,
                #pscrunch=True, fscrunch=False, rm_baseline=rm_baseline,
                pscrunch=True, fscrunch=False, rm_baseline=True,
                flux_prof=False, refresh_arch=False, return_arch=False,
                quiet=quiet)
        title = "%s ; subint %d"%(datafile, isub)
        port = data.masks[isub,0] * data.subints[isub,0]
        if rotate: port = rotate_data(port, rotate)
        show_portrait(port=port, phases=data.phases, freqs=data.freqs[isub],
                title=title, prof=True, fluxprof=True, rvrsd=bool(data.bw < 0))

    def show_fit(self, datafile=None, isub=0, rotate=0.0, show=True,
            return_fit=False, savefig=False, quiet=None):
        """
        Plot the fit results from a subintegration.

        datafile is a single PSRCHIVE archive name; defaults to the first one
            listed in self.datafiles.
        isub is the index of the subintegration to be displayed.
        rotate is a phase [rot] specifying the amount to rotate the portrait.
        quiet=True suppresses output.

        To be improved.
        (see show_residual_plot(...))
        """
        if quiet is None: quiet = self.quiet
        if datafile is None:
            datafile = self.datafiles[0]
        ifile = list(np.array(self.datafiles)[self.ok_idatafiles]).index(
                datafile)
        data = load_data(datafile, dedisperse=False,
                dededisperse=False, tscrunch=self.tscrunch,
                #pscrunch=True, fscrunch=False, rm_baseline=rm_baseline,
                pscrunch=True, fscrunch=False, rm_baseline=True,
                flux_prof=False, refresh_arch=False, return_arch=False,
                quiet=quiet)
        phi = self.phis[ifile][isub]
        DM = self.DMs[ifile][isub]
        GM = self.GMs[ifile][isub]
        if self.bary:  # get fitted values
            DM /= self.doppler_fs[ifile][isub]
            GM /= self.doppler_fs[ifile][isub]**3
        scales = self.scales[ifile][isub]
        freqs = data.freqs[isub]
        nu_ref_DM, nu_ref_GM, nu_ref_tau = self.nu_refs[ifile][isub]
        P = data.Ps[isub]
        phases = data.phases
        if self.is_FITS_model:
            model_data = load_data(self.modelfile, dedisperse=False,
                    dededisperse=False, tscrunch=True, pscrunch=True,
                    fscrunch=False, rm_baseline=True, flux_prof=False,
                    #fscrunch=False, rm_baseline=False, flux_prof=False,
                    refresh_arch=False, return_arch=False, quiet=True)
            model = (model_data.masks * model_data.subints)[0,0]
            if model_data.nchan == 1:
                model = np.tile(model[0], len(freqs)).reshape(len(freqs),
                        model_data.nbin)
            model_name = self.modelfile
        else:
            try:
                model_name, ngauss, model = read_model(self.modelfile, phases,
                        freqs, data.Ps.mean(), quiet=quiet)
                        #freqs, data.Ps[isub], quiet=quiet)     #Track down
                if self.taus[ifile][isub] != 0.0:
                    model_name, model_code, model_nu_ref, ngauss, gparams, \
                            model_fit_flags, model_alpha, model_fit_alpha = \
                            read_model(self.modelfile, quiet=quiet)
                    gparams[1] = 0.0
                    model = gen_gaussian_portrait(model_code, gparams, 0.0,
                            phases, freqs, model_nu_ref)
            except:
                model_name, model = read_spline_model(self.modelfile,
                        freqs, data.nbin, quiet=True) #quiet=bool(quiet+(itoa-1)))
        if self.add_instrumental_response and \
                (self.ird['DM'] or len(self.ird['wids'])):
                    inst_resp_port_FT = instrumental_response_port_FT(
                            data.nbin, freqs, self.ird['DM'], P,
                            self.ird['wids'], self.ird['irf_types'])
                    model = fft.irfft(inst_resp_port_FT * fft.rfft(model),
                            axis=-1)
        if self.taus[ifile][isub] != 0.0:
            tau = self.taus[ifile][isub]
            if self.log10_tau: tau = 10**tau
            alpha = self.alphas[ifile][isub]
            model = fft.irfft(scattering_portrait_FT(
                scattering_times(tau, alpha, freqs, nu_ref_tau), data.nbin) * \
                        fft.rfft(model, axis=1), axis=1)
        port = rotate_portrait_full(data.subints[isub,0], phi, DM, GM, freqs,
                nu_ref_DM, nu_ref_GM, P)
        if rotate:
            model = rotate_data(model, rotate)
            port = rotate_data(port, rotate)
        port *= data.masks[isub,0]
        model_scaled = np.transpose(scales * np.transpose(model))
        titles = ("%s\nSubintegration %d"%(datafile, isub),
                "Fitted Model %s"%(model_name), "Residuals")
        if show:
            show_residual_plot(port=port, model=model_scaled, resids=None,
                    phases=phases, freqs=freqs,
                    noise_stds=data.noise_stds[isub,0], nfit=2, titles=titles,
                    rvrsd=bool(data.bw < 0), savefig=savefig)
        if return_fit:
            return (port, model_scaled, data.ok_ichans[isub], freqs,
                    data.noise_stds[isub,0])


if __name__ == "__main__":

    from optparse import OptionParser

    usage = "Usage: %prog -d <datafile or metafile> -m <modelfile> [options]"
    parser = OptionParser(usage)
    #parser.add_option("-h", "--help",
    #                  action="store_true", dest="help", default=False,
    #                  help="Show this help message and exit.")
    parser.add_option("-d", "--datafiles",
                      action="store", metavar="archive", dest="datafiles",
                      help="PSRCHIVE archive from which to measure TOAs/DMs, or a metafile listing archive filenames.  \
                              ***NB: Files should NOT be dedispersed!!*** \
                              i.e. vap -c dmc <datafile> should return 0!")
    parser.add_option("-m", "--modelfile",
                      action="store", metavar="model", dest="modelfile",
                      help="Model file from ppgauss.py, ppspline.py, or PSRCHIVE FITS file that either has same channel frequencies, nchan, & nbin as datafile(s), or is a single profile (nchan = 1, with the same nbin) to be interpreted as a constant template.")
    parser.add_option("-o", "--outfile",
                      action="store", metavar="timfile", dest="outfile",
                      default=None,
                      help="Name of output .tim file. Will append. [default=stdout]")
    parser.add_option("--errfile",
                      action="store", metavar="errfile", dest="errfile",
                      default=None,
                      help="If specified, will write the fitted DM errors to errfile (desirable if using 'Princeton'-like formatted TOAs). Will append.")
    parser.add_option("-T", "--tscrunch",
                      action="store_true", dest="tscrunch", default=False,
                      help="tscrunch archives before measurement (i.e., return only one set of measurements per archive.")
    parser.add_option("-f", "--format",
                      action="store", metavar="format", dest="format",
                      help="Format of output .tim file; either 'princeton' or 'ipta'.  Default is IPTA-like format.")
    parser.add_option("--nu_ref",
                      action="store", metavar="nu_ref", dest="nu_ref_DM",
                      default=None,
                      help="Topocentric frequency [MHz] to which the output TOAs are referenced, i.e. the frequency that has zero delay from a non-zero DM. 'inf' is used as an argument here for infinite frequency, but the default internal behavior follows TEMPO/2 convention and will write 0.0 for infinite-frequency TOAs. [defaults to zero-covariance frequency, recommended]")
    parser.add_option("--DM",
                      action="store", metavar="DM", dest="DM0", default=None,
                      help="Nominal DM [cm**-3 pc] from which to reference offset DM measurements.  If unspecified, will use the DM stored in each archive.")
    parser.add_option("--no_bary",
                      action="store_false", dest="bary", default=True,
                      help="Do not Doppler-correct DMs, GMs, taus, or nu_tau.  Output values are 'topocentric'.")
    parser.add_option("--one_DM",
                      action="store_true", dest="one_DM", default=False,
                      help="Returns single DM value in output .tim file for all subints in the epoch instead of a fitted DM per subint.")
    parser.add_option("--fix_DM",
                      action="store_false", dest="fit_DM", default=True,
                      help="Do not fit for DM. NB: the parfile DM will still be 'barycentered' in the TOA lines unless --no_bary is used!")
    parser.add_option("--fit_dt4",
                      action="store_true", dest="fit_GM", default=False,
                      help="Fit for delays that scale as nu**-4 and return 'GM' parameters s.t. dt4 = Dconst**2 * GM * nu**-4.  GM has units [cm**-6 pc**2 s**-1] and can be related to a discrete cloud causing refractive, geometric delays.")
    parser.add_option("--fit_scat",
                      action="store_true", dest="fit_scat", default=False,
                      help="Fit for scattering timescale and index per TOA.  Can be used with --fix_alpha.")
    parser.add_option("--no_logscat",
                      action="store_false", dest="log10_tau", default=True,
                      help="If using fit_scat, this flag specifies not to fit the log10 of the scattering timescale, but simply the scattering timescale.")
    parser.add_option("--scat_guess",
                      action="store", dest="scat_guess",
                      metavar="tau,freq,alpha",
                      default=None,
                      help="If using fit_scat, manually specify a comma-separated triplet containing an initial guess for the scattering timescale parameter [s], its reference frequency [MHz], and an initial guess for the scattering index.  Will be used for all archives; supercedes other initial values.")
    parser.add_option("--fix_alpha",
                      action="store_true", dest="fix_alpha", default=False,
                      help="Fix the scattering index value to the value specified as scattering_alpha in pplib.py or alpha in the provided .gmodel file.  Only used in combination with --fit_scat.")
    parser.add_option("--nu_tau",
                      action="store", metavar="nu_ref_tau", dest="nu_ref_tau",
                      default=None,
                      help="Frequency [MHz] to which the output scattering times are referenced, i.e. tau(nu) = tau * (nu/nu_ref_tau)**alpha.  If no_bary is True, this frequency is topocentric, otherwise barycentric. [default=nu_zero (zero-covariance frequency, recommended)]")
    parser.add_option("--print_phase",
                      action="store_true", dest="print_phase", default=False,
                      help="Print the fitted phase shift and its uncertainty on the TOA line with the flag -phs")
    parser.add_option("--print_flux",
                      action="store_true", dest="print_flux", default=False,
                      help="Print an estimate of the overall mean flux density and its uncertainty on the TOA line.")
    parser.add_option("--print_parangle",
                      action="store_true", dest="print_parangle",
                      default=False,
                      help="Print the parallactic angle of each subintegration on the TOA line.")
    parser.add_option("--flags",
                      action="store", metavar="flags", dest="toa_flags",
                      default="",
                      help="Pairs making up TOA flags to be written uniformly to all IPTA-formatted TOAs.  e.g. ('pta','NANOGrav','version',0.1)")
    parser.add_option("--snr_cut",
                      metavar="S/N", action="store", dest="snr_cutoff",
                      default=0.0,
                      help="Set a S/N cutoff for TOAs written.")
    parser.add_option("--showplot",
                      action="store_true", dest="show_plot", default=False,
                      help="Show a plot of fitted data/model/residuals for each subint.  Good for diagnostic purposes only.")
    parser.add_option("--quiet",
                      action="store_true", dest="quiet", default=False,
                      help="Only TOAs printed to standard output, if outfile is None.")

    (options, args) = parser.parse_args()

    if (options.datafiles is None or options.modelfile is None):
        print "\npptoas.py - simultaneously measure TOAs, DMs, and scattering in broadband data\n"
        parser.print_help()
        print ""
        parser.exit()

    datafiles = options.datafiles
    modelfile = options.modelfile
    outfile = options.outfile
    errfile = options.errfile
    tscrunch = options.tscrunch
    format = options.format
    nu_ref_DM = options.nu_ref_DM
    if nu_ref_DM:
        if nu_ref_DM == "inf":
            nu_ref_DM = np.inf
        else:
            nu_ref_DM = np.float64(nu_ref_DM)
        nu_refs = (nu_ref_DM, None)
    else: nu_refs = None
    DM0 = options.DM0
    if DM0: DM0 = np.float64(DM0)
    bary = options.bary
    one_DM = options.one_DM
    fit_DM = options.fit_DM
    fit_GM = options.fit_GM
    fit_scat = options.fit_scat
    log10_tau = options.log10_tau
    scat_guess = options.scat_guess
    if scat_guess:
        scat_guess = [s.upper() for s in scat_guess.split(',')]
        scat_guess = map(float, scat_guess)
    fix_alpha = options.fix_alpha
    nu_ref_tau = options.nu_ref_tau
    if nu_ref_tau:
        nu_ref_tau = np.float64(nu_ref_tau)
        if nu_ref_DM:
            nu_refs = (nu_ref_DM, nu_ref_tau)
        else:
            nu_refs = (None, nu_ref_tau)
    print_phase = options.print_phase
    print_flux = options.print_flux
    print_parangle = options.print_parangle
    k,v = options.toa_flags.split(',')[::2],options.toa_flags.split(',')[1::2]
    addtnl_toa_flags = dict(zip(k,v))
    snr_cutoff = float(options.snr_cutoff)
    show_plot = options.show_plot
    quiet = options.quiet

    gt = GetTOAs(datafiles=datafiles, modelfile=modelfile, quiet=quiet)
    gt.get_TOAs(datafile=None, tscrunch=tscrunch, nu_refs=nu_refs, DM0=DM0,
            bary=bary, fit_DM=fit_DM, fit_GM=fit_GM, fit_scat=fit_scat,
            log10_tau=log10_tau, scat_guess=scat_guess, fix_alpha=fix_alpha,
            print_phase=print_phase, print_flux=print_flux,
            print_parangle=print_parangle, addtnl_toa_flags=addtnl_toa_flags,
            method='trust-ncg', bounds=None, nu_fits=None, show_plot=show_plot,
            quiet=quiet)
    if format == "princeton":
        gt.write_princeton_TOAs(outfile=outfile, one_DM=one_DM,
            dmerrfile=errfile)
    else:
        if one_DM:
            gt.TOA_one_DM_list = [toa for toa in gt.TOA_list]
            for toa in gt.TOA_one_DM_list:
                ifile = list(np.array(gt.datafiles)[gt.ok_idatafiles]).index(
                        toa_archive)
                DDM = gt.DeltaDM_means[ifile]
                DDM_err = gt.DeltaDM_errs[ifile]
                toa.DM = DDM + gt.DM0s[ifile]
                toa.DM_error = DDM_err
                toa.flags['DM_mean'] = True
            write_TOAs(gt.TOA_one_DM_list, inf_is_zero=True,
                    SNR_cutoff=snr_cutoff, outfile=outfile, append=True)
        else:
            write_TOAs(gt.TOA_list, inf_is_zero=True, SNR_cutoff=snr_cutoff,
                    outfile=outfile, append=True)
