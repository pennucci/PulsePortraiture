#!/usr/bin/env python

########
#pptoas#
########

#pptoas is a command-line program used to simultaneously fit for phases (TOAs),
#    and dispersion measures (DMs).  Full-functionality is obtained when using
#    pptoas within an interactive python environment.

#Written by Timothy T. Pennucci (TTP; tim.pennucci@nanograv.org).
#Contributions by Scott M. Ransom (SMR) and Paul B. Demorest (PBD).

from pplib import *

#cfitsio defines a maximum number of files (NMAXFILES) that can be opened in
#the header file fitsio2.h.  Without calling unload() with PSRCHIVE, which
#touches the archive, I am not sure how to close the files.  So, to avoid the
#loop crashing, set a maximum number of archives for pptoas.  Modern machines
#should be able to handle almost 1000.
max_nfile = 999

#See DC_fact in pplib.py
if DC_fact:
    rm_baseline = True
else:
    rm_baseline = False

class TOA:

    """
    TOA class bundles common TOA attributes together with useful functions.
    """

    def __init__(self, archive, frequency, MJD, TOA_error, telescope, DM=None,
            DM_error=None, flags={}):
        """
        Form a TOA.

        archive is the string name of the TOA's archive.
        frequency is the reference frequency [MHz] of the TOA.
        MJD is a PSRCHIVE MJD object (the TOA, topocentric).
        TOA_error is the TOA uncertainty [us].
        telescope is the string designating the observatory.
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
        self.DM = DM
        self.DM_error = DM_error
        self.flags = flags
        for flag in flags.keys():
            exec('self.%s = flags["%s"]'%(flag, flag))

    def write_TOA(self, format="tempo2", outfile=None):
        """
        Print a formatted TOA to standard output or to file.

        format is one of 'tempo2', ... others coming ...
        outfile is the output file name; if None, will print to standard
            output.
        """
        write_TOAs(self, format=format, outfile=outfile, append=True)

    def convert_TOA(self, new_frequency, covariance):
        """
        To do...
        """
        print "Convert TOA to new reference frequency, with new error, \
            if covariance provided."

class GetTOAs:

    """
    GetTOAs is a class with methods to measure TOAs and DMs from data.
    """

    def __init__(self, datafiles, modelfile, quiet=False):
        """
        Unpack all of the data and set initial attributes.

        datafiles is either a single PSRCHIVE file name, or a name of a
            metafile containing a list of archive names.
        modelfile is a a write_model(...)-type of model file specifying the
            Gaussian model parameters.  modelfile can also be an arbitrary
            PSRCHIVE archive, although this feature is
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
        self.modelfile = modelfile
        self.obs = []
        self.nu0s = []
        self.nu_fits = []
        self.nu_refs = []
        self.ok_isubs = []
        self.epochs = []
        self.MJDs = []
        self.Ps = []
        self.phis = []
        self.phi_errs = []
        self.TOAs = []
        self.TOA_errs = []
        self.DM0s = []
        self.DMs = []
        self.DM_errs = []
        self.DeltaDM_means = []
        self.DeltaDM_errs = []
        self.doppler_fs = []
        self.scales = []
        self.scale_errs = []
        self.covariances = []
        self.red_chi2s = []
        self.nfevals = []
        self.rcs = []
        self.fit_durations = []
        self.quiet = quiet
        self.order = []
        self.TOA_list = []

    def get_TOAs(self, datafile=None, nu_ref=None, DM0=None, bary_DM=True,
            fit_DM=True, bounds=[(None, None), (None, None)], nu_fit=None,
            show_plot=False, addtnl_toa_flags={}, quiet=None):
        """
        Measure phases (TOAs) and dispersion measures (DMs).

        datafile defaults to self.datafiles, otherwise it is a single
            PSRCHIVE archive name
        nu_ref is the desired output reference frequency [MHz] of the TOAs;
            defaults to the zero-covariance frequency.
        DM0 is the baseline dispersion measure [cm**-3 pc]; defaults to what is
            stored in each datafile.
        bary_DM=True corrects the measured DMs based on the Doppler motion of
            the observatory with respect to the solar system barycenter.
        fit_DM=False will fit only for a phase; if this is the case, you might
            want to set bary_DM to False.
        bounds is a list of two 2-tuples, giving the lower and upper bounds on
            the phase and dispersion measure, respectively.
        nu_fit is the reference frequency [MHz] used in the fit; defaults to
            a guess at the zero-covariance frequency based on signal-to-noise
            ratios.
        show_plot=True will show a plot at the end of the fitting; it is only
            useful if the number of subintegrations in a datafile > 1.
        addtnl_toa_flags are pairs making up TOA flags to be written uniformly
            to all tempo2-formatted TOAs. e.g. ('pta','NANOGrav','version',0.1)
        quiet=True suppresses output.
        """
        if quiet is None: quiet = self.quiet
        already_warned = False
        warning_message = \
                "You are using an experimental functionality of pptoas!"
        self.nu_ref = nu_ref
        self.DM0 = DM0
        self.bary_DM = bary_DM
        self.ok_idatafiles = []
        start = time.time()
        tot_duration = 0.0
        nu_fit_default = nu_fit
        if datafile is None:
            datafiles = self.datafiles
        else:
            datafiles = [datafile]
        for iarch, datafile in enumerate(datafiles):
            fit_duration = 0.0
            #Load data
            try:
                data = load_data(datafile, dedisperse=False,
                        dededisperse=False, tscrunch=False, pscrunch=True,
                        fscrunch=False, rm_baseline=rm_baseline,
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
                    frontend=frontend, tempo_code=tempo_code)
            nu_fits = np.zeros(nsub, dtype=np.float64)
            nu_refs = np.zeros(nsub, dtype=np.float64)
            phis = np.zeros(nsub, dtype=np.double)
            phi_errs = np.zeros(nsub, dtype=np.double)
            TOAs = np.zeros(nsub, dtype="object")
            TOA_errs = np.zeros(nsub, dtype="object")
            DMs = np.zeros(nsub, dtype=np.float64)
            DM_errs = np.zeros(nsub, dtype=np.float64)
            doppler_fs = np.ones(nsub, dtype=np.float64)
            nfevals = np.zeros(nsub, dtype="int")
            rcs = np.zeros(nsub, dtype="int")
            scales = np.zeros([nsub, nchan], dtype=np.float64)
            scale_errs = np.zeros([nsub, nchan], dtype=np.float64)
            red_chi2s = np.zeros(nsub)
            covariances = np.zeros(nsub)
            #PSRCHIVE epochs are *midpoint* of the integration
            MJDs = np.array([epochs[isub].in_days()
                for isub in xrange(nsub)], dtype=np.double)
            DM_stored = DM # = arch.get_dispersion_measure()
            if self.DM0 is None:
                DM0 = DM_stored
            else:
                DM0 = self.DM0
            if not fit_DM:
                bounds[1] = (DM0, DM0)
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
                if model_data.nchan == 1:
                    model = np.tile(model[0], len(freqs[isub])).reshape(
                            len(freqs[isub]), nbin)
                    print model.shape
            if not quiet:
                print "\nEach of the %d TOAs is approximately %.2f s"%(
                        len(ok_isubs), integration_length / nsub)
                print "Doing Fourier-domain least-squares fit..."
            itoa = 1
            for isub in ok_isubs:
                id = datafile + "_%d"%isub
                epoch = epochs[isub]
                MJD = MJDs[isub]
                P = Ps[isub]
                if not self.is_FITS_model:
                    #Read model
                    try:
                        self.model_name, self.ngauss, model = read_model(
                                self.modelfile, phases, freqs[isub], Ps[isub],
                                quiet=bool(quiet+(itoa-1)))
                    except UnboundLocalError:
                        self.model_name, model = read_interp_model(
                                self.modelfile, freqs[isub], nbin,
                                quiet=True) #bool(quiet+(itoa-1)))
                #else:
                ##THESE FREQUENCIES WILL BE OFF IF AVERAGED CHANNELS##
                #    print model_data.freqs[0, ok_ichans[isub]] - \
                #            freqs[isub,ok_ichans[isub]]
                freqsx = freqs[isub,ok_ichans[isub]]
                portx = subints[isub,0,ok_ichans[isub]]
                modelx = model[ok_ichans[isub]]
                SNRsx = SNRs[isub,0,ok_ichans[isub]]
                #NB: Time-domain uncertainties below
                errs = noise_stds[isub,0,ok_ichans[isub]]
                #nu_fit is a guess at nu_zero, the zero-covariance frequency,
                #which is calculated after. This attempts to minimize the
                #number of function calls.  Lower frequencies mean more calls,
                #and the discrepancy in the phase estimates is at the sub-1ns
                #level, and sub-micro-DM level; the covariances are also
                #different, but all very similar as well.
                if nu_fit_default is None:
                    nu_fit = guess_fit_freq(freqsx, SNRsx)
                else:
                    nu_fit = nu_fit_default
                nu_fits[isub] = nu_fit

                ####################
                #DOPPLER CORRECTION#
                ####################
                #In principle, we should be able to correct the frequencies,
                #but since this is a messy business, it is easier to correct
                #the DM itself (below).
                #df = arch.get_Integration(int(isub)).get_doppler_factor()
                #freqsx = doppler_correct_freqs(freqsx, df)
                #nu0 = doppler_correct_freqs(nu0, df)
                ####################

                ###############
                #INITIAL GUESS#
                ###############
                #Having only one initial guess doesn't speed things up (at all)
                #Having multiple initial guesses is better for generality,
                #e.g. binary systems with poorly determined parameters.
                #One may envision a system that uses the previous phase
                #estimate as the next guess, but that could be bad, if one
                #subint is contaminated or very poorly determined.
                #Also have to be careful below, since the subints are 
                #dedispersed at different nu_fit.
                #Finally, Ns should be larger than nbin for very low S/N data,
                #especially in the case of noisy models...
                rot_port = rotate_data(portx, 0.0,
                        DM_stored, P, freqsx, nu_fit)
                #PSRCHIVE Dedisperses w.r.t. center of band, which is
                #different, in general, from nu_fit; this results in a
                #phase offset w.r.t to what would be seen in the PSRCHIVE
                #dedispersed portrait.
                phase_guess = fit_phase_shift(rot_port.mean(axis=0),
                        modelx.mean(axis=0), Ns=100).phase
                #Currently, fit_phase_shift returns an unbounded phase,
                #so here we transform to be on the interval [-0.5, 0.5]
                #This may not be needed, but hasn't proved dangerous yet...
                phase_guess = phase_guess % 1
                if phase_guess >= 0.5:
                    phase_guess -= 1.0
                DM_guess = DM_stored
                #Need a status bar?

                ####################
                #      THE FIT     #
                ####################
                if not quiet:
                    print "Fitting for TOA #%d"%(itoa)
                if len(freqsx) > 1:
                    results = fit_portrait(portx, modelx,
                            np.array([phase_guess, DM_guess]), P, freqsx,
                            nu_fit, self.nu_ref, errs, bounds=bounds, id=id,
                            quiet=quiet)
                else:  #1-channel hack
                    if not quiet:
                        print "TOA has one frequency channel!..."
                        print "...using Fourier phase gradient routine..."
                        if self.nu_ref is not None:
                            print "--nu_ref will be ignored!"
                    results = fit_phase_shift(portx[0], modelx[0], errs[0],
                            Ns=nbin)
                    results.DM = None #DM_stored
                    results.DM_err = None #0.0
                    results.nu_ref = freqsx[0]
                    results.nfeval = 0
                    results.return_code = -2
                    results.scales = np.array([results.scale])
                    results.scale_errs = np.array([results.scale_error])
                    results.covariance = 0.0
                results.phi = results.phase
                results.phi_err = results.phase_err
                fit_duration += results.duration

                ####################
                #  CALCULATE  TOA  #
                ####################
                results.TOA = epoch + pr.MJD(
                        ((results.phi * P) + backend_delay) /
                        (3600 * 24.))
                results.TOA_err = results.phi_err * P * 1e6 # [us]

                ##########################
                #DOPPLER CORRECTION OF DM#
                ##########################
                if self.bary_DM: #Default is True
                    #NB: the 'doppler factor' retrieved from PSRCHIVE seems to
                    #be the inverse of the convention nu_source/nu_observed
                    df = doppler_factors[isub]
                    if len(freqsx) > 1:
                        results.DM *= df  #NB: No longer the *fitted* value!
                    doppler_fs[isub] = df
                else:
                    doppler_fs[isub] = 1.0

                nu_refs[isub] = results.nu_ref
                phis[isub] = results.phi
                phi_errs[isub] = results.phi_err
                TOAs[isub] = results.TOA
                TOA_errs[isub] = results.TOA_err
                DMs[isub] = results.DM
                DM_errs[isub] = results.DM_err
                nfevals[isub] = results.nfeval
                rcs[isub] = results.return_code
                scales[isub, ok_ichans[isub]] = results.scales
                scale_errs[isub, ok_ichans[isub]] = results.scale_errs
                covariances[isub] = results.covariance
                red_chi2s[isub] = results.red_chi2
                #Compile useful TOA flags
                toa_flags = {}
                toa_flags['be'] = backend
                toa_flags['fe'] = frontend
                toa_flags['f'] = frontend + "_" + backend
                toa_flags['nbin'] = nbin
                toa_flags['nch'] = nchan
                toa_flags['nchx'] = len(freqsx)
                toa_flags['subint'] = isub
                toa_flags['tobs'] = subtimes[isub]
                toa_flags['pp_tmplt'] = self.modelfile
                if self.nu_ref is not None:
                    toa_flags['pp_cov'] = results.covariance
                toa_flags['pp_gof'] = results.red_chi2
                #toa_flags['pp_phs'] = results.phi
                toa_flags['pp_snr'] = results.snr
                for k,v in addtnl_toa_flags.iteritems():
                    toa_flags[k] = v
                self.TOA_list.append(TOA(datafile, results.nu_ref, results.TOA,
                        results.TOA_err, telescope.lower(), results.DM,
                        results.DM_err, toa_flags))
                itoa += 1

            DeltaDMs = DMs - DM0
            #The below returns the weighted mean and the sum of the weights,
            #but needs to do better in the case of small-error outliers from
            #RFI, etc.  Also, last TOA may mess things up...use median...?
            DeltaDM_mean, DeltaDM_var = np.average(DeltaDMs[ok_isubs],
                    weights=DM_errs[ok_isubs]**-2, returned=True)
            DeltaDM_var = DeltaDM_var**-1
            if len(ok_isubs) > 1:
                #The below multiply by the red. chi-squared to inflate the
                #errors.
                DeltaDM_var *= np.sum(
                        ((DeltaDMs[ok_isubs] - DeltaDM_mean)**2) /
                        (DM_errs[ok_isubs]**2)) / (len(DeltaDMs[ok_isubs]) - 1)
            DeltaDM_err = DeltaDM_var**0.5
            self.order.append(datafile)
            self.obs.append(obs)
            self.nu0s.append(nu0)
            self.nu_fits.append(nu_fits)
            self.nu_refs.append(nu_refs)
            self.ok_isubs.append(ok_isubs)
            self.epochs.append(epochs)
            self.MJDs.append(MJDs)
            self.Ps.append(Ps)
            #NB: phis are w.r.t. nu_ref!!!
            self.phis.append(phis)
            self.phi_errs.append(phi_errs)
            #NB: TOAs are w.r.t. nu_ref!!!
            self.TOAs.append(TOAs)
            self.TOA_errs.append(TOA_errs)
            #NB: DMs are Doppler corrected, if bary_DM is set!!!
            self.DM0s.append(DM0)
            self.DMs.append(DMs)
            self.DM_errs.append(DM_errs)
            self.DeltaDM_means.append(DeltaDM_mean)
            self.DeltaDM_errs.append(DeltaDM_err)
            self.doppler_fs.append(doppler_fs)
            self.scales.append(scales)
            self.scale_errs.append(scale_errs)
            self.covariances.append(covariances)
            self.red_chi2s.append(red_chi2s)
            self.nfevals.append(nfevals)
            self.rcs.append(rcs)
            self.fit_durations.append(fit_duration)
            if not quiet:
                print "--------------------------"
                print datafile
                print "~%.4f sec/TOA"%(fit_duration / len(ok_isubs))
                print "Avg. TOA error is %.3f us"%(phi_errs[ok_isubs].mean() *
                        Ps.mean() * 1e6)
            if show_plot:
                stop = time.time()
                tot_duration += stop - start
                self.show_results(datafile)
                #self.show_fit(datafile)
                start = time.time()
        if not show_plot:
            tot_duration = time.time() - start
        if not quiet and len(self.ok_isubs):
            print "--------------------------"
            print "Total time: %.2f sec, ~%.4f sec/TOA"%(tot_duration,
                    tot_duration / (np.array(map(len, self.ok_isubs)).sum()))

    def get_channel_red_chi2s(self, threshold=1.5, show=False):
        """
        Calculate reduced chi-squared values for each profile fit.

        Adds attributes self.channel_red_chi2s and self.zap_channels, the
            latter based on a thresholding value.

        threshold is a reduced chi-squared value which is used to flag channels
            for zapping (cf. ppzap.py).  Values above threshold are added to
            self.zap_channels.
        show=True will show the before/after portraits for each subint with
            proposed channels to zap.
        """
        self.channel_red_chi2s = []
        self.zap_channels = []
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
                for ichan in ok_ichans:
                    channel_red_chi2 = get_red_chi2(port[ichan],
                            model[ichan], errs=noise_stds[ichan],
                            dof=len(port[ichan])-0) #Not sure about exact dof
                    red_chi2s.append(channel_red_chi2)
                    if channel_red_chi2 > threshold: bad_ichans.append(ichan)
                    elif np.isnan(channel_red_chi2): bad_ichans.append(ichan)
                channel_red_chi2s.append(red_chi2s)
                zap_channels.append(bad_ichans)
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

    def write_princeton_TOAs(self, datafile=None, outfile=None, nu_ref=None,
            one_DM=False, dmerrfile=None):
        """
        Write TOAs to file.

        Currently only writes Princeton-formatted TOAs.

        datafile defaults to self.datafiles, otherwise it is a list of
            PSRCHIVE archive names that have been fitted for TOAs.
        outfile is the name of the output file.
        nu_ref is the desired output reference frequency [MHz] of the TOAs;
            defaults to nu_ref from get_TOAs(...).
        one_DM writes the weighted average delta-DM in the TOA file, instead of
            the per-TOA delta-DM.
        dmerrfile is a string specifying the name of a "DM" file to be written
            containing the TOA, the (full) DM, and the DM uncertainty.  This
            output needs improvement!
        """
        if datafile is None:
            datafiles = self.datafiles
        else:
            datafiles = [datafile]
        if outfile is not None:
            sys.stdout = open(outfile,"a")
        if dmerrfile is not None:
            dmerrs = open(dmerrfile,"a")
        for datafile in datafiles:
            ifile = list(np.array(self.datafiles)[self.ok_idatafiles]).index(
                    datafile)
            ok_isubs = self.ok_isubs[ifile]
            DM0 = self.DM0s[ifile]
            nsub = len(self.nu_refs[ifile])
            if nu_ref is None:
                #Default to self.nu_refs
                if self.nu_ref is None:
                    nu_refs = self.nu_refs[ifile]

                else:
                    nu_refs = self.nu_ref * np.ones(nsub)
                TOAs = self.TOAs[ifile]
                TOA_errs = self.TOA_errs[ifile]
            else:
                nu_refs = nu_ref * np.ones(nsub)
                epochs = self.epochs[ifile]
                Ps = self.Ps[ifile]
                phis = self.phis[ifile]
                TOAs = np.zeros(nsub, dtype="object")
                TOA_errs = self.TOA_errs[ifile]
                DMs = self.DMs[ifile]
                DMs_fitted = DMs / self.doppler_fs[ifile]
                for isub in ok_isubs:
                    TOAs[isub] = calculate_TOA(epochs[isub], Ps[isub],
                            phis[isub], DMs_fitted[isub],
                            self.nu_refs[ifile][isub], nu_refs[isub])
            obs_code = obs_codes[self.obs[ifile].telescope.lower()]
            #Currently writes topocentric frequencies
            for isub in ok_isubs:
                TOA_MJDi = TOAs[isub].intday()
                TOA_MJDf = TOAs[isub].fracday()
                TOA_err = TOA_errs[isub]
                if one_DM:
                    DeltaDM_mean = self.DeltaDM_means[ifile]
                    DM_err = self.DeltaDM_errs[ifile]
                    write_princeton_TOA(TOA_MJDi, TOA_MJDf, TOA_err,
                            nu_refs[isub], DeltaDM_mean, obs=obs_code)
                else:
                    DeltaDMs = self.DMs[ifile] - self.DM0s[ifile]
                    DM_err = self.DM_errs[ifile][isub]
                    write_princeton_TOA(TOA_MJDi, TOA_MJDf, TOA_err,
                            nu_refs[isub], DeltaDMs[isub], obs=obs_code)
                if dmerrfile is not None:
                    TOA_MJDi = TOAs[isub].intday()
                    TOA_MJDf = TOAs[isub].fracday()
                    TOA = "%5d"%int(TOA_MJDi) + ("%.13f"%TOA_MJDf)[1:]
                    dmerrs.write("%.3f\t%s\t%.8f\t%.6f\n"%(nu_refs[isub], TOA,
                        self.DMs[ifile][isub], self.DM_errs[ifile][isub]))
        if dmerrfile is not None:
            dmerrs.close()
        sys.stdout = sys.__stdout__

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
                dededisperse=False, tscrunch=False,
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
            return_fit=False, quiet=None):
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
                dededisperse=False, tscrunch=False,
                #pscrunch=True, fscrunch=False, rm_baseline=rm_baseline,
                pscrunch=True, fscrunch=False, rm_baseline=True,
                flux_prof=False, refresh_arch=False, return_arch=False,
                quiet=quiet)
        phi = self.phis[ifile][isub]
        #Pre-corrected DM, if corrected
        DM_fitted = self.DMs[ifile][isub] / self.doppler_fs[ifile][isub]
        scales = self.scales[ifile][isub]
        freqs = data.freqs[isub]
        nu_ref = self.nu_refs[ifile][isub]
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
            except:
                model_name, model = read_interp_model(self.modelfile,
                        freqs, data.nbin, quiet=True) #quiet=bool(quiet+(itoa-1)))
        port = rotate_data(data.subints[isub,0], phi, DM_fitted, P, freqs,
                nu_ref)
        if rotate:
            model = rotate_data(model, rotate)
            port = rotate_data(port, rotate)
        port *= data.masks[isub,0]
        model_scaled = np.transpose(scales * np.transpose(model))
        titles = ("%s\nSubintegration %d"%(datafile, isub),
                "Fitted Model %s"%(model_name), "Residuals")
        if show:
            show_residual_plot(port=port, model=model_scaled, resids=None,
                    phases=phases, freqs=freqs, titles=titles,
                    rvrsd=bool(data.bw < 0))
        if return_fit:
            return (port, model_scaled, data.ok_ichans[isub], freqs,
                    data.noise_stds[isub,0])

    def show_results(self, datafile=None):
        """
        Show a plot of the fitted phases and dispersion measures.

        Only useful if the number of subintegrations > 1.

        datafile is a single PSRCHIVE archive name; defaults to the first one
            listed in self.datafiles.

        To be improved.
        """
        if datafile is None:
            datafile = self.datafiles[0]
        ifile = list(np.array(self.datafiles)[self.ok_idatafiles]).index(
                datafile)
        ok_isubs = self.ok_isubs[ifile]
        nu_fits = self.nu_fits[ifile][ok_isubs]
        nu_refs = self.nu_refs[ifile][ok_isubs]
        MJDs = self.MJDs[ifile][ok_isubs]
        Ps = self.Ps[ifile][ok_isubs]
        phis = self.phis[ifile][ok_isubs]
        phi_errs = self.phi_errs[ifile][ok_isubs]
        #These are the 'barycentric' DMs, if they were corrected (default yes)
        DMs = self.DMs[ifile][ok_isubs]
        DM_errs = self.DM_errs[ifile][ok_isubs]
        DMs_fitted = DMs / self.doppler_fs[ifile][ok_isubs]
        DM0 = self.DM0s[ifile]
        DeltaDM_mean = self.DeltaDM_means[ifile]
        DeltaDM_err = self.DeltaDM_errs[ifile]
        rcs = self.rcs[ifile]
        cols = ['b','k','g','b','r']
        fig = plt.figure()
        pf = np.polynomial.polynomial.polyfit
        #This is to obtain the TOA phase offsets w.r.t. nu_ref
        #Apparently, changing phis in place changes self.phis ???
        phi_primes = phase_transform(phis, DMs_fitted, nu_refs,
                    self.nu0s[ifile], Ps, mod=False)
        #phi_primes may have N rotations incorporated...
        milli_sec_shifts = (phi_primes) * Ps * 1e3
        #Not sure weighting works...
        fit_results = pf(MJDs, milli_sec_shifts, 1, full=True, w=phi_errs**-2)
        resids = (milli_sec_shifts) - (fit_results[0][0] + (fit_results[0][1] *
            MJDs))
        resids_mean,resids_var = np.average(resids, weights=phi_errs**-2,
                returned=True)
        resids_var = resids_var**-1
        if len(ok_isubs) > 1:
            resids_var *= np.sum(((resids - resids_mean)**2) /
                    (phi_errs**2)) / (len(resids) - 1)
        resids_err = resids_var**0.5
        RMS = resids_err
        ax1 = fig.add_subplot(311)
        for ii in range(len(ok_isubs)):
            ax1.errorbar(MJDs[ii], milli_sec_shifts[ii] * 1e3,
                    phi_errs[ii] * Ps[ii] * 1e6, color='%s'
                    %cols[rcs[ii]], fmt='+')
        plt.plot(MJDs, (fit_results[0][0] + (fit_results[0][1] * MJDs)) * 1e3,
                "m--")
        plt.xlabel("MJD")
        plt.ylabel(r"Offset [$\mu$s]")
        ax1.text(0.1, 0.9, "%.2e ms/s"%(fit_results[0][1] / (3600 * 24)),
                ha='center', va='center', transform=ax1.transAxes)
        ax2 = fig.add_subplot(312)
        for ii in range(len(ok_isubs)):
            ax2.errorbar(MJDs[ii], resids[ii] * 1e3, phi_errs[ii] *
                    Ps[ii] * 1e6, color='%s'%cols[rcs[ii]], fmt='+')
        plt.plot(MJDs, np.ones(len(MJDs)) * resids_mean * 1e3, "m--")
        xverts = np.array([MJDs[0], MJDs[0], MJDs[-1], MJDs[-1]])
        yverts = np.array([resids_mean - resids_err, resids_mean + resids_err,
            resids_mean + resids_err, resids_mean - resids_err]) * 1e3
        plt.fill(xverts, yverts, "m", alpha=0.25, ec='none')
        plt.xlabel("MJD")
        plt.ylabel(r"Offset [$\mu$s]")
        ax2.text(0.1, 0.9, r"$\sim$weighted RMS = %d ns"%int(resids_err * 1e6),
                ha='center', va='center', transform=ax2.transAxes)
        ax3 = fig.add_subplot(313)
        for ii in range(len(ok_isubs)):
            ax3.errorbar(MJDs[ii], DMs[ii], DM_errs[ii],
                    color='%s'%cols[rcs[ii]], fmt='+')
        if abs(DeltaDM_mean) / DeltaDM_err < 10:
            plt.plot(MJDs, np.ones(len(MJDs)) * DM0, "r-", label="DM0")
            #plt.text(MJDs[-1], DM0, "DM0", ha="left", va="center", color="r")
            plt.legend(loc=1)
        plt.plot(MJDs, np.ones(len(MJDs)) * (DeltaDM_mean + DM0), "m--")
        xverts = [MJDs[0], MJDs[0], MJDs[-1], MJDs[-1]]
        yverts = [DeltaDM_mean + DM0 - DeltaDM_err,
                  DeltaDM_mean + DM0 + DeltaDM_err,
                  DeltaDM_mean + DM0 + DeltaDM_err,
                  DeltaDM_mean + DM0 - DeltaDM_err]
        plt.fill(xverts, yverts, "m", alpha=0.25, ec='none')
        plt.xlabel("MJD")
        plt.ylabel(r"DM [cm$^{3}$ pc]")
        ax3.text(0.15, 0.9, r"$\Delta$ DM = %.2e $\pm$ %.2e"%(DeltaDM_mean,
            DeltaDM_err), ha='center',
                va='center', transform=ax3.transAxes)
        plt.title(datafile)
        plt.show()


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
                      help="Model file from ppgauss.py, ppinterp.py, or PSRCHIVE FITS file that either has same channel frequencies, nchan, & nbin as datafile(s), or is a single profile (nchan = 1, with the same nbin) to be interpreted as a constant template.")
    parser.add_option("-o", "--outfile",
                      action="store", metavar="timfile", dest="outfile",
                      default=None,
                      help="Name of output .tim file. Will append. [default=stdout]")
    parser.add_option("-f", "--format",
                      action="store", metavar="format", dest="format",
                      help="Format of output .tim file; either 'princeton' or 'tempo2'.  Default is tempo2 format.")
    parser.add_option("--flags",
                      action="store", metavar="flags", dest="toa_flags",
                      default="",
                      help="Pairs making up TOA flags to be written uniformly to all tempo2-formatted TOAs.  e.g. ('pta','NANOGrav','version',0.1)")
    parser.add_option("--nu_ref",
                      action="store", metavar="nu_ref", dest="nu_ref",
                      default=None,
                      help="Frequency [MHz] to which the output TOAs are referenced, i.e. the frequency that has zero delay from a non-zero DM. 'inf' is used for inifite frequency.  [default=nu_zero (zero-covariance frequency, recommended)]")
    parser.add_option("--DM",
                      action="store", metavar="DM", dest="DM0", default=None,
                      help="Nominal DM [cm**-3 pc] from which to reference offset DM measurements.  If unspecified, will use the DM stored in each archive.")
    parser.add_option("--no_bary_DM",
                      action="store_false", dest="bary_DM", default=True,
                      help='Do not Doppler-correct the DM to make a "barycentric DM".')
    parser.add_option("--one_DM",
                      action="store_true", dest="one_DM", default=False,
                      help="Returns single DM value in output .tim file for all subints in the epoch instead of a fitted DM per subint.")
    parser.add_option("--snr_cut",
                      metavar="SNR", action="store", dest="snr_cutoff",
                      default=0.0,
                      help="Set a SNR cutoff for TOAs written.")
    parser.add_option("--errfile",
                      action="store", metavar="errfile", dest="errfile",
                      default=None,
                      help="If specified, will write the fitted DM errors to errfile (desirable if using non-tempo2 formatted TOAs). Will append.")
    parser.add_option("--fix_DM",
                      action="store_false", dest="fit_DM", default=True,
                      help="Do not fit for DM. NB: the parfile DM will still be 'barycentered' in the TOA lines unless --no_bary_DM is used!")
    parser.add_option("--showplot",
                      action="store_true", dest="showplot", default=False,
                      help="Plot fit results for each epoch. Only useful if nsubint > 1.")
    parser.add_option("--quiet",
                      action="store_true", dest="quiet", default=False,
                      help="Only TOAs printed to standard output, if outfile is None.")

    (options, args) = parser.parse_args()

    if (options.datafiles is None or options.modelfile is None):
        print "\npptoas.py - simultaneously measure TOAs and DMs in broadband data\n"
        parser.print_help()
        print ""
        parser.exit()

    datafiles = options.datafiles
    modelfile = options.modelfile
    nu_ref = options.nu_ref
    if nu_ref:
        if nu_ref == "inf":
            nu_ref = np.inf
        else:
            nu_ref = np.float64(nu_ref)
    DM0 = options.DM0
    if DM0: DM0 = np.float64(DM0)
    bary_DM = options.bary_DM
    one_DM = options.one_DM
    fit_DM = options.fit_DM
    outfile = options.outfile
    format = options.format
    k,v = options.toa_flags.split(',')[::2],options.toa_flags.split(',')[1::2]
    addtnl_toa_flags = dict(zip(k,v))
    snr_cutoff = float(options.snr_cutoff)
    errfile = options.errfile
    showplot = options.showplot
    quiet = options.quiet

    gt = GetTOAs(datafiles=datafiles, modelfile=modelfile, quiet=quiet)
    gt.get_TOAs(nu_ref=nu_ref, DM0=DM0, bary_DM=bary_DM, fit_DM=fit_DM,
            show_plot=showplot, addtnl_toa_flags=addtnl_toa_flags, quiet=quiet)
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
            write_TOAs(gt.TOA_one_DM_list, format="tempo2",
                    SNR_cutoff=snr_cutoff, outfile=outfile, append=True)
        else:
            write_TOAs(gt.TOA_list, format="tempo2", SNR_cutoff=snr_cutoff,
                    outfile=outfile, append=True)
