#!/usr/bin/env python

from pplib import *

class GetTOAs:
    """
    """
    def __init__(self, datafiles, modelfile, common=False, quiet=False):
        """
        """
        if file_is_ASCII(datafiles):
            self.metafile = datafiles
            self.datafiles = open(datafiles, "r").readlines()
            self.datafiles = [self.datafiles[ifile][:-1] for ifile in
                    xrange(len(self.datafiles))]
        else:
            self.datafiles = [datafiles]
        self.is_gauss_model = file_is_ASCII(modelfile)
        self.modelfile = modelfile
        self.common = common
        self.obs = []
        self.nu0s = []
        self.nu_fits = []
        self.nu_refs = []
        self.nsubxs = []
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
        self.scalesx = []
        self.scale_errs = []
        self.covariances = []
        self.red_chi2s = []
        self.nfevals = []
        self.rcs = []
        self.fit_durations = []
        self.quiet = quiet
        self.order = []
        if len(self.datafiles) == 1 or self.common is True:
            data = load_data(self.datafiles[0], dedisperse=False,
                    dededisperse=False, tscrunch=True, pscrunch=True,
                    fscrunch=False, rm_baseline=True, flux_prof=False,
                    norm_weights=True, quiet=True)
            if self.is_gauss_model:
                self.model_name, self.ngauss, self.model = read_model(
                        self.modelfile, data.phases, data.freqs, data.Ps[0],
                        self.quiet)
            else:
                self.model_data = load_data(self.modelfile, dedisperse=True,
                        dededisperse=False, tscrunch=True, pscrunch=True,
                        fscrunch=False, rm_baseline=True, flux_prof=False,
                        norm_weights=True, quiet=True)
                self.model_name = self.model_data.source
                self.ngauss = 0
                self.model_weights = self.model_data.weights[0]
                self.model = self.model_data.subints[0,0]
                #self.model = np.transpose(self.model_weights * np.transpose(
                #    self.model_data.subints[0,0]))
            self.source = data.source
            self.nchan = data.nchan
            self.nbin = data.nbin
            self.nu0 = data.nu0
            self.bw = data.bw
            self.freqs = data.freqs
            self.lofreq = self.freqs[0]-(self.bw/(2*self.nchan))
            if self.source is None: self.source = "noname"
            del(data)

    def get_TOAs(self, datafile=None, nu_ref=None, DM0=None, bary_DM=True,
            fit_DM=True, bounds=[(None, None), (None, None)], nu_fit=None,
            show_plot=False, quiet=False):
        """
        """
        self.nu_ref = nu_ref
        self.DM0 = DM0
        self.bary_DM = bary_DM
        start = time.time()
        tot_duration = 0.0
        if datafile is None:
            datafiles = self.datafiles
        else:
            datafiles = [datafile]
        for datafile in datafiles:
            fit_duration = 0.0
            #Load data
            data = load_data(datafile, dedisperse=False,
                    dededisperse=False, tscrunch=False, pscrunch=True,
                    fscrunch=False, rm_baseline=True, flux_prof=False,
                    norm_weights=True, quiet=quiet)
            #Unpack the data dictionary into the local namespace; see load_data
            #for dictionary keys.
            for key in data.keys():
                exec(key + " = data['" + key + "']")
            if source is None: source = "noname"
            #Read model
            if len(datafiles) !=1 and self.common is False:
                self.model_name, self.ngauss, model = read_model(
                        self.modelfile, phases, freqs, Ps.mean(), quiet=quiet)
            else:
                model = self.model
            nu_fits = np.empty(nsubx, dtype=np.float)
            nu_refs = np.empty(nsubx, dtype=np.float)
            phis = np.empty(nsubx, dtype=np.double)
            phi_errs = np.empty(nsubx, dtype=np.double)
            TOAs = np.empty(nsubx, dtype="object")
            TOA_errs = np.empty(nsubx, dtype="object")
            DMs = np.empty(nsubx, dtype=np.float)
            DM_errs = np.empty(nsubx, dtype=np.float)
            doppler_fs = np.empty(nsubx, dtype=np.float)
            nfevals = np.empty(nsubx, dtype="int")
            rcs = np.empty(nsubx, dtype="int")
            scales = np.empty([nsubx, nchan], dtype=np.float)
            #These next two are lists because in principle,
            #the subints could have different numbers of zapped channels.
            scalesx = []
            scale_errs = []
            red_chi2s = np.empty(nsubx)
            covariances = np.empty(nsubx)
            #PSRCHIVE epochs are *midpoint* of the integration
            MJDs = np.array([epochs[isub].in_days()
                for isub in xrange(nsub)], dtype=np.double)
            DM_stored = arch.get_dispersion_measure()
            if self.DM0 is None:
                DM0 = DM_stored
            else:
                DM0 = self.DM0
            if not fit_DM:
                bounds[1] = (DM0, DM0)
            if not quiet:
                print "\nEach of the %d TOAs are approximately %.2f s"%(nsubx,
                        arch.integration_length() / nsub)
                print "Doing Fourier-domain least-squares fit..."
            #These are the subintegration indices that haven't been zapped
            ok_isubs = map(int, np.compress(map(len,
                np.array(subintsxs)[:,0]), np.arange(nsub)))
            for isubx in xrange(nsubx):
                isub = ok_isubs[isubx]
                id = datafile + "_%d_%d"%(isub, isubx)
                epoch = epochs[isub]
                MJD = MJDs[isub]
                P = Ps[isub]
                if self.is_gauss_model:
                    freqsx = freqsxs[isub]
                    portx = subintsxs[isub][0]
                    modelx = np.compress(weights[isub], model, axis=0)
                else:
                    tot_weights = weights[isub] + self.model_weights
                    freqsx = np.compress(tot_weights, freqs)
                    portx = np.compress(tot_weights, subints[isub][0], axis=0)
                    modelx = np.compress(tot_weights, model, axis=0)
                #A proxy for SNR (change this):
                channel_SNRs = portx.std(axis=1) / get_noise(portx, chans=True)
                #Adopted from Lorimer & Kramer '05
                #Weq = portx.sum(axis=1) / portx.max(axis=1)
                #channel_SNRs = portx.sum(axis=1) / (Weq**0.5 *
                #        get_noise(portx, chans=True))
                #nu_fit is a guess at nu_zero, the zero-covariance frequency,
                #which is calculated after. This attempts to minimize the
                #number of function calls.  Lower frequencies mean more calls,
                #and the discrepancy in the phase estimates is at the sub-1ns
                #level, and sub-micro-DM level; the covariances are also
                #different, but all very similar as well.
                if nu_fit is None:
                    nu_fit = guess_fit_freq(freqsx, channel_SNRs)
                nu_fits[isubx] = nu_fit

                ####################
                #DOPPLER CORRECTION#
                ####################
                #In principle, we should be able to correct the frequencies,
                #but since this is a messy business, it is easier to correct
                #the DM itself (below).
                #df = arch.get_Integration(isub).get_doppler_factor()
                #freqsx = doppler_correct_freqs(freqsx, df)
                #nu0 = doppler_correct_freqs(nu0, df)
                ####################

                ###############
                #INITIAL GUESS#
                ###############
                #Having only one initial guess doesn't speed things up (at all)
                #Having multiple initial guesses is better for generality,
                #eg. binary systems with poorly determined parameters.
                #One may envision a system that uses the previous phase
                #estimate as the next guess, but that could be bad, if one
                #subint is contaminated or very poorly determined.
                #Also have to be careful below, since the subints are 
                #dedispersed at different nu_fit
                rot_port = rotate_portrait(portx, 0.0,
                        DM_stored, P, freqsx, nu_fit)
                #PSRCHIVE Dedisperses w.r.t. center of band, which is
                #different, in general, from nu_fit; this results in a
                #phase offset w.r.t to what would be seen in the PSRCHIVE
                #dedispersed portrait.
                phase_guess = fit_phase_shift(rot_port.mean(axis=0),
                        model.mean(axis=0)).phase
                #Currently, fit_phase_shift returns an unbounded phase,
                #so here we transform to be on the interval [-0.5, 0.5]
                #This may not be needed, but hasn't proved dangerous yet...
                phase_guess = phase_guess % 1
                if phase_guess > 0.5:
                    phase_guess -= 1.0
                DM_guess = DM_stored
                #Need a status bar?

                ####################
                #      THE FIT     #
                ####################
                if not quiet:
                    print "Fitting for TOA %d"%(isubx)
                (phi, DM, scalex, param_errs, nu_ref, covariance, red_chi2,
                        duration, nfeval, rc) = fit_portrait(portx, modelx,
                                np.array([phase_guess, DM_guess]), P, freqsx,
                                nu_fit, self.nu_ref, bounds=bounds, id = id,
                                quiet=quiet)
                phi_err, DM_err = param_errs[0], param_errs[1]
                fit_duration += duration

                ####################
                #  CALCULATE  TOA  #
                ####################
                TOA = epochs[isubx] + pr.MJD((phi * P) / (3600 * 24.))
                TOA_err = phi_err * P * 1e6 # [us]

                ##########################
                #DOPPLER CORRECTION OF DM#
                ##########################
                if self.bary_DM: #Default is True
                    #NB: the 'doppler factor' retrieved from PSRCHIVE seems to
                    #be the inverse of the convention nu_source/nu_observed
                    df = arch.get_Integration(isub).get_doppler_factor()
                    DM *= df    #NB: No longer the *fitted* value!
                    doppler_fs[isubx] = df
                else:
                    doppler_fs[isubx] = 1.0

                nu_refs[isubx] = nu_ref
                phis[isubx] = phi
                phi_errs[isubx] = phi_err
                TOAs[isubx] = TOA
                TOA_errs[isubx] = TOA_err
                DMs[isubx] = DM
                DM_errs[isubx] = DM_err
                nfevals[isubx] = nfeval
                rcs[isubx] = rc
                scalesx.append(scalex)
                scale_errs.append(param_errs[2:])
                scale = np.zeros(nchan)
                iscalex = 0
                for ichan in xrange(nchan):
                    if weights[isub, ichan] == 1:
                        scale[ichan] = scalex[iscalex]
                        iscalex += 1
                    else: pass
                scales[isubx] = scale
                covariances[isubx] = covariance
                red_chi2s[isubx] = red_chi2
            DeltaDMs = DMs - DM0
            #The below returns the weighted mean and the sum of the weights,
            #but needs to do better in the case of small-error outliers from
            #RFI, etc.  Also, last TOA may mess things up...use median...?
            DeltaDM_mean, DeltaDM_var = np.average(DeltaDMs,
                    weights=DM_errs**-2, returned=True)
            DeltaDM_var = DeltaDM_var**-1
            if nsubx > 1:
                #The below multiply by the red. chi-squared to inflate the
                #errors.
                DeltaDM_var *= np.sum(((DeltaDMs - DeltaDM_mean)**2) /
                        (DM_errs**2)) / (len(DeltaDMs) - 1)
            DeltaDM_err = DeltaDM_var**0.5
            self.order.append(datafile)
            self.obs.append(arch.get_telescope())
            self.nu0s.append(nu0)
            self.nu_fits.append(nu_fits)
            self.nu_refs.append(nu_refs)
            self.nsubxs.append(nsubx)
            self.epochs.append(np.take(epochs, ok_isubs))
            self.MJDs.append(np.take(MJDs, ok_isubs))
            self.Ps.append(np.take(Ps, ok_isubs))
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
            self.scalesx.append(scalesx)
            self.scale_errs.append(scale_errs)
            self.covariances.append(covariances)
            self.red_chi2s.append(red_chi2s)
            self.nfevals.append(nfevals)
            self.rcs.append(rcs)
            self.fit_durations.append(fit_duration)
            if not quiet:
                print "--------------------------"
                print datafile
                print "~%.2f min/TOA"%(fit_duration / (60. * nsubx))
                print "Avg. TOA error is %.3f us"%(phi_errs.mean() *
                        Ps.mean() * 1e6)
            if show_plot:
                stop = time.time()
                tot_duration += stop - start
                self.show_results(datafile)
                start = time.time()
        if not show_plot:
            tot_duration = time.time() - start
        if not quiet:
            print "--------------------------"
            print "Total time: %.2f, ~%.2f min/TOA"%(tot_duration / 60,
                    tot_duration / (60 * np.sum(np.array(self.nsubxs))))

    def write_TOAs(self, datafile=None, outfile=None, nu_ref=None,
            one_DM=False, dmerrfile=None):
        """
        """
        #FIX - determine observatory
        #FIX - options for different TOA formats
        if datafile is None:
            datafiles = self.datafiles
        else:
            datafiles = [datafile]
        if outfile is not None:
            sys.stdout = open(outfile,"a")
        if dmerrfile is not None:
            dmerrs = open(dmerrfile,"a")
        for datafile in datafiles:
            ifile = datafiles.index(datafile)
            nsubx = self.nsubxs[ifile]
            DM0 = self.DM0s[ifile]
            if nu_ref is None:
                #Default to self.nu_refs
                if self.nu_ref is None:
                    nu_refs = self.nu_refs[ifile]
                else:
                    nu_refs = self.nu_ref * np.ones(nsubx)
                TOAs = self.TOAs[ifile]
                TOA_errs = self.TOA_errs[ifile]
            else:
                nu_refs = nu_ref * np.ones(nsubx)
                epochs = self.epochs[ifile]
                Ps = self.Ps[ifile]
                phis = self.phis[ifile]
                TOAs = np.empty(nsubx, dtype="object")
                TOA_errs = self.TOA_errs[ifile]
                DMs = self.DMs[ifile]
                DMs_fitted = DMs / self.doppler_fs[ifile]
                for isubx in range(nsubx):
                    TOAs[isubx] = calculate_TOA(epochs[isubx], Ps[isubx],
                            phis[isubx], DMs_fitted[isubx],
                            self.nu_refs[ifile][isubx], nu_refs[isubx])
            try:
                obs_code = obs_codes["%s"%self.obs[ifile].lower()]
            except KeyError:
                obs_code = obs_codes["%s"%self.obs[ifile].upper()]
            #Currently writes topocentric frequencies
            for isubx in xrange(nsubx):
                TOA_MJDi = TOAs[isubx].intday()
                TOA_MJDf = TOAs[isubx].fracday()
                TOA_err = TOA_errs[isubx]
                if one_DM:
                    DeltaDM_mean = self.DeltaDM_means[ifile]
                    DM_err = self.DeltaDM_errs[ifile]
                    write_princeton_TOA(TOA_MJDi, TOA_MJDf, TOA_err,
                            nu_refs[isubx], DeltaDM_mean, obs=obs_code)
                else:
                    DeltaDMs = self.DMs[ifile] - self.DM0s[ifile]
                    DM_err = self.DM_errs[ifile][isubx]
                    write_princeton_TOA(TOA_MJDi, TOA_MJDf, TOA_err,
                            nu_refs[isubx], DeltaDMs[isubx], obs=obs_code)
                if dmerrfile is not None:
                    TOA_MJDi = TOAs[isubx].intday()
                    TOA_MJDf = TOAs[isubx].fracday()
                    TOA = "%5d"%int(TOA_MJDi) + ("%.13f"%TOA_MJDf)[1:]
                    dmerrs.write("%s\t%.8f\t%.6f\n"%(TOA, DM0, DM_err))
        if dmerrfile is not None:
            dmerrs.close()
        sys.stdout = sys.__stdout__

    def write_pam_cmds(self, datafile=None, outfile=None):
        """
        DO NOT USE
        """
        if datafile is None:
            datafiles = self.datafiles
        else:
            datafiles = [datafile]
        if outfile is not None:
            of = open(outfile, "a")
        else:
            of = sys.__stdout__
        for datafile in datafiles:
            ifile = datafiles.index(datafile)
            phis = self.phis[ifile]
            phi_errs = self.phi_errs[ifile]
            nu0 = self.nu0s[ifile]
            nu_fits = self.nu_fits[ifile]
            DeltaDM_mean = self.DeltaDM_means[ifile]
            DM0 = self.DM0s[ifile]
            DM = DeltaDM_mean + DM0
            pam_ext = datafile[-datafile[::-1].find("."):] + ".rot"
            #The below returns the weighted mean and the sum of the weights,
            #but needs to do better in the case of small-error outliers from
            #RFI, etc.  Also, last TOA may mess things up...use median...?
            phi_mean, phi_var = np.average(phis, weights=phi_errs**-2,
                    returned=True)
            phi_var = phi_var**-1
            phi_mean = phase_transform(phi_mean, DM, nu_fits.mean(), nu0,
                    np.array(self.Ps).mean())
            of.write("pam -e %s -r %.7f -d %.5f %s\n"%(pam_ext, phi_mean, DM,
                datafile))
        if outfile is not None: of.close()

    def show_subint(self, datafile=None, isubx=0, quiet=False):
        """
        subintx 0 = python index 0
        isubx is currently mapped incorrectly.
        """
        if datafile is None:
            datafile = self.datafiles[0]
        ifile = self.datafiles.index(datafile)
        data = load_data(datafile, dedisperse=True,
                dededisperse=False, tscrunch=False,
                pscrunch=True, fscrunch=False, rm_baseline=True,
                flux_prof=False, norm_weights=True, quiet=quiet)
        title = "%s ; subintx %d"%(datafile, isubx)
        port = np.transpose(data.weights[isubx] * np.transpose(
            data.subints[isubx,0]))
        show_portrait(port=port, phases=data.phases, freqs=data.freqs,
                title=title, prof=True, fluxprof=True, rvrsd=bool(data.bw < 0))

    def show_fit(self, datafile=None, isubx=0, quiet=False):
        """
        subintx 0 = python index 0
        isubx is currently mapped incorrectly.
        """
        if datafile is None:
            datafile = self.datafiles[0]
        ifile = self.datafiles.index(datafile)
        data = load_data(datafile, dedisperse=False,
                dededisperse=False, tscrunch=False,
                pscrunch=True, fscrunch=False, rm_baseline=True,
                flux_prof=False, norm_weights=True, quiet=quiet)
        phi = self.phis[ifile][isubx]
        #Pre-corrected DM, if corrected
        DM_fitted = self.DMs[ifile][isubx] / self.doppler_fs[ifile][isubx]
        scales = self.scales[ifile][isubx]
        freqs = data.freqs
        nu_fit = self.nu_fits[ifile][isubx]
        nu_ref = self.nu_refs[ifile][isubx]
        P = data.Ps[isubx]
        phases = data.phases
        weights = data.weights[isubx]
        if not self.is_gauss_model:
            weights += self.model_weights
            model_name = self.model_name
            model = np.transpose(weights * np.transpose(self.model))
        else:
            model_name, ngauss, model = read_model(self.modelfile, phases,
                    freqs, data.Ps.mean(), quiet=quiet)
                    #freqs, data.Ps[isubx], quiet=quiet)
        port = rotate_portrait(data.subints[isubx,0], phi, DM_fitted, P, freqs,
                nu_ref)
        port = np.transpose(weights * np.transpose(port))
        model_scaled = np.transpose(scales * np.transpose(model))
        titles = ("%s\nSubintegrationx %d"%(datafile, isubx),
                "Fitted Model %s"%(model_name), "Residuals")
        show_residual_plot(port=port, model=model_scaled, resids=None,
                phases=phases, freqs=freqs, titles=titles,
                rvrsd=bool(data.bw < 0))

    def show_results(self, datafile=None):
        """
        Need descriptors in the plots...rx, etc.
        """
        if datafile:
            ifile = self.datafiles.index(datafile)
        else:
            ifile = 0
        nsubx = self.nsubxs[ifile]
        nu_fits = self.nu_fits[ifile]
        nu_refs = self.nu_refs[ifile]
        MJDs = self.MJDs[ifile]
        Ps = self.Ps[ifile]
        phis = self.phis[ifile]
        phi_errs = self.phi_errs[ifile]
        #These are the 'barycentric' DMs, if they were corrected (default yes)
        DMs = self.DMs[ifile]
        DM_errs = self.DM_errs[ifile]
        DMs_fitted = DMs / self.doppler_fs[ifile]
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
                    self.nu0s[ifile], Ps)
        #phi_primes may have N rotations incorporated...
        milli_sec_shifts = (phi_primes) * Ps * 1e3
        #Not sure weighting works...
        fit_results = pf(MJDs, milli_sec_shifts, 1, full=True, w=phi_errs**-2)
        resids = (milli_sec_shifts) - (fit_results[0][0] + (fit_results[0][1] *
            MJDs))
        resids_mean,resids_var = np.average(resids, weights=phi_errs**-2,
                returned=True)
        resids_var = resids_var**-1
        if nsubx > 1:
            resids_var *= np.sum(((resids - resids_mean)**2) /
                    (phi_errs**2)) / (len(resids) - 1)
        resids_err = resids_var**0.5
        RMS = resids_err
        ax1 = fig.add_subplot(311)
        for isubx in xrange(nsubx):
            ax1.errorbar(MJDs[isubx], milli_sec_shifts[isubx] * 1e3,
                    phi_errs[isubx] * Ps[isubx] * 1e6, color='%s'
                    %cols[rcs[isubx]], fmt='+')
        plt.plot(MJDs, (fit_results[0][0] + (fit_results[0][1] * MJDs)) * 1e3,
                "m--")
        plt.xlabel("MJD")
        plt.ylabel(r"Offset [$\mu$s]")
        ax1.text(0.1, 0.9, "%.2e ms/s"%(fit_results[0][1] / (3600 * 24)),
                ha='center', va='center', transform=ax1.transAxes)
        ax2 = fig.add_subplot(312)
        for isubx in xrange(nsubx):
            ax2.errorbar(MJDs[isubx], resids[isubx] * 1e3, phi_errs[isubx] *
                    Ps[isubx] * 1e6, color='%s'%cols[rcs[isubx]], fmt='+')
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
        for isubx in xrange(nsubx):
            ax3.errorbar(MJDs[isubx], DMs[isubx], DM_errs[isubx],
                    color='%s'%cols[rcs[isubx]], fmt='+')
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

    def show_hists(self, datafile=None):
        if datafile is None:
            nfevals = []
            rcs = []
            nsubx = np.array(self.nsubxs).sum()
            for ifile in xrange(len(self.datafiles)):
                nfevals += list(self.nfevals[ifile])
                rcs += list(self.rcs[ifile])
            nfevals = np.array(nfevals)
            rcs = np.array(rcs)
        else:
            ifile = self.datafiles.index(datafile)
            nfevals = self.nfevals[ifile]
            rcs = self.rcs[ifile]
            nsubx = self.nsubxs[ifile]
        cols = ['b','k','g','b','r']
        bins = nfevals.max()
        binmin = nfevals.min()
        rc1=np.zeros(bins - binmin + 1)
        rc2=np.zeros(bins - binmin + 1)
        rc4=np.zeros(bins - binmin + 1)
        rc5=np.zeros(bins - binmin + 1)
        for isubx in xrange(nsubx):
            nfeval = nfevals[isubx]
            if rcs[isubx] == 1:
                rc1[nfeval - binmin] += 1
            elif rcs[isubx] == 2:
                rc2[nfeval - binmin] += 1
            elif rcs[isubx] == 4:
                rc4[nfeval - binmin] += 1
            else:
                print "rc %d discovered at rcs index %d"%(rcs[isubx], isubx)
                rc5[nfevals - binmin] += 1
        width = 1
        b1 = plt.bar(np.arange(binmin - 1, bins) + 0.5, rc1, width,
                color=cols[1])
        b2 = plt.bar(np.arange(binmin - 1, bins) + 0.5, rc2, width,
                color=cols[2], bottom=rc1)
        b4 = plt.bar(np.arange(binmin - 1, bins) + 0.5, rc4, width,
                color=cols[4], bottom=(rc1 + rc2))
        if rc5.sum() != 0:
            b5 = plt.bar(np.arange(binmin - 1, bins) + 0.5, rc5, width,
                    color=cols[3], bottom=(rc1 + rc2 + rc4))
            plt.legend((b1[0], b2[0], b4[0], b5[0]), ('rc=1', 'rc=2', 'rc=4',
                'rc=#'))
        else: plt.legend((b1[0], b2[0], b4[0]), ('rc=1', 'rc=2', 'rc=4'))
        plt.xlabel("nfevals")
        plt.ylabel("counts")
        plt.xticks(np.arange(1, bins + 1))
        if rc5.sum() != 0:
            plt.axis([binmin - width/2.0, bins + width/2.0, 0, max(rc1 + rc2 +
                rc4 + rc5)])
        else:
            plt.axis([binmin - width/2.0, bins + width/2.0, 0, max(rc1 + rc2 +
                rc4)])
        plt.figure(2)
        urcs = np.unique(rcs)
        rchist = np.histogram(rcs, bins=len(urcs))[0]
        for iurc in xrange(len(urcs)):
            rc = urcs[iurc]
            plt.hist(np.ones(rchist[iurc]) * rc, bins=1, color=cols[rc])
            plt.xlabel("rc")
            plt.ylabel("counts")
        plt.show()


if __name__ == "__main__":

    from optparse import OptionParser

    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    #parser.add_option("-h", "--help",
    #                  action="store_true", dest="help", default=False,
    #                  help="Show this help message and exit.")
    parser.add_option("-d", "--datafile",
                      action="store", metavar="archive", dest="datafile",
                      help="PSRCHIVE archive from which to measure TOAs/DMs.")
    parser.add_option("-M", "--metafile",
                      action="store",metavar="metafile", dest="metafile",
                      help="File containing list of archive filenames from which to measure TOAs/DMs.")
    parser.add_option("-m", "--modelfile",
                      action="store", metavar="model", dest="modelfile",
                      help="Model file from ppgauss.py, or PSRCHIVE FITS file that has same channel frequencies, nchan, nbin as datafile(s) (i.e. cannot be used with --uncommon).")
    parser.add_option("-o", "--outfile",
                      action="store", metavar="timfile", dest="outfile",
                      default=None,
                      help="Name of output .tim file name. Will append. [default=stdout]")
    parser.add_option("--nu_ref",
                      action="store", metavar="nu_ref", dest="nu_ref",
                      default=None,
                      help="Frequency [MHz] to which the output TOAs are referenced, i.e. the frequency that has zero delay from a non-zero DM. 'inf' is used for inifite frequency.  [default=nu_zero (zero-covariance frequency, recommended)]")
    parser.add_option("--DM",
                      action="store", metavar="DM", dest="DM0", default=None,
                      help="Nominal DM [cm**-3 pc] (float) from which to measure offset.  If unspecified, will use the DM stored in the archive.")
    parser.add_option("--no_bary_DM",
                      action="store_false", dest="bary_DM", default=True,
                      help='Do not Doppler-correct the fitted DM to make "barycentric DM".')
    parser.add_option("--one_DM",
                      action="store_true", dest="one_DM", default=False,
                      help="Returns single DM value in output .tim file for all subints in the epoch instead of a fitted DM per subint.")
    parser.add_option("--errfile",
                      action="store", metavar="errfile", dest="errfile",
                      default=None,
                      help="If specified, will write the fitted DM errors to errfile. Will append.")
    parser.add_option("--fix_DM",
                      action="store_false", dest="fit_DM", default=True,
                      help="Do not fit for DM.  NB: you'll want to also use --no_bary_DM.")
    parser.add_option("--pam_cmd",
                      action="store_true", dest="pam_cmd", default=False,
                      help='Append pam commands to file "pam_cmd."')
    parser.add_option("--common",
                      action="store_true", dest="common", default=False,
                      help="If supplying a metafile, use this flag if the data are homogenous (i.e. have the same nu0, bw, nchan, nbin)")
    parser.add_option("--showplot",
                      action="store_true", dest="showplot", default=False,
                      help="Plot fit results for each epoch. Only useful if nsubint > 1.")
    parser.add_option("--quiet",
                      action="store_true", dest="quiet", default=False,
                      help="Minimal to stdout.")

    (options, args) = parser.parse_args()

    if (options.datafile is None and options.metafile is None or
            options.modelfile is None):
            print "\npptoas.py - simultaneous least-squares fit for TOAs and DMs\n"
            parser.print_help()
            print ""
            parser.exit()

    datafile = options.datafile
    metafile = options.metafile
    modelfile = options.modelfile
    nu_ref = options.nu_ref
    if nu_ref:
        if nu_ref == "inf":
            nu_ref = np.inf
        else:
            nu_ref = float(nu_ref)
    DM0 = options.DM0
    if DM0: DM0 = float(DM0)
    bary_DM = options.bary_DM
    one_DM = options.one_DM
    fit_DM = options.fit_DM
    pam_cmd = options.pam_cmd
    outfile = options.outfile
    errfile = options.errfile
    common = options.common
    showplot = options.showplot
    quiet = options.quiet

    if metafile is None:
        datafiles = datafile
    else:
        datafiles = metafile
    gt = GetTOAs(datafiles=datafiles, modelfile=modelfile, common=common,
            quiet=quiet)
    gt.get_TOAs(nu_ref=nu_ref, DM0=DM0, bary_DM=bary_DM, fit_DM=fit_DM,
            show_plot=showplot, quiet=quiet)
    gt.write_TOAs(outfile=outfile, one_DM=one_DM, dmerrfile=errfile)
    if pam_cmd: gt.write_pam_cmds()
