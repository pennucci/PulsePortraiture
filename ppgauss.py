#!/usr/bin/env python

#########
#ppgauss#
#########

#ppgauss is a command-line program used to make frequency-dependent,
#    gaussian-component models of pulse portraits.  Full-functionality is
#    obtained when using ppgauss within an interactive python environment.

#Written by Timothy T. Pennucci (TTP; pennucci@virginia.edu).
#Contributions by Scott M. Ransom (SMR) and Paul B. Demorest (PBD)

from matplotlib.patches import Rectangle
from pplib import *


class DataPortrait:

    """
    DataPortrait is a class that contains the data to which a model is fit.

    This class is also useful for the quick examining of PSRCHIVE archives in
    an interactive python environment.
    """

    def __init__(self, datafile=None, quiet=False):
        """
        Unpack all of the data and set initial attributes.

        If datafile is a metafile of PSRCHIVE archives, "join" attributes are
        set, which are used to align the archives.  A large (>3) number of
        archives signficiantly slows the fitting process, and it has only been
        testing for the case that each archive originates from a unique
        receiver.
        quiet=True suppresses output.
        """
        self.init_params = []
        if file_is_ASCII(datafile):
            self.join_params = []
            self.join_fit_flags = []
            self.join_nchans = [0]
            self.join_nchanxs = [0]
            self.join_ichans = []
            self.join_ichanxs = []
            self.nchans = []
            self.metafile = self.datafile = datafile
            self.datafiles = open(datafile, "r").readlines()
            self.datafiles = [self.datafiles[ifile][:-1] for ifile in
                    xrange(len(self.datafiles))]
            self.njoin = len(self.datafiles)
            self.Ps = 0.0
            self.nchan = 0
            self.nchanx = 0
            #self.nu0s = []
            self.lofreq = np.inf
            self.hifreq = 0.0
            self.freqs = []
            self.freqsxs = []
            self.port = []
            self.portx = []
            self.flux_prof = []
            self.flux_profx = []
            self.noise_stds = []
            self.noise_stdsxs = []
            self.SNRs = []
            self.SNRsxs = []
            self.weights = []
            for ifile in range(len(self.datafiles)):
                datafile = self.datafiles[ifile]
                data = load_data(datafile, dedisperse=True, tscrunch=True,
                        pscrunch=True, fscrunch=False, rm_baseline=True,
                        flux_prof=True, norm_weights=True, return_arch=True,
                        quiet=quiet)
                self.nchan += data.nchan
                self.nchanx += len(data.ok_ichans[0])
                if ifile == 0:
                    self.join_nchans.append(self.nchan)
                    self.join_nchanxs.append(self.nchanx)
                    self.join_params.append(0.0)
                    self.join_fit_flags.append(0)
                    self.join_params.append(data.DM*0.0)
                    self.join_fit_flags.append(1)
                    self.nbin = data.nbin
                    self.phases = data.phases
                    refprof = data.prof
                    self.source = data.source
                else:
                    self.join_nchans.append(self.nchan)
                    self.join_nchanxs.append(self.nchanx)
                    prof = data.prof
                    phi = -fit_phase_shift(prof, refprof).phase
                    self.join_params.append(phi)
                    self.join_fit_flags.append(1)
                    self.join_params.append(data.DM*0.0)
                    self.join_fit_flags.append(1)
                self.Ps += data.Ps.mean()
                lf = data.freqs.min() - (abs(data.bw) / (2*data.nchan))
                if lf < self.lofreq:
                    self.lofreq = lf
                hf = data.freqs.max() + (abs(data.bw) / (2*data.nchan))
                if hf > self.hifreq:
                    self.hifreq = hf
                self.freqs.extend(data.freqs)
                self.freqsxs.extend(data.freqsxs[0])
                self.weights.extend(data.weights[0])
                self.port.extend(data.subints[0,0])
                self.portx.extend(data.subintsxs[0][0])
                self.flux_prof.extend(data.flux_prof)
                self.flux_profx.extend(data.flux_profx)
                self.noise_stds.extend(data.noise_stds[0,0])
                self.noise_stdsxs.extend(
                        data.noise_stds[0,0][data.ok_ichans[0]])
                self.SNRs.extend(data.SNRs[0,0])
                self.SNRsxs.extend(data.SNRs[0,0][data.ok_ichans[0]])
            self.Ps /= len(self.datafiles)
            self.Ps = [self.Ps] #This line is a toy
            self.bw = self.hifreq - self.lofreq
            self.freqs = np.array(self.freqs)
            self.freqsxs = np.array(self.freqsxs)
            self.nu0 = self.freqs.mean()
            self.isort = np.argsort(self.freqs)
            self.isortx = np.argsort(self.freqsxs)
            for ijoin in range(self.njoin):
                join_ichans = np.intersect1d(np.where(self.isort >=
                    self.join_nchans[ijoin])[0], np.where(self.isort <
                        self.join_nchans[ijoin+1])[0])
                self.join_ichans.append(join_ichans)
                join_ichanxs = np.intersect1d(np.where(self.isortx >=
                    self.join_nchanxs[ijoin])[0], np.where(self.isortx <
                        self.join_nchanxs[ijoin+1])[0])
                self.join_ichanxs.append(join_ichanxs)
            self.weights = np.array(self.weights)[self.isort]
            self.weights = [self.weights]
            self.port = np.array(self.port)[self.isort]
            self.portx = np.array(self.portx)[self.isortx]
            self.flux_prof = np.array(self.flux_prof)[self.isort]
            self.flux_profx = np.array(self.flux_profx)[self.isortx]
            self.noise_stds = np.array(self.noise_stds)[self.isort]
            self.noise_stdsxs = np.array(self.noise_stdsxs)[self.isortx]
            self.SNRs = np.array(self.SNRs)[self.isort]
            self.SNRsxs = np.array(self.SNRsxs)[self.isortx]
            self.freqs.sort()
            self.freqsxs.sort()
            self.freqsxs = [self.freqsxs]
            self.join_params = np.array(self.join_params)
            self.join_fit_flags = np.array(self.join_fit_flags)
            self.all_join_params = [self.join_ichanxs, self.join_params,
                    self.join_fit_flags]
            self.show_data_portrait()
        else:
            self.njoin = 0
            self.join_params = []
            self.join_ichans = []
            self.all_join_params = []
            self.datafile = datafile
            self.data = load_data(datafile, dedisperse=True,
                    dededisperse=False, tscrunch=True, pscrunch=True,
                    fscrunch=False, rm_baseline=True, flux_prof=True,
                    norm_weights=True, return_arch=True, quiet=quiet)
            #Unpack the data dictionary into the local namespace;
            #see load_data for dictionary keys.
            for key in self.data.keys():
                exec("self." + key + " = self.data['" + key + "']")
            if self.source is None: self.source = "noname"
            self.port = (self.masks * self.subints)[0,0]
            self.portx = self.subintsxs[0][0]
            self.noise_stdsxs = self.noise_stds[0,0,self.ok_ichans[0]]
            self.SNRsxs = self.SNRs[0,0,self.ok_ichans[0]]

    def fit_profile(self, profile, tau=0.0, fixscat=True, auto_gauss=0.0,
            show=True):
        """
        Fit gaussian components to a profile.

        profile is the array containing the profile of length nbin.
        tau != 0.0 is the scattering timescale [bin] added to the fitted
            gaussians; it is also the initial parameter if fixscat=False.
        fixscat=False fits for a scattering timescale.
        auto_gauss != 0.0 specifies the initial guess at a width of a single
            gaussian component to be fit automatically.
        show=False is used if you want auto_gauss to work without checking it.
        """
        fig = plt.figure()
        profplot = fig.add_subplot(211)
        #Noise below may be off
        self.interactor = GaussianSelector(profplot, profile,
                get_noise(profile), tau=tau, fixscat=fixscat,
                auto_gauss=auto_gauss, minspanx=None, minspany=None,
                useblit=True)
        if show: plt.show()
        self.init_params = self.interactor.fitted_params
        self.ngauss = (len(self.init_params) - 2) / 3

    def fit_flux_profile(self, guessA=1.0, guessalpha=0.0, plot=True,
            quiet=False):
        """
        Fit a power-law to the phase-averaged flux spectrum of the data.

        Fitted parameters and uncertainties are added to as class attributes.

        guessA is the initial amplitude parameter.
        guessalpha is the initial spectral index parameter.
        plot=True shows the fit results.
        quiet=True suppresses output.
        """
        #Noise level below may be off
        fp = fit_powlaw(self.flux_profx, np.array([guessA,guessalpha]),
            np.median(self.noise_stdsxs), self.freqsxs[0], self.nu0)
        if not quiet:
            print ""
            print "Flux-density power-law fit"
            print "----------------------------------"
            print "residual mean = %.2f"%fp.residuals.mean()
            print "residual std. = %.2f"%fp.residuals.std()
            print "reduced chi-squared = %.2f"%(fp.chi2 / fp.dof)
            print "A = %.3f +/- %.3f (flux at %.2f MHz)"%(fp.amp,
                    fp.amp_err, self.nu0)
            print "alpha = %.3f +/- %.3f"%(fp.alpha, fp.alpha_err)
        if plot:
            ax1 = plt.subplot(211, position=(0.1,0.1,0.8,0.4))
            ax2 = plt.subplot(212, position=(0.1,0.5,0.8,0.4))
            ax1.plot(self.freqsxs[0], fp.residuals, 'r+')
            ax2.plot(self.freqs, powlaw(self.freqs, self.nu0,
                fp.amp, fp.alpha), 'k-')
            ax2.plot(self.freqsxs[0], self.flux_profx, 'r+')
            ax1.set_xlim(self.freqs.min(), self.freqs.max())
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticklabels([])
            ax1.set_yticks(ax1.get_yticks()[1:-1])
            ax2.set_yticks(ax2.get_yticks()[1:-1])
            ax2.text(0.05, 0.1, r"A$_{\nu_0}$ = %.2f $\pm$ %.2f"%(
                fp.amp, fp.amp_err) + "\n" + r"$\alpha$ = %.2f $\pm$ %.2f"%(
                    fp.alpha, fp.alpha_err), ha="left", va="bottom",
                transform=ax2.transAxes)
            ax1.text(0.05, 0.1, "Residuals",ha="left", va="bottom",
                    transform=ax1.transAxes)
            ax1.set_xlabel("Frequency [MHz]")
            plt.text(0.05, 0.5, "Flux Units")
            ax2.set_title("Average Flux Profile for %s"%self.source)
            plt.show()
        self.spect_A = fp.amp
        self.spect_A_err = fp.amp_err
        self.spect_index = fp.alpha
        self.spect_index_err = fp.alpha_err


    def make_gaussian_model(self, modelfile=None,
            ref_prof=(None, None), tau=0.0, fixloc=False, fixwid=False,
            fixamp=False, fixscat=True, niter=0, fiducial_gaussian=False,
            auto_gauss=0.0, writemodel=False, outfile=None, writeerrfile=False,
            errfile=None, model_name=None, residplot=None, quiet=False):
        """
        Fit a gaussian-component model with independently evolving components.

        This is the main function within ppgauss.

        modelfile is a write_model(...)-type of model file; if provided, the
            fit will use its parameters and flags as a starting point for a
            new fit.
        ref_prof is a tuple specifying the (reference frequency, bandwidth)
            [MHz] of the profile used for an initial fit of gaussian
            components.  The reference frequency will be the model reference
            frequency.
        tau is a scattering timescale [bin]
        fixloc=True does not allow the components' positions to evolve
        fixwid=True does not allow the components' width to evolve
        fixamp=True does not allow the components' height to evolve
        fixscat=True does not fit for a scattering timescale.
        niter is the number of iterations after the initial model fit.
        fiducial_gaussian=True sets fixloc=False for all components except the
            first component fit, which is fixed.
        auto_gauss != 0.0 specifies the initial guess at a width of a single
            gaussian component to be fit automatically.
        writemodel=True writes the fitted model to file.
        outfile is a string designating the name of the output model file name.
        writeerrfile=True writes a model file containing errors on the fitted
            parameters.
        errfile is a string designating the name of the parameter error file.
        model_name is a string designating the name of the model.
        residplot is a string given if a saved output plot of the model, data,
            and residuals is desired.
        quiet=True suppresses output.
        """
        if errfile is None and outfile is not None:
            errfile = outfile + "_err"
        if modelfile:
            (self.model_name, self.nu_ref, self.ngauss, self.init_model_params,
                    self.fit_flags) = read_model(modelfile)
            self.init_model_params[1] *= self.nbin / self.Ps[0]
        else:
            if model_name is None:
                self.model_name = self.source
            else:
                self.model_name = model_name
            #Fit the profile
            if not len(self.init_params):
                self.nu_ref = ref_prof[0]
                self.bw_ref = ref_prof[1]
                if self.nu_ref is None: self.nu_ref = self.nu0
                if self.bw_ref is None: self.bw_ref = abs(self.bw)
                okinds = np.compress(np.less(self.nu_ref - (self.bw_ref/2),
                    self.freqs) * np.greater(self.nu_ref + (self.bw_ref/2),
                    self.freqs) * self.weights[0], np.arange(self.nchan))
                #The below profile average gives a slightly different set of
                #values for the profile than self.profile, if given the full
                #band and center frequency.  Unsure why; shouldn't matter.
                profile = np.take(self.port, okinds, axis=0).mean(axis=0)
                self.fit_profile(profile, tau=tau, fixscat=fixscat,
                        auto_gauss=auto_gauss)
            #All slopes, spectral indices start at 0.0
            locparams = widparams = ampparams = np.zeros(self.ngauss)
            self.init_model_params = np.empty([self.ngauss, 6])
            for igauss in xrange(self.ngauss):
                self.init_model_params[igauss] = np.array(
                        [self.init_params[2::3][igauss], locparams[igauss],
                            self.init_params[3::3][igauss], widparams[igauss],
                            self.init_params[4::3][igauss], ampparams[igauss]])
            self.init_model_params = np.array([self.init_params[0]] +
                [self.init_params[1]] + list(np.ravel(self.init_model_params)))
            self.fit_flags = np.ones(len(self.init_model_params))
            self.fit_flags[1] *= not(fixscat)
            self.fit_flags[3::6] *= not(fixloc)
            self.fit_flags[5::6] *= not(fixwid)
            self.fit_flags[7::6] *= not(fixamp)
            if fiducial_gaussian:
                #ifgauss = self.init_params[4::3].argmax()
                ifgauss = 0
                self.fit_flags[3::6] = 1
                self.fit_flags[3::6][ifgauss] = 0
        #The noise...
        self.portx_noise = np.outer(self.noise_stdsxs, np.ones(self.nbin))
        #self.portx_noise = np.outer(get_noise(self.portx, chans=True),
        #        np.ones(self.nbin))
        #channel_SNRs = np.array([get_SNR(self.portx[ichan]) for ichan in
        #    range(self.nchanx)])
        #self.nu_fit = guess_fit_freq(self.freqsxs[0], channel_SNRs)
        self.nu_fit = guess_fit_freq(self.freqsxs[0], self.SNRsxs)
        #Here's the loop
        if niter < 0: niter = 0
        self.niter = niter
        self.itern = niter
        self.model_params = np.copy(self.init_model_params)
        self.total_time = 0.0
        self.start = time.time()
        #if not quiet:
        #    print "Fitting gaussian model portrait..."
        print "Fitting gaussian model portrait..."
        iterator = self.model_iteration(quiet)
        iterator.next()
        self.cnvrgnc = self.check_convergence(efac=1.0, quiet=quiet)
        if writemodel:
            self.write_model(outfile=outfile, quiet=quiet)
        if writeerrfile:
            self.write_errfile(errfile=errfile, quiet=quiet)
        while (self.niter and not self.cnvrgnc):
            if self.cnvrgnc:
                break
            else:
                if not self.njoin:
                    if not quiet:
                        print "\nRotating data portrait for iteration %d."%(
                                self.itern - self.niter + 1)
                    self.port = rotate_portrait(self.port, self.phi, self.DM,
                            self.Ps[0], self.freqs, self.nu_fit)
                    self.portx = rotate_portrait(self.portx, self. phi,
                            self.DM, self.Ps[0], self.freqsxs[0], self.nu_fit)
                else:
                    if not quiet:
                        print "...iteration %d..."%(self.itern - self.niter +
                                1)
            if not quiet:
                print "Fitting gaussian model portrait..."
            iterator.next()
            self.niter -= 1
            #For safety, write model after each iteration
            if writemodel:
                self.write_model(outfile=outfile, quiet=quiet)
            self.cnvrgnc = self.check_convergence(efac=1.0, quiet=quiet)
            if writeerrfile:
                self.write_errfile(errfile=errfile, quiet=quiet)
        if self.njoin:
            for ii in range(self.njoin):
                jic = self.join_ichans[ii]
                self.port[jic] = rotate_data(self.port[jic],
                        -self.join_params[0::2][ii],
                        -self.join_params[1::2][ii], self.Ps[0],
                        self.freqs[jic], self.nu_ref)
                jicx = self.join_ichanxs[ii]
                self.portx[jicx] = rotate_data(self.port[jicx],
                        -self.join_params[0::2][ii],
                        -self.join_params[1::2][ii], self.Ps[0],
                        self.freqsxs[0][jicx], self.nu_ref)
                self.model[jic] = rotate_data(self.model[jic],
                        -self.join_params[0::2][ii],
                        -self.join_params[1::2][ii], self.Ps[0],
                        self.freqs[jic], self.nu_ref)
            self.model_masked = np.transpose(self.weights[0] *
                    np.transpose(self.model))
            self.modelx = np.compress(self.weights[0], self.model, axis=0)
        if not quiet:
            print "Residuals mean: %.2e"%(self.portx - self.modelx).mean()
            print "Residuals std:  %.2e"%(self.portx - self.modelx).std()
            print "Data std:       %.2e\n"%np.median(self.noise_stdsxs)
            print "Total fit time: %.2f min"%(self.total_time / 60.0)
            print "Total time:     %.2f min\n"%((time.time() - self.start) /
                    60.0)
        if residplot:
            resids = self.port - self.model_masked
            titles = ("%s"%self.datafile, "%s"%self.model_name, "Residuals")
            show_residual_plot(self.port, self.model, resids, self.phases,
                    self.freqs, titles, bool(self.bw < 0), savefig=residplot)

    def model_iteration(self, quiet=False):
        """
        Iterate over a model fit.
        """
        while (1):
            start = time.time()
            fgp = fit_gaussian_portrait(self.portx, self.model_params,
                    self.portx_noise, self.fit_flags, self.phases,
                    self.freqsxs[0], self.nu_ref, self.all_join_params,
                    self.Ps[0], quiet=quiet)
            (self.fitted_params, self.fit_errs, self.chi2, self.dof) = (
                    fgp.fitted_params, fgp.fit_errs, fgp.chi2, fgp.dof)
            if self.njoin:
                self.model_params = self.fitted_params[:-self.njoin*2]
                self.model_param_errs = self.fit_errs[:-self.njoin*2]
                self.join_params = self.fitted_params[-self.njoin*2:]
                self.join_param_errs = self.fit_errs[-self.njoin*2:]
                self.all_join_params[1] = self.join_params
                #FIX, convergence can be based on residuals
                self.phi = 0.5
                self.phierr = 0.0
                self.DM = 1.0
                self.DMerr = 0.0
                self.red_chi2 = fgp.chi2 / fgp.dof
                #This function is a hack for now.
                self.write_join_parameters()
            else:
                self.model_params = self.fitted_params[:]
                self.model_param_errs = self.fit_errs[:]
            self.model = gen_gaussian_portrait(self.fitted_params,
                    self.phases, self.freqs, self.nu_ref,
                    self.join_ichans, self.Ps[0])
            self.model_masked = np.transpose(self.weights[0] *
                    np.transpose(self.model))
            self.modelx = np.compress(self.weights[0], self.model, axis=0)
            if not self.njoin:
                #Currently, fit_phase_shift returns an unbounded phase
                phase_guess = fit_phase_shift(self.portx.mean(axis=0),
                        self.modelx.mean(axis=0)).phase
                phase_guess %= 1
                if phase_guess > 0.5:
                    phase_guess -= 1.0
                DM_guess = 0.0
                fp = fit_portrait(self.portx, self.modelx,
                        np.array([phase_guess, DM_guess]), self.Ps[0],
                        self.freqsxs[0], self.nu_fit, None, None,
                        bounds=[(None, None), (None, None)], id=None,
                        quiet=True)
                self.fp_results = fp
                (self.phi, self.phierr, self.DM, self.DMerr, self.red_chi2) = (
                        fp.phase, fp.phase_err, fp.DM, fp.DM_err, fp.red_chi2)
            self.duration = time.time() - start
            self.total_time += self.duration
            yield

    def check_convergence(self, efac=1.0, quiet=False):
        """
        Check for convergence.

        Considers if the phase and DM in the data, as measured by the fitted
        model, are within the errors (times efac) of the measurements.

        It will not work if datafiles is a metafile.  This will be improved.

        quiet=True suppresses output.
        """
        if not quiet:
            print "Iter %d:"%(self.itern - self.niter)
            print " duration of %.2f min"%(self.duration /  60.)
            if not self.njoin:
                print " phase offset of %.2e +/- %.2e [rot]"%(self.phi,
                        self.phierr)
                print " DM of %.6e +/- %.2e [cm**-3 pc]"%(self.DM, self.DMerr)
            print " red. chi**2 of %.2f."%self.red_chi2
        else:
            if self.niter and (self.itern - self.niter) != 0:
                print "Iter %d..."%(self.itern - self.niter)
        if min(abs(self.phi), abs(1 - self.phi)) < abs(self.phierr)*efac:
            if abs(self.DM) < abs(self.DMerr)*efac:
                print "\nIteration converged.\n"
                return 1
        else:
            return 0

    def write_model(self, outfile=None, append=False, quiet=False):
        """
        Write the model parameters to file.

        outfile is a string designating the name of the output model file name.
        append=True will append to an already existing file of the same name.
        quiet=True suppresses output.
        """
        if outfile is None:
            outfile = self.datafile + ".gmodel"
        model_params = np.copy(self.model_params)
        #Aesthetic mod?
        model_params[2::6] = np.where(model_params[2::6] >= 1.0,
                model_params[2::6] % 1, model_params[2::6])
        #Convert tau (scattering timescale) to sec
        model_params[1] *= self.Ps[0] / self.nbin
        write_model(outfile, self.model_name, self.nu_ref, model_params,
                self.fit_flags, append=append, quiet=quiet)

    def write_errfile(self, errfile=None, append=False, quiet=False):
        """
        Write the model parameter uncertainties to file.

        errfile is a string designating the name of the parameter error file.
        append=True will append to an already existing file of the same name.
        quiet=True suppresses output.
        """
        if errfile is None:
            errfile = self.datafile + ".gmodel_errs"
        model_param_errs = np.copy(self.model_param_errs)
        #Convert tau (scattering timescale) to sec
        model_param_errs[1] *= self.Ps[0] / self.nbin
        write_model(errfile, self.model_name+"_errors", self.nu_ref,
                model_param_errs, self.fit_flags, append=append, quiet=quiet)

    def write_join_parameters(self):
        """
        Write the JOIN parameters to file.

        This function is a hack until something better is developed for how to
        deal with these alignment parameters.
        """
        print "JOIN Parameters:", self.join_params
        joinfile = self.metafile + ".join"
        jf = open(joinfile, "a")
        header = "# archive name" + " "*32 + "phase offset [rot]" + " "*2 + \
                "delta-DM [cm**-3 pc]\n"
        jf.write(header)
        for ifile in xrange(len(self.datafiles)):
            datafile = self.datafiles[ifile]
            phase = self.join_params[ifile*2]
            dm = self.join_params[ifile*2 + 1]
            line = datafile + " "*abs(45-len(datafile)) + "% .10f"%phase + \
                    " "*7 + "% .6f"%dm + "\n"
            jf.write(line)
        jf.close()

    def show_data_portrait(self):
        """
        Show the data portrait.

        See show_portrait(...)
        """
        title = "%s Portrait"%self.source
        show_portrait(np.transpose(self.weights) * self.port, self.phases,
                self.freqs, title, True, True, bool(self.bw < 0))

    def show_model_fit(self):
        """
        Show the model, data, and residuals.

        See show_residual_plot(...)
        """
        resids = self.port - self.model_masked
        titles = ("%s"%self.datafile, "%s"%self.model_name, "Residuals")
        show_residual_plot(self.port, self.model, resids, self.phases,
                self.freqs, titles, bool(self.bw < 0))

class GaussianSelector:
    """
    GaussianSelector is a class for hand-fitting gaussian components.

    Taken and tweaked from SMR's pygaussfit.py
    """

    def __init__(self, ax, profile, errs, tau=0.0, fixscat=True, minspanx=None,
            minspany=None, useblit=True, auto_gauss=0.0):
        """
        Initialize the input parameters and open the interactive window.

        ax is a pyplot axis.
        profile is an array of pulse profile data values.
        errs specifies the uncertainty on the profile values.
        tau is a scattering timescale [bin].
        fixscat=True does not fit for the scattering timescale.
        minspanx, minspany are vestigial.
        useblit should be True.
        auto_gauss != 0.0 specifies the initial guess at a width of a single
            gaussian component to be fit automatically.
        """
        if not auto_gauss:
            print ""
            print "============================================="
            print "Left mouse click to draw a Gaussian component"
            print "Middle mouse click to fit components to data"
            print "Right mouse click to remove a component"
        print "============================================="
        print "Press 'q' or close window when done fitting"
        print "============================================="
        self.ax = ax.axes
        self.profile = profile
        self.proflen = len(profile)
        self.phases = np.arange(self.proflen, dtype='d') / self.proflen
        self.errs = errs
        self.tauguess = tau #in bins
        self.fit_scattering = not fixscat
        if self.fit_scattering and self.tauguess == 0.0:
            self.tauguess = 0.5 #seems to break otherwise
        self.visible = True
        self.DCguess = sorted(profile)[len(profile)/10 + 1]
        self.init_params = [self.DCguess, self.tauguess]
        self.ngauss = 0
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.canvas.mpl_connect('draw_event', self.update_background)
        self.canvas.mpl_connect('key_press_event', self.keypress)
        self.background = None
        self.rectprops = dict(facecolor='white', edgecolor = 'black',
                              alpha=0.5, fill=False)
        self.to_draw = Rectangle((0,0), 0, 1, visible=False, **self.rectprops)
        self.ax.add_patch(self.to_draw)
        self.useblit = useblit
        self.minspanx = minspanx
        self.minspany = minspany
        # will save the data (position at mouseclick)
        self.eventpress = None
        # will save the data (pos. at mouserelease)
        self.eventrelease = None
        self.plot_gaussians(self.init_params)
        self.auto_gauss = auto_gauss
        if self.auto_gauss:
            amp = self.profile.max()
            wid = self.auto_gauss
            first_gauss = amp*gaussian_profile(self.proflen, 0.5, wid)
            loc = 0.5 + fit_phase_shift(self.profile, first_gauss,
                    self.errs).phase
            self.init_params += [loc, wid, amp]
            self.ngauss += 1
            self.plot_gaussians(self.init_params)
            print "Auto-fitting single gaussian profile..."
            fgp = fit_gaussian_profile(self.profile, self.init_params,
                    np.zeros(self.proflen) + self.errs, self.fit_scattering,
                    quiet=True)
            self.fitted_params = fgp.fitted_params
            self.fit_errs = fgp.fit_errs
            self.chi2 = fgp.chi2
            self.dof = fgp.dof
            self.residuals = fgp.residuals
            # scaled uncertainties
            #scaled_fit_errs = fit_errs * np.sqrt(chi_sq / dof)

            # Plot the best-fit profile
            self.plot_gaussians(self.fitted_params)
            fitprof = gen_gaussian_profile(self.fitted_params, self.proflen)
            plt.plot(self.phases, fitprof, c='black', lw=1)
            plt.draw()

            # Plot the residuals
            plt.subplot(212)
            plt.cla()
            residuals = self.profile - fitprof
            plt.plot(self.phases, residuals,'k')
            plt.xlabel('Pulse Phase')
            plt.ylabel('Data-Fit Residuals')
            plt.draw()
            self.eventpress = None
            # will save the data (pos. at mouserelease)
            self.eventrelease = None

    def update_background(self, event):
        """force an update of the background"""
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def ignore(self, event):
        """return True if event should be ignored"""
        # If no button was pressed yet ignore the event if it was out
        # of the axes
        if self.eventpress == None:
            return event.inaxes!= self.ax
        # If a button was pressed, check if the release-button is the
        # same.
        return (event.inaxes != self.ax or
                event.button != self.eventpress.button)

    def press(self, event):
        """on button press event"""
        # Is the correct button pressed within the correct axes?
        if self.ignore(event): return
        # make the drawed box/line visible get the click-coordinates,
        # button, ...
        self.eventpress = event
        if event.button == 1:
            self.to_draw.set_visible(self.visible)
            self.eventpress.ydata = self.DCguess

    def release(self, event):
        """on button release event"""
        if self.eventpress is None or self.ignore(event): return
        # release coordinates, button, ...
        self.eventrelease = event
        if event.button == 1:
            # make the box/line invisible again
            self.to_draw.set_visible(False)
            self.canvas.draw()
            xmin, ymin = self.eventpress.xdata, self.eventpress.ydata
            xmax, ymax = self.eventrelease.xdata, self.eventrelease.ydata
            # calculate dimensions of box 
            if xmin > xmax: xmin, xmax = xmax, xmin
            if ymin > ymax: ymin, ymax = ymax, ymin
            spanx = xmax - xmin
            spany = ymax - ymin
            xproblems = self.minspanx is not None and spanx < self.minspanx
            yproblems = self.minspany is not None and spany < self.minspany
        # call desired function
        self.onselect()
        self.eventpress = None                # reset the variables to their
        self.eventrelease = None              #   inital values

    def update(self):
        """draw using blit or old draw depending on useblit"""
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.to_draw)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def onmove(self, event):
        """on move event"""
        if self.eventpress is None or self.ignore(event): return
        x, y = event.xdata, event.ydata         # actual position 
                                                # with button still pressed
        minx, maxx = self.eventpress.xdata, x   # click-x and actual mouse-x
        miny, maxy = self.eventpress.ydata, y   # click-y and actual mouse-y
        if minx > maxx: minx, maxx = maxx, minx # get them in the right order
        if miny > maxy: miny, maxy = maxy, miny
        self.to_draw.set_x(minx)                # set lower left of box
        self.to_draw.set_y(miny)
        self.to_draw.set_width(maxx - minx)     # set width and height of box
        self.to_draw.set_height(maxy - miny)
        self.update()

    def keypress(self, event):
        """on key press event"""
        if self.ignore(event): return
        self.eventpress = event
        if event.key == 'q':
            self.eventpress = None
            self.close()

    def plot_gaussians(self, params):
        """plot gaussian components and profile"""
        plt.subplot(211)
        plt.cla()
        # Re-plot the original profile
        plt.hlines(0, 0.0, 1.0, color='black', lw=1, alpha=0.3, linestyle=':')
        plt.plot(self.phases, self.profile, c='black', lw=3, alpha=0.3)
        plt.xlabel('Pulse Phase')
        plt.ylabel('Pulse Amplitude')
        DC = params[0]
        tau = params[1]
        # Plot the individual gaussians
        for igauss in xrange(self.ngauss):
            loc, wid, amp = params[(2 + igauss*3):(5 + igauss*3)]
            plt.plot(self.phases, DC + amp*gaussian_profile(self.proflen, loc,
                wid), '%s'%cols[igauss])

    def onselect(self):
        """on select event"""
        event1 = self.eventpress
        event2 = self.eventrelease
        # Left mouse button = add a gaussian
        if event1.button == event2.button == 1:
            x1, y1 = event1.xdata, event1.ydata
            x2, y2 = event2.xdata, event2.ydata
            loc = 0.5 * (x1 + x2)
            wid = np.fabs(x2 - x1)
            #amp = np.fabs(1.05 * (y2 - self.init_params[0]) * (x2 - x1))
            amp = np.fabs(1.05 * (y2 - self.init_params[0]))
            self.init_params += [loc, wid, amp]
            self.ngauss += 1
            self.plot_gaussians(self.init_params)
            plt.draw()
        # Middle mouse button = fit the gaussians
        elif event1.button == event2.button == 2:
            print "Fitting reference gaussian profile..."
            fgp = fit_gaussian_profile(self.profile, self.init_params,
                    np.zeros(self.proflen) + self.errs, self.fit_scattering,
                    quiet=True)
            self.fitted_params = fgp.fitted_params
            self.fit_errs = fgp.fit_errs
            self.chi2 = fgp.chi2
            self.dof = fgp.dof
            self.residuals = fgp.residuals
            # scaled uncertainties
            #scaled_fit_errs = fit_errs * np.sqrt(chi_sq / dof)

            # Plot the best-fit profile
            self.plot_gaussians(self.fitted_params)
            fitprof = gen_gaussian_profile(self.fitted_params, self.proflen)
            plt.plot(self.phases, fitprof, c='black', lw=1)
            plt.draw()

            # Plot the residuals
            plt.subplot(212)
            plt.cla()
            residuals = self.profile - fitprof
            plt.plot(self.phases, self.residuals,'k')
            plt.xlabel('Pulse Phase')
            plt.ylabel('Data-Fit Residuals')
            plt.draw()
        # Right mouse button = remove last gaussian
        elif event1.button == event2.button == 3:
            if self.ngauss:
                self.init_params = self.init_params[:-3]
                self.ngauss -= 1
                self.plot_gaussians(self.init_params)
                plt.draw()
                plt.subplot(212)
                plt.cla()
                plt.xlabel('Pulse Phase')
                plt.ylabel('Data-Fit Residuals')
                plt.draw()

    def close(self):
        """close"""
        plt.close(1)
        plt.close(2)


if __name__ == "__main__":

    from optparse import OptionParser

    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    #parser.add_option("-h", "--help",
    #                  action="store_true", dest="help", default=False,
    #                  help="Show this help message and exit.")
    parser.add_option("-d", "--datafile",
                      action="store", metavar="archive", dest="datafile",
                      help="PSRCHIVE archive from which to generate gaussian model.")
    parser.add_option("-M", "--metafile",
                      action="store", metavar="metafile", dest="metafile",
                      help="(BETA) Will be able to fit several obs. from different bands.  NB: First file in metafile MUST also be the one that contains nu_ref.")
    parser.add_option("-o", "--outfile",
                      action="store", metavar="outfile", dest="outfile",
                      help="Name of output model file name. [default=archive.gmodel]")
    parser.add_option("-e", "--errfile",
                      action="store", metavar="errfile", dest="errfile",
                      help="Name of parameter error file name. [default=outfile_err]")
    parser.add_option("-m", "--model_name",
                      action="store", metavar="model_name", dest="model_name",
                      help="Name given to model. [default=PSRCHIVE Source name]")
    parser.add_option("--nu_ref",
                      action="store", metavar="nu_ref", dest="nu_ref",
                      default=None,
                      help="Reference frequency [MHz] for the gaussian model; the initial profile to fit will be centered on this freq. [default=PSRCHIVE center frequency]")
    parser.add_option("--bw",
                      action="store", metavar="bw", dest="bw_ref",
                      default=None,
                      help="Used with --nu_ref; amount of bandwidth [MHz] centered on nu_ref to average for the initial profile fit. [default=Full bandwidth]")
    parser.add_option("--tau",
                      action="store", metavar="tau", dest="tau", default=0.0,
                      help="Scattering timescale [sec] at nu_ref, assuming alpha=-4.0 (which can be changed internally).  [default=0]")
    parser.add_option("--fitloc",
                      action="store_false", dest="fixloc", default=True,
                      help="Do not fix locations of gaussians across frequency. Use this flag to allow gaussian components to drift with frequency. [default=False]")
    parser.add_option("--fixwid",
                      action="store_true", dest="fixwid", default=False,
                      help="Fix widths of gaussians across frequency. [default=False]")
    parser.add_option("--fixamp",
                      action="store_true", dest="fixamp", default=False,
                      help="Fix amplitudes of gaussians across frequency. [default=False]")
    parser.add_option("--fitscat",
                      action="store_true", dest="fitscat", default=False,
                      help="Fit scattering timescale to tau w.r.t nu_ref.  [default=False]")
    parser.add_option("--niter",
                      action="store", metavar="int", dest="niter", default=0,
                      help="Number of iterations to loop for generating better model. [default=0]")
    parser.add_option("--fgauss",
                      action="store_true", dest="fgauss", default=False,
                      help="Sets fitloc=True except for the first gaussian component fitted in the initial profile fit.  i.e. sets a 'fiducial gaussian'.")
    parser.add_option("--autogauss",
                      action="store", metavar="wid", dest="auto_gauss",
                      default=0.0,
                      help="Automatically fit one gaussian to initial profile with initial width [rot] given as the argument.")
    parser.add_option("--figure", metavar="figurename",
                      action="store", dest="figure", default=False,
                      help="Save PNG figure of final fit to figurename. [default=Not saved]")
    parser.add_option("--verbose",
                      action="store_false", dest="quiet", default=True,
                      help="More to stdout.")

    (options, args) = parser.parse_args()

    if options.datafile is None and options.metafile is None:
        print "\nppgauss.py - generate a gaussian-component model pulse portrait\n"
        parser.print_help()
        print ""
        parser.exit()

    datafile = options.datafile
    metafile = options.metafile
    if metafile is not None: datafile = metafile
    outfile = options.outfile
    errfile = options.errfile
    model_name = options.model_name
    if options.nu_ref: nu_ref = np.float64(options.nu_ref)
    else: nu_ref = options.nu_ref
    if options.bw_ref: bw_ref = np.float64(options.bw_ref)
    else: bw_ref = options.bw_ref
    tau = np.float64(options.tau)
    fixloc = options.fixloc
    fixwid = options.fixwid
    fixamp = options.fixamp
    fixscat = not options.fitscat
    niter = int(options.niter)
    fgauss = options.fgauss
    auto_gauss = float(options.auto_gauss)
    figure = options.figure
    quiet = options.quiet

    dp = DataPortrait(datafile=datafile, quiet=quiet)
    tau *= dp.nbin / dp.Ps[0]
    dp.make_gaussian_model(modelfile = None,
            ref_prof=(nu_ref, bw_ref), tau=tau, fixloc=fixloc, fixwid=fixwid,
            fixamp=fixamp, fixscat=fixscat, niter=niter,
            fiducial_gaussian=fgauss, auto_gauss=auto_gauss, writemodel=True,
            outfile=outfile, writeerrfile=True, errfile=errfile,
            model_name=model_name, residplot=figure, quiet=quiet)
