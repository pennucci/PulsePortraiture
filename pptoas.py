#!/usr/bin/env python

from pplib import *

class GaussianModelPortrait:
    """
    """
    def __init__(self, modelfile, nbin, freqs, portweights=None, quiet=False):
        """
        """
        self.modelfile = modelfile
        self.nbin = nbin
        self.freqs = freqs
        self.portweights = portweights
        self.phases = np.arange(nbin, dtype='d') / nbin
        self.source, self.ngauss, self.model = make_model(modelfile,
                self.phases, self.freqs, quiet=quiet)
        if portweights is not None:
            self.modelmasked, self.modelx = screen_portrait(self.model,
                    portweights)
        else:
            self.modelmasked, self.modelx = self.model, self.model

    def show_model_portrait(self):
        title = "%s Model Portrait"%self.source
        show_port(self.model, self.freqs, title=title)

class SmoothedModelPortrait:
    """
    """
    #FIX need to interpolate, or somehow account for 
    #missing channels in model...
    def __init__(self, modelfile, quiet=False):
        """
        """
        self.modelfile = modelfile
        (self.source,self.arch,self.port,self.portx,self.noise_stdev,self.fluxprof,self.fluxprofx,self.prof,self.nbin,self.phases,self.nu0,self.bw,self.nchan,self.freqs,self.freqsx,self.nsub,self.P,self.MJD,self.weights,self.normweights,self.maskweights,self.portweights) = load_data(modelfile,dedisperse=True,tscrunch=True,pscrunch=True,quiet=quiet,rm_baseline=(0,0))
        self.model = self.port
        self.modelx = self.portx

class GetTOAs:
    """
    """
    def __init__(self, datafile, modelfile, mtype=None, DM0=None ,bary_DM=True,
                 one_DM=False, pam_cmd=False, outfile=None, errfile=None,
                 write_TOAs=True, quiet=False):
        """
        """
        start = time.time()
        self.datafile = datafile
        self.modelfile = modelfile
        self.mtype = mtype
        self.outfile = outfile
        (self.source,self.arch,self.ports,self.portxs,self.noise_stdev,self.fluxprof,self.fluxprofx,self.prof,self.nbin,self.phases,self.nu0,self.bw,self.nchan,self.freqs,self.freqsx,self.nsub,self.Ps,self.epochs,self.weights,self.normweights,self.maskweights,self.portweights) = load_data(datafile,dedisperse=False,tscrunch=False,pscrunch=True,quiet=quiet,rm_baseline=(0,0))
        self.phis = np.empty(self.nsub,dtype=np.double)
        self.phi_errs = np.empty(self.nsub)
        self.DMs = np.empty(self.nsub,dtype=np.float)
        self.DM_errs = np.empty(self.nsub)
        self.nfevals = np.empty(self.nsub, dtype='int')
        self.rcs = np.empty(self.nsub, dtype='int')
        self.scales = np.empty([self.nsub, self.nchan])
        #These next two are lists becuase in principle,
        #the subints could have different numbers of zapped channels.
        self.scalesx = []
        self.scale_errs = []
        self.red_chi2s = np.empty(self.nsub)
        self.fit_duration = 0.0
        self.MJDs = np.array([self.epochs[ii].in_days()
            for ii in xrange(self.nsub)],dtype=np.double)
        if DM0:
            self.DM0 = DM0
        else:
            self.DM0 = self.arch.get_dispersion_measure()
        if self.mtype == "gauss":
            self.modelportrait = GaussianModelPortrait(modelfile, self.nbin,
                    self.freqs, portweights=None, quiet=quiet)
        elif self.mtype == "smooth":
            self.modelportrait=SmoothedModelPortrait(modelfile, quiet=quiet)
        else:
            print 'Model type must be either "gauss" or "smooth".'
            sys.exit()
        mp = self.modelportrait
        #FIX - determine observatory
        if write_TOAs:
            obs = self.arch.get_telescope()
            obs_codes = ["@", "0", "1", "2"]
            obs = "1"
        if not quiet:
            print "\nEach of the %d TOAs are approximately %.2f s"%(self.nsub,
                    self.arch.integration_length() / self.nsub)
            print "Doing Fourier-domain least-squares fit..."
        for nn in range(self.nsub):
            dataportrait = self.portxs[nn]
            portx_fft = np.fft.rfft(dataportrait, axis=1)
            pw = self.portweights[nn]
            model, modelx = screen_portrait(mp.model, pw)
            freqsx = ma.masked_array(self.freqs,
                    mask=self.maskweights[nn]).compressed()
            nu0 = self.nu0
            P = self.Ps[nn]
            MJD = self.MJDs[nn]
            ####################
            #DOPPLER CORRECTION#
            ####################
            #In principle, we should be able to correct the frequencies, but
            #since this is a messy business, it is easier to correct the DM
            #itself (below).
            #df = self.arch.get_Integration(nn).get_doppler_factor()
            #freqsx = doppler_correct_freqs(freqsx,df)
            #nu0 = doppler_correct_freqs(self.nu0,df)
            ####################
            if nn == 0:
                rot_dataportrait = rotate_portrait(self.portxs.mean(axis=0),
                        0.0, self.DM0,P, freqsx, nu0)
                #PSRCHIVE Dedisperses w.r.t. center of band...??
                #Currently, first_guess ranges between +/- 0.5
                phaseguess = first_guess(rot_dataportrait,modelx,nguess=1000)
                DMguess = self.DM0
                #if not quiet: print "Phase guess: %.8f ; DM guess: %.5f"%(
                #        phaseguess, DMguess)
            #The below else clause might not be a good idea if RFI or something
            #throws it completely off, whereas first phaseguess only depends
            #on pulse profile...but there may be special cases when invidual
            #guesses are needed.
            #else:
            #    phaseguess = self.phis[nn-1]
            #    DMguess = self.DMs[nn-1]
            #   if not quiet:
            #       print """
            #       Phase guess: %.8f ; DM guess: %.5f"%(phaseguess, DMguess)
            #       """
            #Need a status bar?
            if not quiet: print "Fitting for TOA %d"%(nn+1)
            phi, DM, nfeval, rc, scalex, param_errs, red_chi2, duration = \
                    fit_portrait(self.portxs[nn], modelx, np.array([phaseguess,
                        DMguess]), P, freqsx, nu0, scales=True)
            self.fit_duration += duration
            self.phis[nn] = phi
            self.phi_errs[nn] = param_errs[0]
            ####################
            #DOPPLER CORRECTION#
            ####################
            if bary_DM: #Default is True
                #NB: the 'doppler factor' retrieved from psrchive seems to be
                #the inverse of the convention nu_source/nu_observed
                df = self.arch.get_Integration(nn).get_doppler_factor()
                DM *= df
            self.DMs[nn] = DM
            self.DM_errs[nn] = param_errs[1]
            self.nfevals[nn] = nfeval
            self.rcs[nn] = rc
            self.scalesx.append(scalex)
            self.scale_errs.append(param_errs[2:])
            scale = np.zeros(self.nchan)
            ss = 0
            for ii in range(self.nchan):
                if self.normweights[nn, ii] == 1:
                    scale[ii] = scalex[ss]
                    ss += 1
                else: pass
            self.scales[nn] = scale
            self.red_chi2s[nn] = red_chi2
        self.DeltaDMs = self.DMs - self.DM0
        #The below returns the weighted mean and the sum of the weights, but
        #needs to do better in the case of small-error outliers from RFI, etc.         #Also, last TOA may mess things up...use median...?
        self.DeltaDM_mean, self.DeltaDM_var = np.average(self.DeltaDMs,
                weights=self.DM_errs**-2, returned=True)
        self.DeltaDM_var = self.DeltaDM_var**-1
        if self.nsub > 1:
            #The below multiplie by the red. chi-squared to inflate the errors.
            self.DeltaDM_var *= np.sum(((self.DeltaDMs-self.DeltaDM_mean)**2) /
                    (self.DM_errs**2)) / (len(self.DeltaDMs) - 1)
        self.DeltaDM_err = self.DeltaDM_var**0.5
        if write_TOAs:
            toas = np.array([self.epochs[nn] + pr.MJD((self.phis[nn] *
                self.Ps[nn]) / (3600 * 24.)) for nn in xrange(self.nsub)])
            toa_errs = self.phi_errs * self.Ps * 1e6
            if self.outfile: sys.stdout = open(self.outfile,"a")
            #Currently writes topocentric frequencies
            #Need option for different kinds of TOA output
            for nn in range(self.nsub):
                if one_DM:
                    write_princeton_toa(toas[nn].intday(), toas[nn].fracday(),
                            toa_errs[nn], self.nu0, self.DeltaDM_mean, obs=obs)
                else:
                    write_princeton_toa(toas[nn].intday(), toas[nn].fracday(),
                            toa_errs[nn], self.nu0, self.DeltaDMs[nn], obs=obs)
        sys.stdout = sys.__stdout__
        self.tot_duration = time.time() - start
        if not quiet:
            print "-------------------------"
            print "~%.2f min/TOA"%(self.fit_duration / (60. * self.nsub))
            print "Total ~%.2f min/TOA"%(self.tot_duration / (60 * self.nsub))
            print "Avg. TOA error is %.3f us"%(self.phi_errs.mean() *
                self.Ps.mean() * 1e6)
        if pam_cmd:
            pc = open("pam_cmds", "a")
            pam_ext = self.datafile[-self.datafile[::-1].find("."):] + ".rot"
            #The below returns the weighted mean and the sum of the weights,
            #but needs to do better in the case of small-error outliers from
            #RFI, etc.  Also, last TOA may mess things up...use median...?
            self.phi_mean, self.phi_var = np.average(self.phis,
                    weights=self.phi_errs**-2, returned=True)
            self.phi_var = self.phi_var**-1
            pc.write("pam -e %s -r %.7f -d %.5f %s\n"%(pam_ext, self.phi_mean,
                self.DeltaDM_mean + self.DM0, self.datafile))
            pc.close()
        if errfile:
            ef = open(errfile, "a")
            if one_DM:
                ef.write("%.5e\n"%self.DeltaDM_err)
            else:
                for nn in range(self.nsub):
                    ef.write("%.5e\n"%self.DM_errs[nn])

    def show_subint(self, subint=0):
        """
        subint 0 = python index 0
        """
        ii = subint
        title = "Subint %d"%(subint)
        show_port(self.ports[ii], self.freqs, title=title)

    def show_fit(self, subint=0):
        """
        subint 0 = python index 0
        """
        fitfig = plt.figure()
        ii = subint
        phi = self.phis[ii]
        DM = self.DMs[ii]
        scales = self.scales[ii]
        scalesx = self.scalesx[ii]
        freqs = self.freqs
        freqsx = self.freqsx
        nu0 = self.nu0
        P = self.Ps[ii]
        port = self.ports[ii]
        portx = self.portxs[ii]
        model, modelx = screen_portrait(self.modelportrait.model,
                self.portweights[ii])
        fitmodel = np.transpose(self.scales[ii] * np.transpose(
            rotate_portrait(model, -phi, 0.0, P, freqs, nu0)))
        fitmodelx = np.transpose(self.scalesx[ii] * np.transpose(
            rotate_portrait(modelx, -phi, 0.0, P, freqsx, nu0)))
        port = rotate_portrait(port, phi, DM, P, freqs, nu0)
        aspect = "auto"
        origin = "lower"
        extent = (0.0, 1.0, self.freqs[0], self.freqs[-1])
        plt.subplot(221)
        plt.title("Data Portrait")
        plt.imshow(port, aspect=aspect, origin=origin, extent=extent)
        plt.subplot(222)
        plt.title("Fitted Model Portrait")
        plt.imshow(fitmodel, aspect=aspect, origin=origin, extent=extent)
        plt.subplot(223)
        plt.title("Residuals")
        plt.imshow(port - fitmodel, aspect=aspect, origin=origin,
                extent=extent)
        plt.colorbar()
        #plt.subplot(224)
        #plt.title(r"Log$_{10}$(abs(Residuals/Data))")
        #plt.imshow(np.log10(abs(port - fitmodel) / port), aspect=aspect,
        #           origin=origin, extent=extent)
        #plt.colorbar()
        plt.show()

    def show_results(self):
        """
        """
        cols = ['b','k','g','b','r']
        if fignum:
            fig = plt.figure(fignum)
        else:
            fig = plt.figure()
        pf = np.polynomial.polynomial.polyfit
        milli_sec_shift = self.phis * self.Ps * 1e3
        fit_results = pf(self.MJDs, milli_sec_shift, 1, full=True,
                w=self.phi_errs**-2)    #Not sure weighting works...
        resids = (milli_sec_shift) - (fit_results[0][0] +
                (fit_results[0][1] * self.MJDs))
        resids_mean,resids_var = np.average(resids, weights=self.phi_errs**-2,
                returned=True)
        resids_var = resids_var**-1
        if self.nsub > 1:
            resids_var *= np.sum(((resids - resids_mean)**2) /
                    (self.phi_errs**2))/(len(resids) - 1)
        resids_err = resids_var**0.5
        RMS = resids_err
        ax1 = fig.add_subplot(311)
        for nn in range(len(self.phis)):
            ax1.errorbar(self.MJDs[nn], self.phis[nn] * self.Ps[nn] * 1e6,
                    self.phi_errs[nn] * self.Ps[nn] * 1e6, color='%s'
                    %cols[self.rcs[nn]], fmt='+')
        plt.plot(self.MJDs, (fit_results[0][0] + (fit_results[0][1] *
            self.MJDs)) * 1e3, "m--")
        plt.xlabel("MJD")
        plt.ylabel(r"Offset [$\mu$s]")
        ax1.text(0.1, 0.9, "%.2e ms/s"%(fit_results[0][1] / (3600 * 24)),
                ha='center', va='center', transform=ax1.transAxes)
        ax2 = fig.add_subplot(312)
        for nn in range(len(self.phis)):
            ax2.errorbar(self.MJDs[nn], resids[nn] * 1e3, self.phi_errs[nn] *
                    self.Ps[nn] * 1e6, color='%s'%cols[self.rcs[nn]], fmt='+')
        plt.plot(self.MJDs, np.ones(len(self.MJDs)) * resids_mean * 1e3, "m--")
        xverts = np.array([self.MJDs[0], self.MJDs[0], self.MJDs[-1],
            self.MJDs[-1]])
        yverts = np.array([resids_mean - resids_err, resids_mean + resids_err,
            resids_mean + resids_err, resids_mean - resids_err]) * 1e3
        plt.fill(xverts, yverts, "m", alpha=0.25, ec='none')
        plt.xlabel("MJD")
        plt.ylabel(r"Offset [$\mu$s]")
        ax2.text(0.1, 0.9, r"$\sim$weighted RMS = %d ns"%int(resids_err * 1e6),
                ha='center', va='center', transform=ax2.transAxes)
        ax3 = fig.add_subplot(313)
        for nn in range(len(self.phis)):
            ax3.errorbar(self.MJDs[nn], self.DMs[nn], self.DM_errs[nn],
                    color='%s'%cols[self.rcs[nn]], fmt='+')
        if abs(self.DeltaDM_mean) / self.DeltaDM_err < 10:
            plt.plot(self.MJDs, np.ones(len(self.MJDs)) * self.DM0, "r-")
        plt.plot(self.MJDs, np.ones(len(self.MJDs)) * (self.DeltaDM_mean +
            self.DM0), "m--")
        xverts = [self.MJDs[0], self.MJDs[0], self.MJDs[-1], self.MJDs[-1]]
        yverts = [self.DeltaDM_mean + self.DM0 - self.DeltaDM_err,
                  self.DeltaDM_mean + self.DM0 + self.DeltaDM_err,
                  self.DeltaDM_mean + self.DM0 + self.DeltaDM_err,
                  self.DeltaDM_mean + self.DM0 - self.DeltaDM_err]
        plt.fill(xverts, yverts, "m", alpha=0.25, ec='none')
        plt.xlabel("MJD")
        plt.ylabel(r"DM [pc cm$^{3}$]")
        ax3.text(0.15, 0.9, r"$\Delta$ DM = %.2e $\pm$ %.2e"
                %(self.DeltaDM_mean, self.DeltaDM_err), ha='center',
                va='center', transform=ax3.transAxes)
        plt.show()

    def show_hists(self):
        cols = ['b','k','g','b','r']
        bins = self.nfevals.max()
        binmin = self.nfevals.min()
        rc1=np.zeros(bins - binmin + 1)
        rc2=np.zeros(bins - binmin + 1)
        rc4=np.zeros(bins - binmin + 1)
        rc5=np.zeros(bins - binmin + 1)
        for nn in range(len(self.nfevals)):
            nfeval = self.nfevals[nn]
            if self.rcs[nn] == 1:
                rc1[nfeval - binmin] += 1
            elif self.rcs[nn] == 2:
                rc2[nfeval - binmin] += 1
            elif self.rcs[nn] == 4:
                rc4[nfeval - binmin] += 1
            else:
                print "rc %d discovered!"%self.rcs[nn]
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
        urcs = np.unique(self.rcs)
        rchist = np.histogram(self.rcs, bins=len(urcs))[0]
        for uu in range(len(urcs)):
            rc = urcs[uu]
            plt.hist(np.ones(rchist[uu]) * rc, bins=1, color=cols[rc])
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
                      help="PSRCHIVE archive from which to generate TOAs.")
    parser.add_option("-M", "--metafile",
                      action="store",metavar="metafile", dest="metafile",
                      help="List of archive filenames in metafile.")
    parser.add_option("-m", "--modelfile",
                      action="store", metavar="model", dest="modelfile",
                      help=".model file to which the data are fit.")
    parser.add_option("-t", "--modeltype",
                      action="store", metavar="mtype", dest="mtype",
                      help='Must be either "gauss" (created by ppgauss.py) or "smooth" (created by psrsmooth).')
    parser.add_option("-o", "--outfile",
                      action="store", metavar="timfile", dest="outfile",
                      default=None,
                      help="Name of output .tim file name. Will append. [default=stdout]")
    parser.add_option("--DM",
                      action="store", metavar="DM", dest="DM0", default=None,
                      help="Nominal DM [pc cm**-3] (float) from which to measure offset.  If unspecified, will use the DM stored in the archive.")
    parser.add_option("--no_bary_DM",
                      action="store_false", dest="bary_DM", default=True,
                      help='Do not Doppler-correct the fitted DM to make "barycentric DM".')
    parser.add_option("--one_DM",
                      action="store_true", dest="one_DM", default=False,
                      help="Returns single DM value in output .tim file for the epoch instead of a fitted DM per subint.")
    parser.add_option("--errfile",
                      action="store", metavar="errfile", dest="errfile",
                      default=None,
                      help="If specified, will write the fitted DM errors to errfile. Will append.")
    parser.add_option("--pam_cmd",
                      action="store_true", dest="pam_cmd", default=False,
                      help="Append pam commands to file pam_cmd.")
    parser.add_option("--showplot",
                      action="store_true", dest="showplot", default=False,
                      help="Plot fit results. Only useful if nsubint > 1.")
    parser.add_option("--quiet",
                      action="store_true", dest="quiet", default=False,
                      help="Minimal to stdout.")

    (options, args) = parser.parse_args()

    if (options.datafile is None and options.metafile is None or
            options.modelfile is None or options.mtype is None):
            print "\npptoas.py - least-squares fit for TOAs and DMs.\n"
            parser.print_help()
            parser.exit()

    datafile = options.datafile
    metafile = options.metafile
    modelfile = options.modelfile
    mtype = options.mtype
    DM0 = options.DM0
    if DM0: DM0 = float(DM0)
    bary_DM = options.bary_DM
    one_DM = options.one_DM
    pam_cmd = options.pam_cmd
    outfile = options.outfile
    errfile = options.errfile
    showplot = options.showplot
    quiet = options.quiet

    if not metafile:
        gt = GetTOAs(datafile, modelfile, mtype, DM0, bary_DM, one_DM, pam_cmd,
                outfile, errfile, quiet=quiet)
        if showplot: gt.show_results()
    else:
        datafiles = open(metafile, "r").readlines()
        for datafile in datafiles:
            gt = GetTOAs(datafile[:-1], modelfile, mtype, DM0, bary_DM, one_DM,
                    pam_cmd, outfile, errfile, quiet=quiet)
            if showplot: gt.show_results()
