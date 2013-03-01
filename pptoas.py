#!/usr/bin/env python

from pplib import *

class GetTOAs:
    """
    """
    def __init__(self, datafiles, modelfile, DM0=None , bary_DM=True,
                 one_DM=False, common=True, quiet=False):
        """
        """
        start = time.time()
        self.datafiles = datafiles
        self.modelfile = modelfile
        self.DM0 = DM0
        self.bary_DM = bary_DM
        self.one_DM = one_DM
        self.common = common
        self.nsubs = []
        self.phis = []
        self.phi_errs = []
        self.DMs = []
        self.DM_errs = []
        self.scales = []
        self.scalesx = []
        self.scale_errs = []
        self.red_chi2s = []
        self.nfevals = []
        self.rcs = []
        self.MJDs = []
        self.fit_durations = []
        if len(self.datafiles) == 1 or self.common is True:
            self.data = load_data(self.datafile[0], dedisperse=False,
                    dededisperse=False, tscrunch=True, pscrunch=True,
                    rm_baseline=True, flux_prof=False, quiet=True)
            self.ngauss, self.model = read_model(self.modelfile,
                    self.data['phases'], self.data['freqs'], quiet))
            if len(self.datafiles) != 1:
                del(self.data)
            else:
                #Unpack the data dictionary into the local namespace; see
                #load_data for dictionary keys.
                for key in self.data.keys():
                    exec("self." + key + " = self.data['" + key + "']")
                if self.source is None: self.source = "noname"
                self.lofreq = self.freqs[0]-(self.bw/(2*self.nchan))
            #Need to remember to del(last DMs, etc for uncommon multiple files)
            #after get_toas finishes; final self.model statement, etc and to
            #make arrays of all things

    def get_toas(showplot=False)
        """
        """
        for datafile in self.datafiles:
            fit_duration = 0.0
            #Load data
            if len(self.datafiles) == 1: data = self.data
            else:
                data = load_data(datafile, dedisperse=False,
                        dededisperse=False, tscrunch=False, pscrunch=True,
                        rm_baseline=True, flux_prof=False, quiet=True)
            #Unpack the data dictionary into the local namespace; see load_data
#           for dictionary keys.
            for key in data.keys():
                exec(key + " = data['" + key + "']")
            if source is None: source = "noname"
            #Read model
            if len(self.datafiles) !=1 and self.common is False:
                ngauss, model = read_model(self.modelfile, phases, freqs,
                        quiet=quiet)
            else:
                ngauss, model = self.ngauss, self.model
            phis = np.empty(nsub, dtype=np.double)
            phi_errs = np.empty(nsub)
            DMs = np.empty(nsub, dtype=np.float)
            DM_errs = np.empty(nsub)
            nfevals = np.empty(nsub, dtype='int')
            rcs = np.empty(nsub, dtype='int')
            scales = np.empty([nsub, nchan])
            #These next two are lists becuase in principle,
            #the subints could have different numbers of zapped channels.
            scalesx = []
            scale_errs = []
            red_chi2s = np.empty(nsub)
            MJDs = np.array([epochs[ii].in_days()
                for ii in xrange(nsub)], dtype=np.double)
            if self.DM0 is None:
                DM0 = arch.get_dispersion_measure()
            else:
                DM0 = self.DM0
            #FIX - determine observatory
            if write_TOAs:
                obs = arch.get_telescope()
                obs_codes = ["@", "0", "1", "2"]
                obs = "1"
            if not quiet:
                print "\nEach of the %d TOAs are approximately %.2f s"%(nsub,
                        arch.integration_length() / nsub)
                print "Doing Fourier-domain least-squares fit..."
            self.nsubs.append(nsub)
            for nn in range(nsub):
                portx = subintsx[nn][0]
                portx_fft = np.fft.rfft(portx, axis=1)
                modelx = np.compress(weights[nn], model, axis=1)
                freqsx = freqsx[nn]
                nu0 = nu0
                P = Ps[nn]
                MJD = MJDs[nn]
                ####################
                #DOPPLER CORRECTION#
                ####################
                #In principle, we should be able to correct the frequencies, but
                #since this is a messy business, it is easier to correct the DM
                #itself (below).
                #df = arch.get_Integration(nn).get_doppler_factor()
                #freqsx = doppler_correct_freqs(freqsx, df)
                #nu0 = doppler_correct_freqs(nu0, df)
                ####################
                if nn == 0:
                    rot_portx = rotate_portrait(subints.mean(axis=0)[0],
                            0.0, DM0, P, freqsx, nu0)
                    #PSRCHIVE Dedisperses w.r.t. center of band...??
                    #Currently, first_guess ranges between +/- 0.5
                    phaseguess = first_guess(rot_portx, model,
                            nguess=1000)
                    DMguess = DM0
                #    if not quiet: print "Phase guess: %.8f ; DM guess: %.5f"%(
                #            phaseguess, DMguess)
                #The below else clause might not be a good idea if RFI or
                #something throws it completely off, whereas first phaseguess
                #only depends on pulse profile...but there may be special cases
                #when invidual guesses are needed.
                #else:
                #    phaseguess = phis[nn-1]
                #    DMguess = DMs[nn-1]
                #   if not quiet:
                #       print """
                #       Phase guess: %.8f
                #       DM guess:    %.5f"""%(phaseguess, DMguess)
                #
                #Need a status bar?
                if not quiet: print "Fitting for TOA %d"%(nn+1)
                phi, DM, nfeval, rc, scalex, param_errs, red_chi2, duration = \
                        fit_portrait(subintsx[nn][0], modelx,
                                np.array([phaseguess, DMguess]), P, freqsx,
                                nu0, scales=True)
                fit_duration += duration
                phis[nn] = phi
                phi_errs[nn] = param_errs[0]
                ####################
                #DOPPLER CORRECTION#
                ####################
                if bary_DM: #Default is True
                    #NB: the 'doppler factor' retrieved from PSRCHIVE seems to be
                    #the inverse of the convention nu_source/nu_observed
                    df = arch.get_Integration(nn).get_doppler_factor()
                    DM *= df
                DMs[nn] = DM
                DM_errs[nn] = param_errs[1]
                nfevals[nn] = nfeval
                rcs[nn] = rc
                scalesx.append(scalex)
                scale_errs.append(param_errs[2:])
                scale = np.zeros(nchan)
                ss = 0
                for ii in range(nchan):
                    if weights[nn,ii] == 1:
                        scale[ii] = scalex[ss]
                        ss += 1
                    else: pass
                scales[nn] = scale
                red_chi2s[nn] = red_chi2
            self.phis.append(phis)
            self.phi_errs.append(phi_errs)
            self.DMs.append(DMs)
            self.DM_errs.append(DM_errs)
            self.scales.append(scales)
            self.scalesx.append(scalesx)
            self.scale_errs.append(scale_errs)
            self.red_chi2s.append(red_chi2s)
            self.nfevals.append(nfevals)
            self.rcs.append(rcs)
            self.MJDs.append(MJDs)
            self.fit_durations.append(fit_duration)

            DeltaDMs = DMs - DM0
            #The below returns the weighted mean and the sum of the weights,
            #but needs to do better in the case of small-error outliers from
            #RFI, etc.  Also, last TOA may mess things up...use median...?
            DeltaDM_mean, DeltaDM_var = np.average(DeltaDMs,
                    weights=DM_errs**-2, returned=True)
            DeltaDM_var = DeltaDM_var**-1
            if nsub > 1:
                #The below multiplie by the red. chi-squared to inflate the
                #errors.
                DeltaDM_var *= np.sum(((DeltaDMs - DeltaDM_mean)**2) /
                        (DM_errs**2)) / (len(DeltaDMs) - 1)
            DeltaDM_err = DeltaDM_var**0.5
            
            
            
            
    def write_toas():
            
            
            if write_TOAs:
                toas = np.array([epochs[nn] + pr.MJD((phis[nn] *
                    Ps[nn]) / (3600 * 24.)) for nn in xrange(nsub)])
                toa_errs = phi_errs * Ps * 1e6
                if outfile: sys.stdout = open(outfile,"a")
                #Currently writes topocentric frequencies
                #Need option for different kinds of TOA output
                for nn in range(nsub):
                    if one_DM:
                        write_princeton_toa(toas[nn].intday(), toas[nn].fracday(),
                                toa_errs[nn], nu0, DeltaDM_mean, obs=obs)
                    else:
                        write_princeton_toa(toas[nn].intday(), toas[nn].fracday(),
                                toa_errs[nn], nu0, DeltaDMs[nn], obs=obs)
            sys.stdout = sys.__stdout__
            tot_duration = time.time() - start
            if not quiet:
                print "-------------------------"
                print "~%.2f min/TOA"%(fit_duration / (60. * nsub))
                print "Total ~%.2f min/TOA"%(tot_duration / (60 * nsub))
                print "Avg. TOA error is %.3f us"%(phi_errs.mean() *
                    Ps.mean() * 1e6)
            if pam_cmd:
                pc = open("pam_cmds", "a")
                pam_ext = datafile[-datafile[::-1].find("."):] + ".rot"
                #The below returns the weighted mean and the sum of the weights,
                #but needs to do better in the case of small-error outliers from
                #RFI, etc.  Also, last TOA may mess things up...use median...?
                phi_mean, phi_var = np.average(phis,
                        weights=phi_errs**-2, returned=True)
                phi_var = phi_var**-1
                pc.write("pam -e %s -r %.7f -d %.5f %s\n"%(pam_ext, phi_mean,
                    DeltaDM_mean + DM0, datafile))
                pc.close()
            if errfile:
                ef = open(errfile, "a")
                if one_DM:
                    ef.write("%.5e\n"%DeltaDM_err)
                else:
                    for nn in range(nsub):
                        ef.write("%.5e\n"%DM_errs[nn])
            if showplot: self.show_results()

    def show_subint(self, subint=0):
        """
        subint 0 = python index 0
        """
        ii = subint
        title = "Subint %d"%subint
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
        interpolation = "none"
        origin = "lower"
        extent = (0.0, 1.0, self.freqs[0], self.freqs[-1])
        plt.subplot(221)
        plt.title("Data Portrait")
        plt.imshow(port, aspect=aspect, interpolation=interpolation,
                origin=origin, extent=extent)
        plt.subplot(222)
        plt.title("Fitted Model Portrait")
        plt.imshow(fitmodel, aspect=aspect, interpolation=interpolation,
                origin=origin, extent=extent)
        plt.subplot(223)
        plt.title("Residuals")
        plt.imshow(port - fitmodel, aspect=aspect, interpolation=interpolation,
                origin=origin, extent=extent)
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
                      help='Append pam commands to file "pam_cmd."')
    parser.add_option("--uncommon",
                      action="store_true", dest="uncommon", default=False,
                      help="If supplying a metafile, use this flag if the data are not homogenous (i.e. have different nchan, nbin, nu0)")
    parser.add_option("--showplot",
                      action="store_true", dest="showplot", default=False,
                      help="Plot fit results. Only useful if nsubint > 1.")
    parser.add_option("--quiet",
                      action="store_true", dest="quiet", default=False,
                      help="Minimal to stdout.")

    (options, args) = parser.parse_args()

    if (options.datafile is None and options.metafile is None or
            options.modelfile is None):
            print "\npptoas.py - least-squares fit for TOAs and DMs.\n"
            parser.print_help()
            parser.exit()

    datafile = options.datafile
    metafile = options.metafile
    modelfile = options.modelfile
    DM0 = options.DM0
    if DM0: DM0 = float(DM0)
    bary_DM = options.bary_DM
    one_DM = options.one_DM
    pam_cmd = options.pam_cmd
    outfile = options.outfile
    errfile = options.errfile
    common = not options.uncommon
    showplot = options.showplot
    quiet = options.quiet

    if metafile is None:
        datafiles = [datafile]
    else:
        datafiles = open(metafile, "r").readlines()
        datafiles = [datafiles[xx][:-1] for xx in xrange(len(datafiles)]
    gt = GetTOAs(datafiles=datafiles, modelfile=modelfile, DM0=DM0,
        bary_DM=bary_DM, one_DM=one_DM, common=common, quiet=quiet)
    gt.get_toas(showplot)
    gt.write_toas(outfile)
    if errfile is not None: gt.write_dm_errfile(errfile)
    if pam_cmd: gt.write_pam_cmds()
