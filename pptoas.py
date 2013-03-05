#!/usr/bin/env python

from pplib import *

class GetTOAs:
    """
    """
    def __init__(self, datafiles, modelfile, DM0=None , one_DM=False,
            bary_DM=True, common=True, quiet=False):
        """
        """
        self.datafiles = datafiles
        self.modelfile = modelfile
        self.DM0 = DM0
        self.one_DM = one_DM
        self.bary_DM = bary_DM
        self.common = common
        self.obs = []
        self.nu0s = []
        self.nsubs = []
        self.epochs = []
        self.MJDs = []
        self.Ps = []
        self.phis = []
        self.phi_errs = []
        self.DM0s = []
        self.DMs = []
        self.DM_errs = []
        self.DeltaDM_means = []
        self.DeltaDM_errs = []
        self.scales = []
        self.scalesx = []
        self.scale_errs = []
        self.red_chi2s = []
        self.nfevals = []
        self.rcs = []
        self.fit_durations = []
        self.quiet = quiet
        if len(self.datafiles) == 1 or self.common is True:
            self.data = load_data(self.datafiles[0], dedisperse=False,
                    dededisperse=False, tscrunch=True, pscrunch=True,
                    rm_baseline=True, flux_prof=False, quiet=True)
            self.model_name, self.ngauss, self.model = read_model(
                    self.modelfile, self.data['phases'], self.data['freqs'],
                    self.quiet)
            #Unpack the data dictionary into the local namespace; see load_data
            #for dictionary keys.
            #for key in self.data.keys():
            #    exec("self." + key + " = self.data['" + key + "']")
            #if self.source is None: self.source = "noname"
            self.nchan = self.data['nchan']
            self.nbin = self.data['nbin']
            self.nu0 = self.data['nu0']
            self.bw = self.data['bw']
            self.freqs = self.data['freqs']
            self.lofreq = self.freqs[0]-(self.bw/(2*self.nchan))
            del(self.data)

    def get_toas(self, datafile=None, show_plot=False, append=True, safe=False,
            quiet=False):
        """
        Only use append if you are prepared to keep track of the order in which
        you have appended to the global lists (it could then be different from
        the ordered metafile containing datafile names.
        """
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
                    rm_baseline=True, flux_prof=False, quiet=quiet)
            #Unpack the data dictionary into the local namespace; see load_data
#           for dictionary keys.
            for key in data.keys():
                exec(key + " = data['" + key + "']")
            if source is None: source = "noname"
            #Read model
            if len(datafiles) !=1 and self.common is False:
                self.model_name, self.ngauss, model = read_model(
                        self.modelfile, phases, freqs, quiet=quiet)
            else:
                ngauss, model = self.ngauss, self.model
            phis = np.empty(nsubx, dtype=np.double)
            phi_errs = np.empty(nsubx)
            DMs = np.empty(nsubx, dtype=np.float)
            DM_errs = np.empty(nsubx)
            nfevals = np.empty(nsubx, dtype='int')
            rcs = np.empty(nsubx, dtype='int')
            scales = np.empty([nsubx, nchan])
            #These next two are lists because in principle,
            #the subints could have different numbers of zapped channels.
            scalesx = []
            scale_errs = []
            red_chi2s = np.empty(nsubx)
            MJDs = np.array([epochs[nn].in_days()
                for nn in xrange(nsub)], dtype=np.double)
            if self.DM0 is None:
                DM0 = arch.get_dispersion_measure()
            else:
                DM0 = self.DM0
            if not quiet:
                print "\nEach of the %d TOAs are approximately %.2f s"%(nsubx,
                        arch.integration_length() / nsub)
                print "Doing Fourier-domain least-squares fit..."
            #These are the subintegration indices that haven't been zapped
            ok_sub_inds = map(int, np.compress(map(len,
                np.array(subintsx)[:,0]), np.arange(nsub)))
            for nn in range(nsubx):
                subi = ok_sub_inds[nn]
                MJD = MJDs[subi]
                P = Ps[subi]
                freqsx = freqsxs[subi]
                portx = subintsx[subi][0]
                portx_fft = np.fft.rfft(portx, axis=1)
                modelx = np.compress(weights[subi], model, axis=0)
                ####################
                #DOPPLER CORRECTION#
                ####################
                #In principle, we should be able to correct the frequencies, but
                #since this is a messy business, it is easier to correct the DM
                #itself (below).
                #df = arch.get_Integration(subi).get_doppler_factor()
                #freqsx = doppler_correct_freqs(freqsx, df)
                #nu0 = doppler_correct_freqs(nu0, df)
                ####################
                if nn == 0:
                    rot_port = rotate_portrait(subints.mean(axis=0)[0], 0.0,
                            DM0, P, freqs, nu0)
                    #PSRCHIVE Dedisperses w.r.t. center of band...??
                    #Currently, first_guess ranges between +/- 0.5
                    phaseguess = first_guess(rot_port, model, nguess=1000)
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
                if not quiet: print "Fitting for TOA %d"%(nn)
                phi, DM, nfeval, rc, scalex, param_errs, red_chi2, duration = \
                        fit_portrait(portx, modelx,
                            np.array([phaseguess, DMguess]), P, freqsx,
                            nu0, scales=True, quiet=quiet)
                fit_duration += duration
                phis[nn] = phi
                phi_errs[nn] = param_errs[0]
                ####################
                #DOPPLER CORRECTION#
                ####################
                if bary_DM: #Default is True
                    #NB: the 'doppler factor' retrieved from PSRCHIVE seems to be
                    #the inverse of the convention nu_source/nu_observed
                    df = arch.get_Integration(subi).get_doppler_factor()
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
                    if weights[subi,ii] == 1:
                        scale[ii] = scalex[ss]
                        ss += 1
                    else: pass
                scales[nn] = scale
                red_chi2s[nn] = red_chi2
            DeltaDMs = DMs - DM0
            #The below returns the weighted mean and the sum of the weights,
            #but needs to do better in the case of small-error outliers from
            #RFI, etc.  Also, last TOA may mess things up...use median...?
            DeltaDM_mean, DeltaDM_var = np.average(DeltaDMs,
                    weights=DM_errs**-2, returned=True)
            DeltaDM_var = DeltaDM_var**-1
            if nsubx > 1:
                #The below multiplie by the red. chi-squared to inflate the
                #errors.
                DeltaDM_var *= np.sum(((DeltaDMs - DeltaDM_mean)**2) /
                        (DM_errs**2)) / (len(DeltaDMs) - 1)
            DeltaDM_err = DeltaDM_var**0.5
            if append:
                self.obs.append(arch.get_telescope())
                self.nu0s.append(nu0)
                self.nsubs.append(nsubx)
                self.epochs.append(np.take(epochs, ok_sub_inds))
                self.MJDs.append(np.take(MJDs, ok_sub_inds))
                self.Ps.append(np.take(Ps, ok_sub_inds))
                self.phis.append(phis)
                self.phi_errs.append(phi_errs)
                self.DM0s.append(DM0)
                self.DMs.append(DMs)
                self.DM_errs.append(DM_errs)
                self.DeltaDM_means.append(DeltaDM_mean)
                self.DeltaDM_errs.append(DeltaDM_err)
                self.scales.append(scales)
                self.scalesx.append(scalesx)
                self.scale_errs.append(scale_errs)
                self.red_chi2s.append(red_chi2s)
                self.nfevals.append(nfevals)
                self.rcs.append(rcs)
                self.fit_durations.append(fit_duration)
            if not quiet:
                print "--------------------------"
                print "~%.2f min/TOA"%(fit_duration / (60. * nsubx))
                print "Avg. TOA error is %.3f us"%(phi_errs.mean() *
                        Ps.mean() * 1e6)
            if show_plot:
                stop = time.time()
                tot_duration += stop - start
                self.show_results(datafile)
                start = time.time()
            if not append:
                return (phis, phi_errs, DMs, DM_errs, DeltaDM_mean,
                        DeltaDM_err, scales, scalesx, scale_errs, red_chi2s,
                        nfevals, rcs, fit_duration)
            if safe:
                self.write_toas("pptoas_toas.bak", datafile)
                self.write_dm_errs("pptoas_dmerrs.bak", datafile)
        if not show_plot:
            tot_duration = time.time() - start
        if not quiet:
            print "--------------------------"
            print "Total time: %.2f, ~%.2f min/TOA"%(tot_duration / 60,
                    tot_duration / (60 * np.sum(np.array(self.nsubs))))

    def write_toas(self, outfile=None, datafile=None, retrn=False):
        """
        """
        #FIX - determine observatory
        if datafile is None:
            datafiles = self.datafiles
        else:
            datafiles = [datafile]
        for datafile in datafiles:
            dfi = datafiles.index(datafile)
            try:
                obs_code = obs_codes["%s"%self.obs[dfi].lower()]
            except(KeyError):
                obs_code = obs_codes["%s"%self.obs[dfi].upper()]
            nu0 = self.nu0s[dfi]
            nsubx = self.nsubs[dfi]
            epochs = self.epochs[dfi]
            Ps = self.Ps[dfi]
            phis = self.phis[dfi]
            phi_errs = self.phi_errs[dfi]
            toas = np.array([epochs[nn] + pr.MJD((phis[nn] *
                Ps[nn]) / (3600 * 24.)) for nn in xrange(nsubx)])
            toa_errs = phi_errs * Ps * 1e6
            if outfile is not None: sys.stdout = open(outfile,"a")
            #Currently writes topocentric frequencies
            #Need option for different kinds of TOA output
            for nn in range(nsubx):
                if self.one_DM:
                    DeltaDM_mean = self.DeltaDM_means[dfi]
                    write_princeton_toa(toas[nn].intday(), toas[nn].fracday(),
                            toa_errs[nn], nu0, DeltaDM_mean, obs=obs_code)
                else:
                    DeltaDMs = self.DMs[dfi] - self.DM0s[dfi]
                    write_princeton_toa(toas[nn].intday(), toas[nn].fracday(),
                            toa_errs[nn], nu0, DeltaDMs[nn], obs=obs_code)
        sys.stdout = sys.__stdout__
        if retrn: return toas, toa_errs

    def write_dm_errs(self, outfile=None, datafile=None):
        if datafile is None:
            datafiles = self.datafiles
        else:
            datafiles = [datafile]
        for datafile in datafiles:
            dfi = datafiles.index(datafile)
            nsubx = self.nsubs[dfi]
            of = open(outfile, "a")
            if self.one_DM:
                DeltaDM_err = self.DeltaDM_errs[dfi]
                for nn in range(nsubx):
                    of.write("%.5e\n"%DeltaDM_err)
            else:
                DM_errs = self.DM_errs[dfi]
                for nn in range(nsubx):
                    of.write("%.5e\n"%DM_errs[nn])

    def write_pam_cmds(self, outfile=None, datafile=None):
        if outfile is None: outfile = "pam_cmds"
        of = open(outfile, "a")
        if datafile is None:
            datafiles = self.datafiles
        else:
            datafiles = [datafile]
        for datafile in datafiles:
            dfi = datafiles.index(datafile)
            phis = self.phis[dfi]
            phi_errs = self.phi_errs[dfi]
            DeltaDM_mean = self.DeltaDM_means[dfi]
            DM0 = self.DM0s[dfi]
            pam_ext = datafile[-datafile[::-1].find("."):] + ".rot"
            #The below returns the weighted mean and the sum of the weights,
            #but needs to do better in the case of small-error outliers from
            #RFI, etc.  Also, last TOA may mess things up...use median...?
            phi_mean, phi_var = np.average(phis, weights=phi_errs**-2,
                    returned=True)
            phi_var = phi_var**-1
            pc.write("pam -e %s -r %.7f -d %.5f %s\n"%(pam_ext, phi_mean,
                DeltaDM_mean + DM0, datafile))
        of.close()

    def show_subint(self, datafile=None, subint=0, quiet=False):
        """
        subint 0 = python index 0
        """
        if datafile is not None:
            dfi = datafiles.index(datafile)
        else:
            dfi = 0
        data = load_data(datafile, dedisperse=True, dededisperse=False,
                tscrunch=False, pscrunch=True, rm_baseline=True,
                flux_prof=False, quiet=quiet)
        title = "%s ; subint %d"%(datafile, subint)
        show_port(data['subints'][subint,0], data['freqs'], title=title)

    def show_fit(self, datafile, subint=0, quiet=False):
        """
        subint 0 = python index 0
        """
        if datafile is not None:
            dfi = datafiles.index(datafile)
        else:
            dfi = 0
        data = load_data(datafile, dedisperse=True, dededisperse=False,
                tscrunch=False, pscrunch=True, rm_baseline=True,
                flux_prof=False, quiet=quiet)
        fitfig = plt.figure()
        nn = subint
        phi = self.phis[dfi][nn]
        DM = self.DMs[dfi][nn]
        scales = self.scales[dfi][nn]
        scalesx = self.scalesx[dfi][nn]
        freqs = data['freqs']
        freqsxs = data['freqsxs'][nn]
        nu0 = data['nu0']
        P = data['Ps'][nn]
        port = data['subints'][nn,0]
        port = rotate_portrait(port, phi, DM, P, freqs, nu0)
        model_name, ngauss, model = read_model(self.modelfile, phases, freqs,
                quiet=quiet)
        model_fitted = np.transpose(scales * np.transpose(rotate_portrait(
            model, -phi, 0.0, P, freqs, nu0)))
        aspect = "auto"
        interpolation = "none"
        origin = "lower"
        extent = (0.0, 1.0, self.freqs[0], self.freqs[-1])
        plt.subplot(221)
        plt.title("%s ; subint %d"%(datafile, nn))
        plt.imshow(port, aspect=aspect, interpolation=interpolation,
                origin=origin, extent=extent)
        plt.subplot(222)
        plt.title("%s; Fitted Model"%(model_name))
        plt.imshow(model_fitted, aspect=aspect, interpolation=interpolation,
                origin=origin, extent=extent)
        plt.subplot(223)
        plt.title("Residuals")
        plt.imshow(port - model_fitted, aspect=aspect,
                interpolation=interpolation,
                origin=origin, extent=extent)
        plt.colorbar()
        #plt.subplot(224)
        #plt.title(r"Log$_{10}$(abs(Residuals/Data))")
        #plt.imshow(np.log10(abs(port - model_fitted) / port), aspect=aspect,
        #           origin=origin, extent=extent)
        #plt.colorbar()
        plt.show()

    def show_results(self, datafile=None):
        """
        """
        if datafile:
            dfi = self.datafiles.index(datafile)
        else:
            dfi = 0
        nsubx = self.nsubs[dfi]
        MJDs = self.MJDs[dfi]
        Ps = self.Ps[dfi]
        phis = self.phis[dfi]
        phi_errs = self.phi_errs[dfi]
        DMs = self.DMs[dfi]
        DM_errs = self.DM_errs[dfi]
        DM0 = self.DM0s[dfi]
        DeltaDM_mean = self.DeltaDM_means[dfi]
        DeltaDM_err = self.DeltaDM_errs[dfi]
        rcs = self.rcs[dfi]
        cols = ['b','k','g','b','r']
        fig = plt.figure()
        pf = np.polynomial.polynomial.polyfit
        milli_sec_shift = phis * Ps * 1e3
        #Not sure weighting works...
        fit_results = pf(MJDs, milli_sec_shift, 1, full=True, w=phi_errs**-2)
        resids = (milli_sec_shift) - (fit_results[0][0] + (fit_results[0][1] *
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
        for nn in range(nsubx):
            ax1.errorbar(MJDs[nn], phis[nn] * Ps[nn] * 1e6, phi_errs[nn] *
                    Ps[nn] * 1e6, color='%s'%cols[rcs[nn]], fmt='+')
        plt.plot(MJDs, (fit_results[0][0] + (fit_results[0][1] * MJDs)) * 1e3,
                "m--")
        plt.xlabel("MJD")
        plt.ylabel(r"Offset [$\mu$s]")
        ax1.text(0.1, 0.9, "%.2e ms/s"%(fit_results[0][1] / (3600 * 24)),
                ha='center', va='center', transform=ax1.transAxes)
        ax2 = fig.add_subplot(312)
        for nn in range(nsubx):
            ax2.errorbar(MJDs[nn], resids[nn] * 1e3, phi_errs[nn] * Ps[nn] *
                    1e6, color='%s'%cols[rcs[nn]], fmt='+')
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
        for nn in range(nsubx):
            ax3.errorbar(MJDs[nn], DMs[nn], DM_errs[nn],
                    color='%s'%cols[rcs[nn]], fmt='+')
        if abs(DeltaDM_mean) / DeltaDM_err < 10:
            plt.plot(MJDs, np.ones(len(MJDs)) * DM0, "r-")
        plt.plot(MJDs, np.ones(len(MJDs)) * (DeltaDM_mean + DM0), "m--")
        xverts = [MJDs[0], MJDs[0], MJDs[-1], MJDs[-1]]
        yverts = [DeltaDM_mean + DM0 - DeltaDM_err,
                  DeltaDM_mean + DM0 + DeltaDM_err,
                  DeltaDM_mean + DM0 + DeltaDM_err,
                  DeltaDM_mean + DM0 - DeltaDM_err]
        plt.fill(xverts, yverts, "m", alpha=0.25, ec='none')
        plt.xlabel("MJD")
        plt.ylabel(r"DM [pc cm$^{3}$]")
        ax3.text(0.15, 0.9, r"$\Delta$ DM = %.2e $\pm$ %.2e"%(DeltaDM_mean,
            DeltaDM_err), ha='center',
                va='center', transform=ax3.transAxes)
        plt.title(datafile)
        plt.show()

    def show_hists(self, datafile=None):
        if datafile is None:
            nfevals = np.array(self.nfevals)
            rcs = np.array(self.rcs)
            nsubx = self.nsubs
        else:
            dfi = self.datafiles.index(datafile)
            nfevals = self.nfevals[dfi]
            rcs = self.rcs[dfi]
            nsubx = self.nsubs[dfi]
        cols = ['b','k','g','b','r']
        bins = nfevals.max()
        binmin = nfevals.min()
        rc1=np.zeros(bins - binmin + 1)
        rc2=np.zeros(bins - binmin + 1)
        rc4=np.zeros(bins - binmin + 1)
        rc5=np.zeros(bins - binmin + 1)
        for nn in range(nsubx):
            nfeval = nfevals[nn]
            if rcs[nn] == 1:
                rc1[nfeval - binmin] += 1
            elif rcs[nn] == 2:
                rc2[nfeval - binmin] += 1
            elif rcs[nn] == 4:
                rc4[nfeval - binmin] += 1
            else:
                print "rc %d discovered at rcs index %d"%(rcs[nn], nn)
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
            print "\npptoas.py - simultaneous least-squares fit for TOAs and DMs\n"
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
        datafiles = [datafiles[xx][:-1] for xx in xrange(len(datafiles))]
    gt = GetTOAs(datafiles=datafiles, modelfile=modelfile, DM0=DM0,
        one_DM=one_DM, bary_DM=bary_DM, common=common, quiet=quiet)
    gt.get_toas(show_plot=showplot, safe=False, quiet=quiet)
    gt.write_toas(outfile)
    if errfile is not None: gt.write_dm_errs(errfile)
    if pam_cmd: gt.write_pam_cmds()
