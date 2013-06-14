#!/usr/bin/env python

from pplib import *

#gaps in plotting
#which band to put first, hand rotate
#to use beginning template from data or gauss?
#stopping/convergence criteria
#smoothing
#documentation
#Stokes versus coherence
#independent alignment of different pols?  alignment via something other
#than phase shift? PPA?
#-F -T -P options for output standard profile?
#SNR cutoff!
#fiducial phase option
#all data same pol state, or pscrunch
#use norm_weights?
#proper t- f- p- scrunch calculations?

class AlignData:
    """
    """
    def __init__(self, datafiles, templatefile=None, gauss=0.0, niter=0,
            pscrunch=False, talign=False, falign=False, align_all=False,
            quiet=False):
        if file_is_ASCII(datafiles):
            self.metafile = datafiles
            self.datafiles = open(datafiles, "r").readlines()
            self.datafiles = [self.datafiles[iarch][:-1] for iarch in
                    xrange(len(self.datafiles))]
        else:
            self.datafiles = [datafiles]
        self.nfile = len(self.datafiles)
        self.templatefile = templatefile
        self.gauss = gauss
        self.niter = niter
        self.pscrunch = pscrunch
        if not self.pscrunch:
            print "\nNote pol'n state of files!"
            print "\nThis seems broken..."
            self.npol = 4
        else:
            self.npol = 1
        self.talign = talign
        self.falign = falign
        self.align_all = align_all
        if self.align_all:
            self.talign = self.falign = False
        self.quiet = quiet
        self.nsub = 0
        self.length = 0.0
        self.P = 0.0
        self.nchan = 0
        self.nu0s = []
        self.lofreq = np.inf
        self.hifreq = 0.0
        self.bws = []
        all_data = []
        epochs = []
        okisub = []
        freqs = []
        okichan = []
        weights = []
        print "\nReading data from archives..."
        for iarch in xrange(self.nfile):
            datafile = self.datafiles[iarch]
            data = load_data(datafile, dedisperse=True, dededisperse=False,
                    tscrunch=False, pscrunch=self.pscrunch, fscrunch=False,
                    rm_baseline=True, flux_prof=True, norm_weights=False,
                    quiet=self.quiet)
            if iarch == 0:
                self.nbin = data.nbin
                self.phases = data.phases
                self.state = data.state
                self.obs = data.arch.get_telescope()
                amp = data.prof.max()
                if data.source is None:
                    source = "None"
                else:
                    source = data.source
                if self.gauss:
                    wid = self.gauss
                    self.model = amp * gaussian_profile(self.nbin, loc=0.5,
                            wid=wid)
                else:
                    if self.templatefile is not None:
                        self.model = load_data(self.templatefile,
                                dedisperse=True, dededisperse=False,
                                tscrunch=True, pscrunch=True,
                                fscrunch=True, rm_baseline=True,
                                flux_prof=False, norm_weights=True,
                                quiet=True).prof
                    else:
                        self.model = data.prof
            self.nsub += data.nsub
            self.length += (data.arch.end_time() -
                    data.arch.start_time()).in_seconds()
            self.P += data.Ps.mean()
            epochs.extend(data.epochs[isub] for isub in range(data.nsub))
            okisub.append(data.okisub)
            self.nchan += data.nchan
            self.nu0s.append(data.nu0)
            lf = data.freqs.min() - (abs(data.bw) / (2*data.nchan))
            if lf < self.lofreq:
                self.lofreq = lf
            hf = data.freqs.max() + (abs(data.bw) / (2*data.nchan))
            if hf > self.hifreq:
                self.hifreq = hf
            self.bws.append(data.bw)
            freqs.append(data.freqs)
            okichan.append(data.okichan)
            all_data.append(data.subints)
            weights.append(data.weights)
        self.P /= len(self.datafiles)    #Will be used only as a toy period
        self.all_data = all_data
        self.epochs = epochs
        self.mean_MJD = np.array([np.array(self.epochs).ravel()[isub].in_days()
            for isub in range(self.nsub)]).mean()
        self.start_MJD = \
                np.array([np.array(self.epochs).ravel()[isub].in_days()
                    for isub in range(self.nsub)]).min()
        self.okisub = okisub
        self.freqs = freqs
        self.bw = self.hifreq - self.lofreq
        self.okichan = okichan
        self.weights = weights
        tot_weight = [weights[iarch].sum(axis=0) for iarch in
                range(self.nfile)]
        tot_freq = [tot_weight[iarch] * freqs[iarch] for iarch in
                range(self.nfile)]
        self.weight = []
        [self.weight.extend(tot_weight[iarch]) for iarch in range(self.nfile)]
        self.weight = np.array(self.weight).sum()
        self.nu0 = []
        [self.nu0.extend(tot_freq[iarch]) for iarch in range(self.nfile)]
        self.nu0 = np.array(self.nu0).sum() / self.weight

    def align_data(self):
        """
        """
        if self.niter:
            self.itern = 0
            iterator = self.alignment_iteration(self.quiet)
            print "Aligning profiles..."
        while(self.niter):
            self.itern += 1
            iterator.next()
            self.niter -= 1

    def alignment_iteration(self, quiet=False):
        """
        """
        while(1):
            if not quiet: print "Iteration %d/%d..."%(self.itern,
                    self.itern+self.niter-1)
            new_model = np.zeros([self.npol, self.nbin])
            for iarch in xrange(self.nfile):
                okisub = self.okisub[iarch]
                okichan = self.okichan[iarch]
                if self.state is "Coherence":
                    #For state "Coherence", total intensity I = AA + BB, which
                    #are the first two pol'ns.  cf. van Stratten, Demorest, &
                    #Oslowski (2012)
                    data = self.all_data[iarch][:, 0] + \
                            self.all_data[iarch][:, 1]
                else:
                    data = self.all_data[iarch][:, 0]
                data = np.take(data, okisub, axis=0)
                data = np.take(data, okichan, axis=1)
                #Which to do first?
                if self.talign:
                    profs = data.mean(axis=1)
                    for iprof in xrange(len(profs)):
                        prof = profs[iprof]
                        phase = fit_phase_shift(prof, self.model).phase
                        phase %= 1
                        if phase > 0.5: phase -= 1.0
                        self.all_data[iarch][okisub[iprof]] = rotate_data(
                                self.all_data[iarch][okisub[iprof]],
                                phase)
                        #Update 'data'
                        data[iprof] = rotate_data(data[iprof], phase)
                if self.falign:
                    profs = data.mean(axis=0)
                    for iprof in xrange(len(profs)):
                        prof = profs[iprof]
                        phase = fit_phase_shift(prof, self.model).phase
                        phase %= 1
                        if phase > 0.5: phase -= 1.0
                        self.all_data[iarch][:, :, okichan[iprof]] = \
                                rotate_data(self.all_data[iarch][:, :,
                                    okichan[iprof]], phase)
                if self.align_all:
                    print "This seems broken..."
                    profs = np.copy(data)
                    for isub in xrange(len(profs)):
                        for ichan in xrange(len(profs[0])):
                            prof = profs[isub, ichan]
                            phase = fit_phase_shift(prof, self.model).phase
                            phase %= 1
                            if phase > 0.5: phase -= 1.0
                            self.all_data[iarch][okisub[isub], :,
                                    okichan[ichan]] = rotate_data(
                                            self.all_data[iarch][okisub[isub],
                                                :, okichan[ichan]], phase)
                new_model += self.all_data[iarch].mean(axis=2).mean(axis=0)
            self.model = new_model / self.nfile
            if self.state is "Coherence":
                self.model = self.model[0] + self.model[1]
            else:
                self.model = self.model[0]
            yield

    def make_portrait(self):
        """
        """
        freqs = []
        [freqs.extend(self.freqs[iarch]) for iarch in range(self.nfile)]
        freqs = np.unique(np.array(freqs))
        nchan = len(freqs)
        nbin = self.nbin
        weights = np.zeros(nchan)
        port = np.zeros([self.npol, nchan, nbin])
        for iarch in xrange(self.nfile):
            data = self.all_data[iarch].sum(axis=0)
            data_weights = self.weights[iarch].sum(axis=0)
            for ichan in xrange(len(self.freqs[iarch])):
                freq = self.freqs[iarch][ichan]
#                port[:, list(freqs).index(freq)] += data[:, ichan]
                port[:, list(freqs).index(freq)] += data[:, ichan] * \
                        data_weights[ichan]
                weights[list(freqs).index(freq)] += data_weights[ichan]
        mask = np.einsum('i,j', weights, np.ones(nbin))
        port = np.where(mask > 0, port / mask, port)
        #port = np.where(mask > 0, port / mask, np.zeros(port.shape))
        self.portrait = port
        self.portrait_freqs = freqs
        self.portrait_weights = weights

    def make_profile(self):
        """
        """
        nbin = self.nbin
        weight = 0.0
        prof = np.zeros([self.npol, nbin])
        for iarch in xrange(self.nfile):
            data = self.all_data[iarch].sum(axis=0).sum(axis=1)
            data_weight = self.weights[iarch].sum(axis=0).sum(axis=0)
            prof += data * data_weight
            weight += data_weight
        prof /= weight
        self.profile = prof
        self.profile_weight = weight

    def write_data(self, data, freqs, nu0, bw, tsub, start_MJD, weights,
            ephemeris=None, outfile=None, dedispersed=True, quiet=False):
        """
        Data needs to have shape nsub, npol, nchan, nbin
        """
        rm = False
        if ephemeris is None:
            ephemeris = "ppalign.par"
            datafile = self.datafiles[0]
            cmd = "vap -E %s > %s"%(datafile, ephemeris)
            import os
            os.system(cmd)
            rm_cmd = "rm %s"%ephemeris
            rm = True
        if outfile is None:
            outfile = "new_archive.ppalign.fits"
        if not quiet:
            print "Writing archive %s..."%outfile
        write_archive(data=data, ephemeris=ephemeris, freqs=freqs, nu0=nu0,
                bw=bw, outfile=outfile, tsub=tsub, start_MJD=start_MJD,
                weights=weights, dedispersed=dedispersed, state=self.state,
                obs=self.obs, quiet=quiet)
        if rm: os.system(rm_cmd)

if __name__ == "__main__":

    from optparse import OptionParser

    usage = "usage: %prog [Options]"
    parser = OptionParser(usage)
    #parser.add_option("-h", "--help",
    #                  action="store_true", dest="help", default=False,
    #                  help="Show this help message and exit.")
    parser.add_option("-d", "--datafile",
                      action="store", metavar="archive", dest="datafile",
                      help="PSRCHIVE archive to align.")
    parser.add_option("-M", "--metafile",
                      action="store", metavar="metafile", dest="metafile",
                      help="List of archive filenames in metafile to be aligned.")
    parser.add_option("-t", "--template",
                      action="store", metavar="template", dest="templatefile",
                      default=None,
                      help="PSRCHIVE archive containing initial template. [default=None]")
    parser.add_option("-g", "--gauss",
                      action="store", metavar="wid", dest="gauss",
                      default=0.0,
                      help="Single gaussian FWHM width for initial template. [default=None]")
    parser.add_option("-E", "--ephemeris",
                      action="store", metavar="ephemeris", dest="ephemeris",
                      default=None,
                      help="Ephemeris file to be installed in output archive(s). Danger. [default=Ephemeris stored in first file in metafile]")
    parser.add_option("--niter",
                      action="store", metavar="int", dest="niter", default=3,
                      help="Number of iterations to loop over archives. [default=3]")
    parser.add_option("--pscrunch",
                      action="store_true", dest="pscrunch", default=False,
                      help="pscrunch archives before aligning. [default=False]")
    parser.add_option("--talign",
                      action="store_true", dest="talign", default=False,
                      help="Align archives over subint, averaging over frequencies. Can be used with falign.  [deafult=False]")
    parser.add_option("--falign",
                      action="store_true", dest="falign", default=False,
                      help="Align archives over frequency, averaging over subintegrations. Can be used with talign.  [deafult=False]")
    parser.add_option("--align_all",
                      action="store_true", dest="align_all", default=False,
                      help="Align profiles individually across subints and frequencies.  Overrides talign and falign.  [deafult=False]")
    parser.add_option("--port",
                      action="store", metavar="filename",
                      dest="port_name", default=None,
                      help="Output a concatenated frequency-phase portrait of the aligned, averaged data with name filename.")
    parser.add_option("--prof",
                      action="store", metavar="filename",
                      dest="prof_name", default=None,
                      help="Output a profile of the aligned, averaged data with name filename.")
    parser.add_option("--ext",
                      action="store", metavar="ext",
                      dest="ext", default=None,
                      help="Output the aligned data file-by-file with file extention ext.")
    parser.add_option("--verbose",
                      action="store_false", dest="quiet", default=True,
                      help="More to stdout.")
    (options, args) = parser.parse_args()

    if (options.datafile is None and options.metafile is None) or (
            options.templatefile is None and options.gauss is None) or (
                    options.port_name is None and options.prof_name is None \
                            and options.ext is None):
        print "\nppalign.py -- arbitrarily align profiles\n"
        parser.print_help()
        print ""
        parser.exit()

    datafile = options.datafile
    metafile = options.metafile
    templatefile = options.templatefile
    gauss = float(options.gauss)
    ephemeris = options.ephemeris
    niter = int(options.niter)
    pscrunch = options.pscrunch
    talign = options.talign
    falign = options.falign
    align_all = options.align_all
    prof_name = options.prof_name
    port_name = options.port_name
    ext = options.ext
    quiet = options.quiet

    if metafile is None:
        datafiles = datafile
    else:
        datafiles = metafile

    ad = AlignData(datafiles=datafiles, templatefile=templatefile, gauss=gauss,
            niter=niter, pscrunch=pscrunch, talign=talign, falign=falign,
            align_all=align_all, quiet=quiet)
    ad.align_data()

    if port_name is not None:
        ad.make_portrait()
        ad.write_data(np.array([ad.portrait]), ad.portrait_freqs, ad.nu0, None,
                ad.length, pr.MJD(ad.start_MJD),
                np.array([ad.portrait_weights]), ephemeris, port_name,
                dedispersed=True, quiet=quiet)

    if prof_name is not None:
        ad.make_profile()
        ad.write_data(np.einsum('ikjl', np.array([[ad.profile]])),
                np.array([ad.nu0]), ad.nu0, ad.bw, ad.length,
                pr.MJD(ad.start_MJD), np.array([[ad.profile_weight]]),
                ephemeris, prof_name, dedispersed=True, quiet=quiet)

    if ext is not None:
        for iarch in range(ad.nfile):
            data = ad.all_data[iarch]
            nsub, npol, nchan, nbin = data.shape
            outfile = ad.datafiles[iarch] + "." + ext
            arch = load_data(ad.datafiles[iarch]).arch
            #NEED PSCRUNCH OPTION IN HERE
            #CORRUPT DATA??
            arch.set_dedispersed(True)
            arch.dedisperse()
            isub = 0
            for subint in arch:
                for ipol in xrange(npol):
                    for ichan in xrange(nchan):
                        prof = subint.get_Profile(ipol, ichan)
                        prof.get_amps()[:] = data[isub, ipol, ichan]
                isub += 1
            arch.unload(outfile)
            if not quiet: print "\nUnloaded %s.\n"%outfile
