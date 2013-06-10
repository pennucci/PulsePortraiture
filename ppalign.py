#!/usr/bin/env python

from pplib import *

#need to average same freqs...not just Tscrunch
#gaps in plotting
#which band to put first, hand rotate
#to use beginning template from data or gauss?
#stopping/convergence criteria
#smoothing
#polarization info!
#documentation
#Stokes versus coherence
#independent alignment of different pols?  alignment via something other
#than phase shift? PPA?
#write separate arbitrarily aligned files or one big one?
#-F -T -P options for output standard profile?
#SNR cutoff!
#fiducial phase option
#all data same pol state, or psrunch
#use norm_weights?
#proper t- f- p- scrunch calculations?

class AlignData:
    """
    """
    def __init__(self, datafiles, templatefile=None, gauss=None, niter=1,
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
        self.P = 0.0
        self.nchan = 0
        self.lofreq = np.inf
        self.hifreq = 0.0
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
                if self.gauss is not None:
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
            self.P += data.Ps.mean()
            epochs.append(data.epochs)
            okisub.append(data.okisub)
            self.nchan += data.nchan
            lf = data.freqs.min() - (abs(data.bw) / (2*data.nchan))
            if lf < self.lofreq:
                self.lofreq = lf
            hf = data.freqs.max() + (abs(data.bw) / (2*data.nchan))
            if hf > self.hifreq:
                self.hifreq = hf
            freqs.append(data.freqs)
            okichan.append(data.okichan)
            all_data.append(data.subints)
            weights.append(data.weights)
        self.P /= len(self.datafiles)    #Will be used only as a toy period
        self.all_data = all_data
        self.epochs = epochs
        self.okisub = okisub
        self.freqs = freqs
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

    def write_data(self, data, freqs, nu0, tsub, start_MJD, weights,
            ephemeris=None, outfile=None, dedispersed=False, quiet=False):
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
                bw=None, outfile=outfile, tsub=None, start_MJD=None,
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
#    parser.add_option("-m", "--modify",
#                      action="store_true", dest="modify", default=False,
#                      help="Modify the original files on disk [default=False].")
    parser.add_option("-e", "--ext",
                      action="store", dest="ext", default=None,
                      help="Write new files with this extension.")
    parser.add_option("-o", "--outfile",
                      action="store", metavar="outfile", dest="outfile",
                      default=None,
                      help="Name of single output archive name.")
    parser.add_option("-t", "--template",
                      action="store", metavar="template", dest="templatefile",
                      default=None,
                      help="PSRCHIVE archive containing initial template. [default=None]")
    parser.add_option("-g", "--gauss",
                      action="store", metavar="wid", dest="gauss",
                      default=None,
                      help="Single gaussian FWHM width for initial template. [default=None]")
    parser.add_option("-E", "--ephemeris",
                      action="store", metavar="ephemeris", dest="ephemeris",
                      default=None,
                      help="Ephemeris file to be installed. Danger. [default=Ephemeris stored in first file in metafile]")
    parser.add_option("--niter",
                      action="store", metavar="int", dest="niter", default=1,
                      help="Number of iterations to loop over archives. [default=1]")
    parser.add_option("--no_pscrunch",
                      action="store_false", dest="pscrunch", default=True,
                      help="Do not pscrunch archives before aligning. [default=pscrunch]")
    parser.add_option("--verbose",
                      action="store_false", dest="quiet", default=True,
                      help="More to stdout.")
    (options, args) = parser.parse_args()

    if (options.datafile is None and options.metafile is None) or (
            options.template is None and options.gauss is None):
        print "\nppalign.py -- arbitrarily align profiles\n"
        parser.print_help()
        print ""
        parser.exit()

    metafile = options.metafile
    outfile = options.outfile
    templatefile = options.templatefile
    if options.gauss is not None:
        gauss = float(options.gauss)
    else:
        gauss = None
    ephemeris = options.ephemeris
    niter = int(options.niter)
    pscrunch = options.pscrunch
    quiet = options.quiet

    ad = AlignData(metafile=metafile, outfile=outfile,
            templatefile=templatefile, gauss=gauss, ephemeris=ephemeris,
            niter=niter, pscrunch=pscrunch, quiet=quiet)
    ad.align_data()
    ad.make_portrait()
    #or
    ad.make_profile()
    ad.write_data()
