#!/usr/bin/env python

from pplib import *

#need to average same freqs
#gaps in plotting
#which band to put first, hand rotate
#to use beginning template from data or gauss?
#stopping/convergence criteria
#smoothing
#polarization info!
#documentation

class AlignData:
    """
    """
    def __init__(self, metafile, outfile="ppalign.fits", template=None,
            gauss=None, niter=1, quiet=False):
        self.metafile = metafile
        self.outfile = outfile
        self.template = template
        self.gauss = gauss
        self.niter = niter
        self.quiet = quiet

    def align_data(self):
        """
        Currently should only employ T-scrunched archs.  Needs to be able to
        take
        FT-scrunched template archive.
        """
        self.lofreq = np.inf
        self.hifreq = 0.0
        self.P= 0.0
        datafiles = open(self.metafile, "r").readlines()
        datafiles = [datafiles[ifile][:-1] for ifile in xrange(len(datafiles))]
        freqs = []
        profs = []
        fluxes = []
        weights = []
        print "Reading data from archives..."
        for ifile in xrange(len(datafiles)):
            datafile = datafiles[ifile]
            data = load_data(datafile, dedisperse=True, dededisperse=False,
                    tscrunch=True, pscrunch=True, rm_baseline=True,
                    flux_prof=True, quiet=self.quiet)
            if ifile == 0:
                self.nbin = data.nbin
                self.phases = data.phases
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
                    if self.template is not None:
                        self.model = load_data(self.template, dedisperse=True,
                                dededisperse=False, tscrunch=True,
                                pscrunch=True, rm_baseline=True,
                                flux_prof=False, norm_weights=False,
                                quiet=self.quiet).prof
                    else:
                        self.model = data.prof
            lf = data.freqs.min() - (abs(data.bw) / (2*data.nchan))
            if lf < self.lofreq:
                self.lofreq = lf
            hf = data.freqs.max() + (abs(data.bw) / (2*data.nchan))
            if hf > self.hifreq:
                self.hifreq = hf
            self.P += data.Ps[0] #This can't possibly be smart
            subint = (data.masks * data.subints)[0,0]
            for ichan in xrange(data.nchan):
                freqs.append(data.freqs[ichan])
                profs.append(subint[ichan])
                weights.append(data.weights[0, ichan])
                fluxes.append(data.flux_prof[ichan])
        iis = np.lexsort((fluxes, freqs))
        self.freqs = np.array([freqs[ii] for ii in iis])
        self.profs = np.array([profs[ii] for ii in iis])
        self.weights = np.array([weights[ii] for ii in iis])
        self.fluxes = np.array([fluxes[ii] for ii in iis])
        self.port = np.copy(self.profs)
        self.portx = np.compress(self.weights, self.port, axis=0)
        self.freqsx = np.compress(self.weights, self.freqs)
        self.okinds = np.compress(self.weights, range(len(self.port)))
        if self.niter:
            self.itern = 0
            iterator = self.alignment_iteration(self.quiet)
            print "Aligning profiles..."
        while(self.niter):
            self.itern += 1
            iterator.next()
            self.niter -= 1
        self.data = DataBunch(freqs=self.freqs, freqsx=self.freqsx,
                fluxes=self.fluxes, phases=self.phases, port=self.port,
                portx=self.portx, weights=self.weights)

    def alignment_iteration(self, quiet=False):
        """
        """
        while(1):
            if not quiet: print "Iteration %d/%d..."%(self.itern,
                    self.itern+self.niter-1)
            for ichan in xrange(len(self.portx)):
                prof = self.portx[ichan]
                phase = fit_phase_shift(prof, self.model).phase
                phase %= 1
                if phase > 0.5: phase -= 1.0
                self.portx[ichan] = rotate_portrait([prof], phase)[0]
                self.port[self.okinds[ichan]] = self.portx[ichan]
            self.model = self.portx.mean(axis=0)
            yield

    def write_archive(self, outfile=None):
        """
        """
        if outfile is None:
            outfile = self.outfile
        print "WRITE %s"%outfile

if __name__ == "__main__":

    from optparse import OptionParser

    usage = "usage: %prog [Options]"
    parser = OptionParser(usage)
    #parser.add_option("-h", "--help",
    #                  action="store_true", dest="help", default=False,
    #                  help="Show this help message and exit.")
    parser.add_option("-M", "--metafile",
                      action="store", metavar="metafile", dest="metafile",
                      help="File containing list of archive file names.")
    parser.add_option("-o", "--outfile",
                      action="store", metavar="outfile", dest="outfile",
                      help="Name of output archive name. [default=ppalign.fits]")
    parser.add_option("-t", "--template",
                      default=None,
                      action="store", metavar="template", dest="template",
                      help="PSRCHIVE archive containing initial template. [default=None]")
    parser.add_option("-g", "--gauss",
                      action="store", metavar="wid", dest="gauss",
                      default=None,
                      help="Single gaussian FWHM width for initial template. [default=None]")
    parser.add_option("--niter",
                      action="store", metavar="int", dest="niter", default=1,
                      help="Number of iterations to loop over archives. [default=1]")
    parser.add_option("--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="More to stdout.")
    (options, args) = parser.parse_args()

    if options.metafile is None:
        print "\nppmake.py -- make standard portrait\n"
        parser.print_help()
        print ""
        parser.exit()

    metafile = options.metafile
    outfile = options.outfile
    template = options.template
    if options.gauss is not None:
        gauss = float(options.gauss)
    else:
        gauss = None
    niter = int(options.niter)
    quiet = not options.verbose

    ad = AlignData(metafile=metafile, outfile=outfile, template=template,
            gauss=gauss, niter=niter, quiet=quiet)
    ad.align_data()
    ad.write_archive()
