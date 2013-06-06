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

class AlignData:
    """
    """
    def __init__(self, metafile, outfile="ppalign.fits", template=None,
            gauss=None, ephemeris=None, niter=1, tscrunch=True, pscrunch=True,
            align=False, quiet=False):
        self.metafile = metafile
        self.outfile = outfile
        self.template = template
        self.gauss = gauss
        self.ephemeris = ephemeris
        self.niter = niter
        self.tscrunch = tscrunch
        if not self.tscrunch:
            self.tscrunch = True
            print "Sorry, but we have to tscrunch for now..."
        self.pscrunch = pscrunch
        if not self.pscrunch:
            print "Note pol'n state of files!"
            self.npol = 4
        else:
            self.npol = 1
        self.quiet = quiet
        if align: self.align_data()

    def align_data(self):
        """
        Currently should only employ T-scrunched archs.
        """
        self.nchan = 0
        self.lofreq = np.inf
        self.hifreq = 0.0
        self.P = 0.0
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
                    tscrunch=self.tscrunch, pscrunch=self.pscrunch,
                    rm_baseline=True, flux_prof=True, quiet=self.quiet)
            if ifile == 0:
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
                    if self.template is not None:
                        self.model = load_data(self.template, dedisperse=True,
                                dededisperse=False, tscrunch=True,
                                pscrunch=True, rm_baseline=True,
                                flux_prof=False, norm_weights=False,
                                quiet=self.quiet).prof
                    else:
                        self.model = data.prof
            self.nchan += data.nchan
            lf = data.freqs.min() - (abs(data.bw) / (2*data.nchan))
            if lf < self.lofreq:
                self.lofreq = lf
            hf = data.freqs.max() + (abs(data.bw) / (2*data.nchan))
            if hf > self.hifreq:
                self.hifreq = hf
            self.P += data.Ps[0]
            #subint = (data.masks * data.subints)[0]
            subint = data.masks * data.subints
            for ichan in xrange(data.nchan):
            #    freqs.append(data.freqs[ichan])
                profs.append(subint[:, :, ichan])
            #    weights.append(data.weights[0, ichan])
            #    fluxes.append(data.flux_prof[ichan])
            freqs += list(data.freqs)
            weights += list(data.weights[0])
            fluxes += list(data.flux_prof)
        self.P /= len(datafiles)    #Will be used only as a toy period
        iis = np.lexsort((fluxes, freqs))
        self.freqs = np.array([freqs[ii] for ii in iis])
        self.profs = np.array([profs[ii] for ii in iis])
        self.weights = np.array([weights[ii] for ii in iis])
        self.fluxes = np.array([fluxes[ii] for ii in iis])
        self.port = np.copy(self.profs)
        self.portx = np.compress(self.weights, self.port, axis=0)   #TEST
        self.freqsx = np.compress(self.weights, self.freqs)
        self.nchanx = len(self.freqsx)
        self.bw = self.hifreq - self.lofreq
        self.okinds = np.compress(self.weights, range(len(self.port)))
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
            for ichan in xrange(len(self.portx)):
                profs = self.portx[:, :, ichan]
                if self.state is "Coherence":
                    prof = self.portx[:, 0, ichan] + self.portx[:, 1, ichan]
                    prof = prof.mean(axis=0)
                else:
                    prof = self.portx[:, 0, ichan].mean(axis=0)
                phase = fit_phase_shift(prof, self.model).phase
                phase %= 1
                if phase > 0.5: phase -= 1.0
                self.portx[:, :, ichan] = rotate_portrait(profs, phase)
                self.port[:, :, self.okinds[ichan]] = self.portx[:, :, ichan]
            #This next statement should really go above somewhere
            self.model = self.portx.mean(axis=0).mean(axis=0).mean(axis=0)
            yield

    def write_portrait(self, outfile=None, ephemeris=None, quiet=False):
        """
        """
        if outfile is None:
            outfile = self.outfile
        if ephemeris is None:
            ephemeris = self.ephemeris
        if ephemeris is None:
            datafile = self.datafiles[0]
            cmd = "vap -E %s >> ppalign.par"%datafile
            import os
            os.system(cmd)
        #The following is copied from make_fake_pulsar; this needs to be
        #combined...
        write_archive(self.port, ephemeris, self.freqs, nu0=None, bw=None,
        outfile=outfile, tsub=None, start_MJD=None, weights=self.weights,
        dedispersed=True, state=self.state, obs=self.obs, quiet=quiet)

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
                      action="store", metavar="template", dest="template",
                      default=None,
                      help="PSRCHIVE archive containing initial template. [default=None]")
    parser.add_option("-g", "--gauss",
                      action="store", metavar="wid", dest="gauss",
                      default=None,
                      help="Single gaussian FWHM width for initial template. [default=None]")
    parser.add_option("-e", "--ephemeris",
                      action="store", metavar="ephemeris", dest="ephemeris",
                      default=None,
                      help="Ephemeris file to be installed. Danger. [default=Ephemeris stored in first file in metafile]")
    parser.add_option("--niter",
                      action="store", metavar="int", dest="niter", default=1,
                      help="Number of iterations to loop over archives. [default=1]")
    parser.add_option("--no_tscrunch",
                      action="store_false", dest="tscrunch", default=True,
                      help="Do not tscrunch archives before aligning. [default=tscrunch] (currently does not work otherwise)")
    parser.add_option("--no_pscrunch",
                      action="store_false", dest="pscrunch", default=True,
                      help="Do not pscrunch archives before aligning. [default=pscrunch]")
    parser.add_option("--verbose",
                      action="store_false", dest="quiet", default=True,
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
    ephemeris = options.ephemeris
    niter = int(options.niter)
    tscrunch = options.tscrunch
    pscrunch = options.pscrunch
    quiet = options.quiet

    ad = AlignData(metafile=metafile, outfile=outfile, template=template,
            gauss=gauss, ephemeris=ephemeris, niter=niter, tscrunch=tscrunch,
            pscrunch=pscrunch, align=True, quiet=quiet)
    ad.write_portrait()
