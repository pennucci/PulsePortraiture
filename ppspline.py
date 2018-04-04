#!/usr/bin/env python

############
# ppspline #
############

#ppspline is a command-line program to make a frequency-parameterized model of
#    wideband profile evolution.  The parameterization is good within the range
#    of data frequencies, provided there are not huge gaps in frequency.  For
#    an input nchan x nbin average portrait of aligned profiles, the model is a
#    B-spline representation of the curve traced out by the nchan profile
#    amplitude vectors in an nbin vector space.  Since the profile shapes are
#    highly correlated, the B-spline representation can be reduced to << nbin 
#    dimensions, and is limited to ten dimensions.  Therefore, the profile
#    variability is decomposed using PCA and a small number of eigenprofiles
#    that encompass most of the profile evolution are selected.  The input data
#    should be high S/N, averaged, and "aligned" (e.g. output from
#    ppalign.py).  Pre-normalization of the input is encouraged (specifically
#    using normalize_portrait('prof')), as is using smooth=True in the
#    make_model function.

#Written by Timothy T. Pennucci (TTP; tim.pennucci@nanograv.org).

from pplib import *

class DataPortrait(DataPortrait):
    """
    DataPortrait is a class that contains the data to which a model is fit.

    This class adds methods and attributes to the parent class specific to
        modeling profile evolution with a B-spline curve.
    """

    def make_spline_model(self, ncomp=None, smooth=True, k=3, sfac=1.0,
            nmax=None, model_name=None, quiet=False):
        """
        Make a model based on PCA and B-spline interpolation.

        ncomp is the number of PCA components to use in the B-spline
            parameterization; ncomp <= 10.  If None, ncomp is the largest
            number of consecutive eigenvectors that, after being smoothed, have
            a non-zero autocorrelation.  ncomp=0 will return a portrait with
            just the mean profile.
        smooth=True will smooth the eigenvectors and mean profile using
            wavelet_smooth and a reduced chi-squared figure-of-merit.
        k is the polynomial degree of the spline; cubic splines (k=3)
            recommended; 1 <= k <= 5.  NB: polynomial order = degree + 1.
        sfac is a multiplicative smoothing factor passed to si.splprep; greater
            values result in more smoothing.  sfac=0 will make an interpolating
            model anchored on the input data profiles.
        nmax is the maximum number of breakpoints (unique knots) to allow.  If
            provided, this may override sfac and enforce smoothing based on
            nmax breakpoints.  That is, in the case the fit returns n > nmax
            breakpoints, it will refit using maximum nmax breakpoints,
            irrespective of the other smoothing condition.  To convert from a
            maximum desired number of B-splines, subtract k-1.  nmax should be
            >= 2.
        model_name is the name of the model; defaults to self.datafile +
            '.spl'
        quiet=True suppresses output.
        """

        #Definitions
        port = self.portx
        pca_weights = self.SNRsxs / np.sum(self.SNRsxs)
        mean_prof = (port.T * pca_weights).T.sum(axis=0) / pca_weights.sum()
        freqs = self.freqsxs[0]
        nu_lo = freqs.min()
        nu_hi = freqs.max()
        #Do principal component analysis
        ncomp, reconst_port, eigvec, eigval = pca(port, mean_prof, pca_weights,
                ncomp=ncomp, quiet=quiet)
        if ncomp > 10: ncomp = 10
        if ncomp == 0: #Will make model with constant average port
            eigval = np.zeros(len(eigval))
            eigvec = np.zeros(eigvec.shape)
            ncomp = 1

        if smooth:
            smooth_mean_prof = smart_smooth(mean_prof)
            smooth_eigvec = np.copy(eigvec)
            smooth_eigvec[:,:ncomp] = smart_smooth(smooth_eigvec.T[:ncomp]).T
            smooth_eigvec[:,ncomp:] = np.zeros(smooth_eigvec[:,ncomp:].shape)

        delta_port = port - mean_prof #Or use smooth_mean_prof?

        if smooth:
            reconst_port = np.dot(smooth_eigvec[:,:ncomp],
                    np.dot(smooth_eigvec[:,:ncomp].T, delta_port.T)).T \
                            + smooth_mean_prof
            #Find the projections of the profiles onto the basis components
            proj_port = np.dot(smooth_eigvec[:,:ncomp].T, delta_port.T).T
        else:
            reconst_port = np.dot(eigvec[:,:ncomp], np.dot(eigvec[:,:ncomp].T,
                delta_port.T)).T + mean_prof
            #Find the projections of the profiles onto the basis components
            proj_port = np.dot(eigvec[:,:ncomp].T, delta_port.T).T

        spl_weights = pca_weights
        s = sfac
        if self.bw < 0: flip = -1   #u in si.splprep has to be increasing...
        else: flip = 1
        #Find the B-spline curve traced by the projected vectors, parameterized
        #by frequency
        (tck,u), fp, ier, msg = si.splprep(proj_port[::flip].T,
                w=spl_weights[::flip], u=freqs[::flip], ub=nu_lo, ue=nu_hi,
                k=k, task=0, s=s, t=None, full_output=1, nest=None, per=0,
                quiet=int(quiet))
        if nmax is not None and len(np.unique(tck[0])) > nmax:
            if nmax < 2:
                print "nmax needs to be >= 2; setting nmax = 2..."
                nmax = 2
            if nmax == 2: s = np.inf
            (tck,u), fp, ier, msg = si.splprep(proj_port[::flip].T,
                    w=spl_weights[::flip], u=freqs[::flip], ub=nu_lo, ue=nu_hi,
                    k=k, task=0, s=s, t=None, full_output=1, nest=nmax+(k*2),
                    per=0, quiet=int(quiet))

        if ier > 1: #Will also catch when ier == "unknown"
            print "Something went wrong in si.splprep for %s:\n%s"%(
                    self.source, msg)

        #Build model
        if smooth:
            modelx = gen_spline_portrait(smooth_mean_prof, freqs,
                    smooth_eigvec[:,:ncomp], tck)
            model = gen_spline_portrait(smooth_mean_prof, self.freqs[0],
                    smooth_eigvec[:,:ncomp], tck)
        else:
            modelx = gen_spline_portrait(mean_prof, freqs, eigvec[:,:ncomp],
                    tck)
            model = gen_spline_portrait(mean_prof, self.freqs[0],
                    eigvec[:,:ncomp], tck)

        #Assign new attributes
        self.ncomp = ncomp
        self.eigvec = eigvec
        self.eigval = eigval
        self.mean_prof = mean_prof
        if smooth:
            self.smooth_mean_prof = smooth_mean_prof
            self.smooth_eigvec = smooth_eigvec
        self.proj_port = proj_port
        #tck contains the knot locations t, B-spline coefficients c, and
        #polynomial degree k -- end knots will have multiplicity k+1, interior
        #breakpoints will have multiplicity 1 for maximum continuity.  The
        #number of B-splines will be n = l + k, where l is the number of
        #intervals.  l = number of breakpoints - 1 = number of unique knots - 1
        # = len(tck[0]) - 2*tck[2] - 1.
        self.tck, self.u, self.fp, self.ier, self.msg = tck, u, fp, ier, msg
        if model_name is None: self.model_name = self.datafile + '.spl'
        else: self.model_name = model_name
        self.model = model
        self.modelx = modelx
        self.model_masked = self.model * self.masks[0,0]

        if not quiet:
            if proj_port.sum():
                print "B-spline interpolation model %s uses %d basis profile components and %d breakpoints."%(self.model_name, ncomp,
                        len(np.unique(self.tck[0])))
            else:
                print "B-spline interpolation model %s uses 0 basis profile components; it returns the average profile."%(self.model_name)

    def write_model(self, outfile, quiet=False):
        """
        Write the output (pickle file) model to outfile.
        """
        of = open(outfile, "wb")
        if hasattr(self, "smooth_eigvec"):
            pickle.dump([self.model_name, self.source, self.datafile,
                self.smooth_mean_prof, self.smooth_eigvec[:,:self.ncomp],
                self.tck], of)
        else:
            pickle.dump([self.model_name, self.source, self.datafile,
                self.mean_prof, self.eigvec[:,:self.ncomp], self.tck], of)
        of.close()
        if not quiet:
            print "Wrote modelfile %s."%outfile


if __name__ == "__main__":

    from optparse import OptionParser

    usage = "Usage: %prog -d <datafile> [options]"
    parser = OptionParser(usage)
    #parser.add_option("-h", "--help",
    #                  action="store_true", dest="help", default=False,
    #                  help="Show this help message and exit.")
    parser.add_option("-d", "--datafile",
                      action="store", metavar="archive", dest="datafile",
                      help="PSRCHIVE archive from which to make model, or a metafile listing multiple archives (i.e., from different bands).  If providing a metafile, the achives must already be aligned.")
    parser.add_option("-o", "--modelfile",
                      action="store", metavar="modelfile", dest="modelfile",
                      help="Name for output model (pickle) file. [default=datafile.spl].")
    parser.add_option("-l", "--model_name",
                      action="store", metavar="model_name", dest="model_name",
                      default=None,
                      help="Optional name for model [default=datafile.spl].")
    parser.add_option("-a", "--archive",
                      action="store", metavar="archive", dest="archive",
                      default=None,
                      help="Name for optional output PSRCHIVE archive.  Will work only if the input is a single archive.")
    parser.add_option("-N", "--norm",
                      action="store", metavar="normalization", dest="norm",
                      default="mean",
                      help="Normalize the input data by channel ('None', 'mean' [default], 'max' (not recommended), 'rms' (off-pulse noise), 'prof' (mean profile flux), or 'abs' (sqrt{vector modulus})).")
    parser.add_option("-s", "--smooth",
                      action="store_true", metavar="smooth", dest="smooth",
                      default=False,
                      help="Smooth the eigenvectors and mean profile using default wavelet_smooth options and smart_smooth.")
    parser.add_option("-n", "--ncomp",
                      action="store", metavar="ncomp", dest="ncomp",
                      default=None,
                      help="Number of principal components to use in PCA reconstruction of the data.  ncomp is limited to a maximum of 10 by the B-spline representation in scipy.interpolate.  The default automatically finds ncomp significant, smoothed eigenvectors.")
    parser.add_option("-k", "--degree",
                      action="store", metavar="degree", dest="k", default=3,
                      help="Degree of the spline.  Cubic splines (k=3) are recommended [default]. 1 <= k <=5.")
    parser.add_option("-t", "--knots",
                      action="store", metavar="max_knots", dest="nmax",
                      default=None,
                      help="The maximum number of unique knots.  This functions esentially as an ignorant smoothing condition in case the default settings return a fit with more than max_knots number of unique knots in the spline model.  e.g., 10 unique knots are more than usually necessary.")
    parser.add_option("--quiet",
                      action="store_true", dest="quiet", default=False,
                      help="Suppresses output.")

    (options, args) = parser.parse_args()

    if (options.datafile is None):
        print "\nppspline.py - make a pulse portrait model using PCA & B-spline interpolation\n"
        parser.print_help()
        print ""
        parser.exit()

    datafile = options.datafile
    modelfile = options.modelfile
    model_name = options.model_name
    archive = options.archive
    norm = options.norm
    smooth = options.smooth
    if options.ncomp is not None: ncomp = int(options.ncomp)
    else: ncomp = None
    k = int(options.k)
    if options.nmax is not None: nmax = int(options.nmax)
    else: nmax = None
    quiet = options.quiet

    dp = DataPortrait(datafile)

    if norm in ("mean", "max", "prof", "rms", "abs"):
        dp.normalize_portrait(norm)

    dp.make_spline_model(ncomp=ncomp, smooth=smooth, k=k, sfac=1.0, nmax=nmax,
            model_name=model_name, quiet=quiet)

    if modelfile is None: modelfile = datafile + ".spl"
    dp.write_model(modelfile, quiet=quiet)

    if archive is not None and len(dp.datafiles) == 1:
        dp.write_archive(archive, quiet=quiet)
