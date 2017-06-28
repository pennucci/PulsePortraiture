#!/usr/bin/env python

##########
#ppinterp#
##########

#ppinterp is a command-line program to make a parameterized model of wideband
#    profile evolution.  The parameterization is good within the range of data
#    frequencies, provided they are not widely separated frequencies.  The
#    parameterization is a B-spline representation of the curve traced out by
#    the nchan profile amplitude vectors in an nbin vector space.  Since the
#    B-spline representation function is limited to a ten dimensional space,
#    we decompose the profile variability in nbin space using PCA and choose
#    the top ten components encompassing most of the profile evolution.  The
#    input data should be high S/N, averaged, and "aligned" (e.g. output from
#    ppalign.py).  Channel normalization is encouraged, and smoothing can be
#    done either on the input data (option provided) or on the constructed
#    output (noisy) model.

#Written by Timothy T. Pennucci (TTP; pennucci@email.virginia.edu).

from pplib import *

class DataPortrait(DataPortrait):
    """
    DataPortrait is a class that contains the data to which a model is fit.

    This class adds methods and attributes to the parent class specific to
        modeling profile evolution with a B-spline curve.
    """

    def make_interp_model(self, ncomp=None, eigfac=0.99, k=3, sfac=1.5,
            model_name=None, quiet=False):
        """
        Make a model based on PCA and B-spline interpolation.

        ncomp is the number of PCA components to use in the B-spline
            parameterization; ncomp <= 10.  If None, ncomp is the smallest
            number of eigenvectors whose eigenvalues sum to greater than
            eigfac*sum(eigenvalues), or 10, whichever is smaller.
        eigfac determines ncomp if ncomp is None.
        k is the degree of the spline; cubic splines (k=3) recommended;
            1 <= k <= 5.
        sfac is a multiplicative smoothing factor; greater values result in
            more smoothing.  sfac=0 will make an interpolating model anchored
            on the input data profiles.
        model_name is the name of the model; defaults to
            self.datafile + '.interp'
        quiet=True suppresses output.
        """

        #Definitions
        port = self.portx
        mean_prof = np.average(port, axis=0) #Seems to work best
        freqs = self.freqsxs[0]
        nu_lo = freqs.min()
        nu_hi = freqs.max()
        #Do principal component analysis
        ncomp, reconst_port, eigvec, eigval = pca(port, mean_prof, ncomp=ncomp,
                eigfac=eigfac, quiet=quiet)
        delta_port = port - mean_prof
        if ncomp > 10:
            ncomp = 10
            reconst_port = np.dot(eigvec[:,:ncomp], np.dot(eigvec[:,:ncomp].T,
                delta_port.T)).T + mean_prof
        #Find the projections of the profiles onto the basis components
        proj_port = np.dot(eigvec[:,:ncomp].T, delta_port.T).T
        weights = get_noise(proj_port, chans=True)**-1 #See si.splprep docs
        s = len(proj_port) * sfac #Seems to work OK
        if self.bw < 0: flip = -1
        else: flip = 1
        #Find the B-spline curve traced by the projected vectors, parameterized
        #by frequency
        (tck,u), fp, ier, msg = si.splprep(proj_port[::flip].T,
                w=weights[::flip], u=freqs[::flip], ub=nu_lo, ue=nu_hi, k=k,
                task=0, s=s, t=None, full_output=1, nest=None, per=0,
                quiet=int(quiet))
        if ier > 0:
            print "Something went wrong in si.splprep:\n%s"%msg
        if not quiet:
            print "B-spline interpolation model uses %d basis profile components."%ncomp

        #Build model
        modelx = gen_interp_portrait(mean_prof, freqs, eigvec[:,:ncomp], tck)
        model = gen_interp_portrait(mean_prof, self.freqs[0], eigvec[:,:ncomp],
                tck)

        #Assign new attributes
        self.ncomp = ncomp
        self.model_name = model_name
        self.eigvec = eigvec
        self.eigval = eigval
        self.mean_prof = mean_prof
        self.tck, self.u, self.fp, self.ier, self.msg = tck, u, fp, ier, msg
        if model_name is None: self.model_name = self.datafile + '.interp'
        else: self.model_name = model_name
        self.model = model
        self.modelx = modelx
        self.model_masked = self.model * self.masks[0,0]

    def write_model(self, outfile, quiet=False):
        """
        Write the output (pickle file) model to outfile.
        """
        of = open(outfile, "wb")
        pickle.dump([self.model_name, self.source, self.datafile,
            self.mean_prof, self.eigvec[:,:ncomp], self.tck], of)
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
                      help="PSRCHIVE archive from which to make model.")
    parser.add_option("-o", "--modelfile",
                      action="store", metavar="modelfile", dest="modelfile",
                      help="Name for output model (pickle) file. [default=datafile.spl].")
    parser.add_option("-l", "--model_name",
                      action="store", metavar="model_name", dest="model_name",
                      default=None,
                      help="Optional name for model [default=datafile_interp].")
    parser.add_option("-a", "--archive",
                      action="store", metavar="archive", dest="archive",
                      default=None,
                      help="Name for optional output PSRCHIVE archive.")
    parser.add_option("-N", "--norm",
                      action="store", metavar="normalization", dest="norm",
                      default="mean",
                      help="Normalize the input data by channel ('None', 'mean' [default], 'max' (not recommended), 'rms' (off-pulse noise), or 'abs' (sqrt{vector modulus})).")
    parser.add_option("-f", "--filter",
                      action="store_true", metavar="filter", dest="filtre",
                      default=False,
                      help="Pre-filter the data using default low-pass filter function.")
    parser.add_option("-s", "--smooth",
                      action="store_true", metavar="smooth", dest="smooth",
                      default=False,
                      help="Pre-smooth the data using default wavelet_smooth options.")
    parser.add_option("-n", "--ncomp",
                      action="store", metavar="ncomp", dest="ncomp",
                      default=10,
                      help="Number of principal components to use in PCA reconstruction of the data.  ncomp is limited to a maximum of 10 [default] by the B-spline representation in scipy.interpolate.")
    parser.add_option("-k", "--degree",
                      action="store", metavar="degree", dest="k", default=3,
                      help="Degree of the spline.  Cubic splines (k=3) are recommended [default]. 1 <= k <=5.")
    parser.add_option("--quiet",
                      action="store_true", dest="quiet", default=False,
                      help="Suppresses output.")

    (options, args) = parser.parse_args()

    if (options.datafile is None):
        print "\nppinterp.py - make a pulse portrait model using PCA & B-splines\n"
        parser.print_help()
        print ""
        parser.exit()

    datafile = options.datafile
    modelfile = options.modelfile
    model_name = options.model_name
    archive = options.archive
    norm = options.norm
    filtre = options.filtre
    smooth = options.smooth
    ncomp = int(options.ncomp)
    k = int(options.k)
    quiet = options.quiet

    dp = DataPortrait(datafile)

    if norm in ("mean", "max", "rms", "abs"): dp.normalize_portrait(norm)
    if filtre: dp.filter_portrait() #Can take a while
    if smooth: dp.smooth_portrait()

    dp.make_interp_model(ncomp=ncomp, k=k, modelfile=modelfile,
            model_name=model_name, archive=archive, quiet=quiet)

    if modelfile is None: modelfile = datafile + ".spl"
    dp.write_model(modelfile, quiet=quiet)

    if archive is not None: dp.write_archive(archive, quiet=quiet)
