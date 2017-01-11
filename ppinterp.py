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

from ppgauss import DataPortrait
from pplib import *


def make_interp_model(dp, norm="mean", smooth=False, ncomp=10, k=3,
        modelfile=None, modelname=None, outfile=None, quiet=False):
    """
    Make a model based on PCA and B-spline parameterization of a ncomp curve.

    Important quantities are added as attributes of dp.

    dp is an object from the class DataPortrait.
    norm is the portrait normalization method (None, 'mean', 'max', or 'rms').
    smooth=True will use the default settings from wavelet_smooth to smooth.
    ncomp is the number of PCA components to use in the B-spline
        parameterization; ncomp <= 10 (recommended).
    k is the degree of the spline; cubic splines (k=3) recommended; 1 <= k <=5.
    modelfile is the name of the written pickle file that will contain the
        model.
    modelname is the name of the model; defaults to dp.datafile + '.interp'
    outfile is the name of a written PSRCHIVE fits file to be constructed from 
        the model at the same (non-zero weighted channel) frequencies.
    quiet=True suppresses output.
    """
    if norm in ("mean", "max", "rms"): dp.normalize_portrait(norm)
    if smooth: dp.smooth_portrait()

    port = dp.portx
    #mean_prof = dp.prof #bad choice
    mean_prof = np.average(port, axis=0) #works, simple
    #mean_prof = np.average(port, axis=0, weights=dp.noise_stdsxs**-2) #obvious
    #mean_prof = np.average(port, axis=0, weights=dp.SNRsxs) #not sure
    freqs = dp.freqsxs[0]
    nu_lo = freqs.min()
    nu_hi = freqs.max()
    #weights = dp.noise_stdsxs**-2
    weights = dp.SNRsxs
    reconst_port, eigvec, eigval = pca(port, mean_prof, ncomp=ncomp,
            quiet=quiet)
    delta_port = port - mean_prof
    proj_port = np.dot(eigvec[:,:ncomp].T, delta_port.T).T
    if dp.bw < 0: flip = -1
    else: flip = 1

    (tck,u), fp, ier, msg = si.splprep(proj_port[::flip].T,
            w=weights[::flip]**0.5, u=freqs[::flip], ub=nu_lo, ue=nu_hi, k=k,
            task=0, s=None, t=None, full_output=1, nest=None, per=0,
            quiet=int(quiet))
    if ier > 0:
        print "Something went wrong in si.splprep:\n%s"%msg
    model_port = build_interp_portrait(mean_prof, freqs, eigvec[:,:ncomp], tck)
    
    if modelfile is not None:
        of = open(modelfile, "wb")
        if modelname is None: modelname = dp.datafile + '.interp'
        pickle.dump([modelname, dp.source, datafile, mean_prof,
            eigvec[:,:ncomp], tck], of)
        of.close()

    if outfile is not None:
        new_data = np.zeros(dp.arch.get_data().shape)
        ichanx = 0
        for ichan,weight in enumerate(dp.weights[0]):
            if weight > 0:
                new_data[0,0,ichan] = model_port[ichanx]
                ichanx += 1
        unload_new_archive(new_data, dp.arch, outfile, DM=0.0, dmc=0,
                weights=None, quiet=quiet)
    
    dp.modelname = modelname 
    dp.eigvec = eigvec
    dp.eigval = eigval
    dp.mean_prof = mean_prof
    dp.tck, dp.u, dp.fp, dp.ier, dp.msg = tck, u, fp, ier, msg
    dp.model_port = model_port
        

if __name__ == "__main__":

    from optparse import OptionParser

    usage = "Usage: %prog -d <datafile> -o <outfile> [options]"
    parser = OptionParser(usage)
    #parser.add_option("-h", "--help",
    #                  action="store_true", dest="help", default=False,
    #                  help="Show this help message and exit.")
    parser.add_option("-d", "--datafile",
                      action="store", metavar="archive", dest="datafile",
                      help="PSRCHIVE archive from which to make model.")
    parser.add_option("-m", "--modelfile",
                      action="store", metavar="modelfile", dest="modelfile",
                      help="Name for output model (pickle) file.")
    parser.add_option("-s", "--modelname",
                      action="store", metavar="modelname", dest="modelname",
                      default=None,
                      help="Optional name for model [default=datafile_interp].")    
    parser.add_option("-o", "--outfile",
                      action="store", metavar="outfile", dest="outfile",
                      default=None,
                      help="Name for optional output PSRCHIVE archive.")
    parser.add_option("-N", "--norm",
                      action="store", metavar="normalization", dest="norm",
                      default="mean",
                      help="Normalize the input data by channel ('None', 'mean' [default], 'max' (not recommended), or 'rms').")
    parser.add_option("-S", "--smooth",
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

    if (options.datafile is None or options.modelfile is None):
        print "\nppinterp.py - make a pulse portrait model using PCA & B-splines\n"
        parser.print_help()
        print ""
        parser.exit()

    datafile = options.datafile
    modelfile = options.modelfile
    modelname = options.modelname
    outfile = options.outfile
    norm = options.norm
    smooth = options.smooth
    ncomp = int(options.ncomp)
    k = int(options.k)
    quiet = options.quiet
    
    dp = DataPortrait(datafile)
    make_interp_model(dp, norm=norm, smooth=smooth, ncomp=ncomp, k=k,
            modelfile=modelfile, modelname=modelname, outfile=outfile,
            quiet=quiet)
