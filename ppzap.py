#!/usr/bin/env python

#######
#ppzap#
#######

#ppzap is a command-line program
#    Full-functionality is obtained when using

#Written by Timothy T. Pennucci (TTP; tim.pennucci@nanograv.org).

from pptoas import *

#percent channels zapped per subint / archive / input

def print_paz_cmds(datafiles, zap_list, modify=True, outfile=None,
        quiet=False):
    """
    """
    if outfile is not None:
        sys.stdout = open(outfile, "a")
    lines = []
    for iarch, datafile in enumerate(datafiles):
        count = 0
        for isub in range(len(zap_list[iarch])):
            count += len(zap_list[iarch][isub])
        if count:
            if modify:
                paz_outfile = datafile
            else:
                ii = datafile[::-1].find(".")
                if ii < 0: paz_outfile = datafile + ".zap"
                else: paz_outfile = datafile[:-ii] + "zap"
                print "paz -e zap %s"%datafile
        for isub, bad_ichans in enumerate(zap_list[iarch]):
            for bad_ichan in bad_ichans:
                print "paz -m -I -z %d -w %d %s"%(bad_ichan, isub, paz_outfile)
    sys.stdout = sys.__stdout__
    if outfile is not None and not quiet:
        print "Wrote %s."%outfile


if __name__ == "__main__":

    from optparse import OptionParser

    usage = "Usage: %prog -d <datafile or metafile> -m <modelfile> [options]"
    parser = OptionParser(usage)
    #parser.add_option("-h", "--help",
    #                  action="store_true", dest="help", default=False,
    #                  help="Show this help message and exit.")
    parser.add_option("-d", "--datafiles",
                      action="store", metavar="archive", dest="datafiles",
                      help="PSRCHIVE archive, or a metafile listing archive filenames, to examine.  \
                              ***NB: Files should NOT be dedispersed!!*** \
                              i.e. vap -c dmc <datafile> should return 0!")
    parser.add_option("-m", "--modelfile",
                      action="store", metavar="model", dest="modelfile",
                      help="Model file from ppgauss.py, ppinterp.py, or PSRCHIVE FITS file that either has same channel frequencies, nchan, & nbin as datafile(s), or is a single profile (nchan = 1, with the same nbin) to be interpreted as a constant template.")
    parser.add_option("-t", "--threshold",
                      metavar="red_chi2", action="store", dest="threshold",
                      default=1.5,
                      help="Set a reduced chi-squared threshold for flagging bad channels [default=1.5].")
    parser.add_option("-o", "--outfile",
                      action="store", metavar="outfile", dest="outfile",
                      default=None,
                      help="Name of output paz command file. Will append. [default=stdout]")
    parser.add_option("--modify",
                      action="store_true", dest="modify", default=False,
                      help="paz commands will modify original datafiles.")
    parser.add_option("--show",
                      action="store_true", dest="show", default=False,
                      help="Show zapped portrait for each subint with proposed channels to zap.")
    parser.add_option("--hist",
                      action="store_true", dest="hist", default=False,
                      help="Plot histogram of channel reduced chi-squared values.")
    parser.add_option("--quiet",
                      action="store_true", dest="quiet", default=False,
                      help="Suppress output.")

    (options, args) = parser.parse_args()

    if (options.datafiles is None or options.modelfile is None):
        print "\nppzap.py - Identify bad channels to zap.\n"
        parser.print_help()
        print ""
        parser.exit()

    datafiles = options.datafiles
    modelfile = options.modelfile
    threshold = float(options.threshold)
    outfile = options.outfile
    modify = options.modify
    show = options.show
    hist = options.hist
    quiet = options.quiet

    gt = GetTOAs(datafiles=datafiles, modelfile=modelfile, quiet=True)
    gt.get_TOAs(quiet=True)
    gt.get_channel_red_chi2s(threshold=threshold, show=show)
    print_paz_cmds(gt.datafiles, gt.zap_channels, modify=modify,
            outfile=outfile, quiet=quiet)

    nchan = 0
    nzap = 0
    for iarch in range(len(gt.datafiles)):
        for isub in range(len(gt.channel_red_chi2s[iarch])):
            nchan += len(gt.channel_red_chi2s[iarch][isub])
            nzap += len(gt.zap_channels[iarch][isub])

    if not quiet:
        print "ppzap.py found %d bad channels out of a total %d channels fit (=%.2f%%)."%(nzap, nchan, 100*float(nzap)/nchan)

    if hist:
        red_chi2s = []
        for iarch in range(len(gt.datafiles)):
            for isub in range(len(gt.channel_red_chi2s[iarch])):
                red_chi2s.extend(gt.channel_red_chi2s[iarch][isub])
        red_chi2s = np.nan_to_num(np.array(red_chi2s))
        plt.hist(red_chi2s, bins=min(50, len(red_chi2s)), log=True)
        ymin, ymax = plt.ylim()
        plt.vlines(threshold, ymin, ymax, linestyles='dashed')
        plt.ylim(ymin, ymax)
        plt.xlabel(r"Reduced $\chi^2$")
        plt.ylabel("#")
        plt.title("%s\n"%datafiles + r"%d / %d channels w/ $\chi^2_{red}$ > %.1f"%(nzap, nchan, threshold))
        plt.savefig(datafiles+"_ppzap_hist.png")
