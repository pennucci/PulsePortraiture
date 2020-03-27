#!/usr/bin/env python

#########
# ppzap #
#########

# ppzap is a command-line program used to flag bad channels that may have gotten
#    through other rounds of zapping.  The basic algorithm is an very simple
#    iterative approach using the channel noise levels, and the other approach
#    uses a fitted model and the calculated channel reduced chi-squared values.
#    Full-functionality is obtained when using ppzap within an interactive
#    python environment.

# Written by Timothy T. Pennucci (TTP; tim.pennucci@nanograv.org).

from __future__ import print_function

from builtins import map
from builtins import range

from pptoas import *


def get_zap_channels(data, nstd=3):
    """
    Return list of proposed channels to zap using median algorithm.

    The algorithm looks at the median noise level in each total intensity
        subintegration and flags channels that are more than nstd standard
        deviations away from it.  The channels are removed, and the process
        iterates until there are no more flagged channels.

    data is a DataBunch object from load_data(...), or a DataPortrait object.
    nstd is the number of standard deviations above the median that serves as
        the threshold for flagging bad channels.
    """
    zap_channels = []
    for isub in data.ok_isubs:
        ichans = list(np.copy(data.ok_ichans[isub]))
        zap_ichans = []
        while (len(ichans)):
            noise_stds = data.noise_stds[isub, 0, ichans]
            median = np.median(noise_stds)
            std = np.std(noise_stds)
            bad_ichans = list(np.where(noise_stds > median + nstd * std)[0])
            if len(bad_ichans):
                zap_ichans.extend(list(np.array(ichans)[bad_ichans]))
                for ichan in np.array(ichans)[bad_ichans]:
                    ichans.pop(ichans.index(ichan))
            else:
                break
        zap_ichans.sort()
        zap_channels.append(zap_ichans)
    return zap_channels


def print_paz_cmds(datafiles, zap_list, all_subs=False, modify=True,
                   outfile=None, quiet=False):
    """
    Print paz commands given a list of datafiles and a zap list.

    datafiles is a list of the datafiles.
    zap_list is the list returned by get_zap_channels(...) that can be indexed
       zap_list[iarch][isub], which would return channel indices to zap.
    all_subs=True will apply the zapping of a channel in any one subint to all
        subints in that archive.
    modify=True would print a '-m' argument for paz, otherwise '-e zap'.
    outfile=None prints to std_out, otherwise it's a file to append to.
    quiet=True suppresses output.
    """
    if not len(datafiles) or not len(zap_list):
        if not quiet:
            print("Nothing to zap.")
            return None
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
                if ii < 0:
                    paz_outfile = datafile + ".zap"
                else:
                    paz_outfile = datafile[:-ii] + "zap"
                print("paz -e zap %s" % datafile)
        last_line = ""
        for isub, bad_ichans in enumerate(zap_list[iarch]):
            for bad_ichan in bad_ichans:
                if not all_subs:
                    print("paz -m -I -z %d -w %d %s" % (bad_ichan, isub,
                                                        paz_outfile))
                else:
                    line = "paz -m -z %d %s" % (bad_ichan, paz_outfile)
                    if line != last_line: print(line)
                    last_line = line
    sys.stdout = sys.__stdout__
    if outfile is not None and not quiet:
        print("Wrote %s." % outfile)


if __name__ == "__main__":

    from optparse import OptionParser

    usage = "Usage: %prog -d <datafile or metafile> [options]"
    parser = OptionParser(usage)
    # parser.add_option("-h", "--help",
    #                  action="store_true", dest="help", default=False,
    #                  help="Show this help message and exit.")
    parser.add_option("-d", "--datafiles",
                      action="store", metavar="archive", dest="datafiles",
                      help="PSRCHIVE archive, or a metafile listing archive filenames, to examine.  \
                              ***NB: Files should NOT be dedispersed!!*** \
                              i.e. vap -c dmc <datafile> should return 0!")
    parser.add_option("-n", "--num_std",
                      action="store", metavar="num_std", dest="nstd",
                      default=5.0,
                      help="Channels with noise levels greater than num_std standard deviations away from the median t value will be flagged.  This process is iterated until there are zero flagged channels.  This is the default method for ppzap, but is ignored if -m is provided.")
    parser.add_option("-N", "--norm",
                      action="store", metavar="normalization", dest="norm",
                      default=None,
                      help="Used only with -n, this will normalize the data before proceeding.  Normalization method is one of 'mean', 'max', 'prof', 'rms', or 'abs'.")
    parser.add_option("-m", "--modelfile",
                      action="store", metavar="model", dest="modelfile",
                      default=None,
                      help="Model file from ppgauss.py, ppspline.py, or PSRCHIVE FITS file that either has same channel frequencies, nchan, & nbin as datafile(s), or is a single profile (nchan = 1, with the same nbin) to be interpreted as a constant template.")
    parser.add_option("-T", "--tscrunch",
                      action="store_true", dest="tscrunch", default=False,
                      help="Examine tscrunch'ed archives and apply channel zapping to all subints.")
    parser.add_option("-S", "--SNR-threshold",
                      metavar="S/N", action="store", dest="SNR_threshold",
                      default=8.0,
                      help="Set a TOA signal-to-noise ratio threshold for flagging low S/N channels; this is used in combination with the number of channels fit to ensure a wideband TOA S/N greater than SNR_threshold [default=8.0].")
    parser.add_option("-R", "--rchi2-threshold",
                      metavar="red_chi2", action="store",
                      dest="rchi2_threshold", default=1.3,
                      help="Set a reduced chi-squared threshold for flagging bad channels [default=1.3].")
    parser.add_option("-o", "--outfile",
                      action="store", metavar="outfile", dest="outfile",
                      default=None,
                      help="Name of output paz command file. Will append. [default=stdout]")
    parser.add_option("--modify",
                      action="store_true", dest="modify", default=False,
                      help="paz commands will modify original datafiles.")
    parser.add_option("--hist",
                      action="store_true", dest="hist", default=False,
                      help="Plot histogram of channel reduced chi-squared values.")
    parser.add_option("--quiet",
                      action="store_true", dest="quiet", default=False,
                      help="Suppress output.")

    (options, args) = parser.parse_args()

    if (options.datafiles is None):
        print("\nppzap.py - Identify bad channels to zap.\n")
        parser.print_help()
        print("")
        parser.exit()

    datafiles = options.datafiles
    nstd = float(options.nstd)
    norm = options.norm
    modelfile = options.modelfile
    tscrunch = options.tscrunch
    SNR_threshold = float(options.SNR_threshold)
    rchi2_threshold = float(options.rchi2_threshold)
    outfile = options.outfile
    modify = options.modify
    hist = options.hist
    quiet = options.quiet

    if modelfile is not None:
        gt = GetTOAs(datafiles=datafiles, modelfile=modelfile, quiet=True)
        gt.get_TOAs(tscrunch=tscrunch, quiet=True)
        gt.get_channels_to_zap(SNR_threshold=SNR_threshold,
                               rchi2_threshold=rchi2_threshold, iterate=True, show=False)
        ok_datafiles = list(np.array(gt.datafiles)[gt.ok_idatafiles])
        print_paz_cmds(ok_datafiles, gt.zap_channels, all_subs=tscrunch,
                       modify=modify, outfile=outfile, quiet=quiet)

        nchan = 0
        nzap = 0
        for iarch in range(len(ok_datafiles)):
            for isub in range(len(gt.channel_red_chi2s[iarch])):
                nchan += len(gt.channel_red_chi2s[iarch][isub])
                nzap += len(gt.zap_channels[iarch][isub])

        if hist:
            red_chi2s = []
            for iarch in range(len(ok_datafiles)):
                for isub in range(len(gt.channel_red_chi2s[iarch])):
                    red_chi2s.extend(gt.channel_red_chi2s[iarch][isub])
            red_chi2s = np.nan_to_num(np.array(red_chi2s))
            nzap_rchi2 = sum(np.array(red_chi2s) > rchi2_threshold)
            plt.hist(red_chi2s, bins=min(50, len(red_chi2s)), log=True)
            ymin, ymax = plt.ylim()
            plt.vlines(rchi2_threshold, ymin, ymax, linestyles='dashed')
            plt.ylim(ymin, ymax)
            plt.xlabel(r"Reduced $\chi^2$")
            plt.ylabel("#")
            plt.title("%s\n" % datafiles + r"%d / %d channels w/ $\chi^2_{red}$ > %.1f" % (
            nzap_rchi2, nchan, rchi2_threshold))
            plt.savefig(datafiles + "_ppzap_hist.png")

        if not quiet:
            print("ppzap.py found %d channels to zap out of a total %d channels fit (=%.2f%%) in %s." % (
            nzap, nchan, 100 * float(nzap) / nchan, datafiles))

    else:
        if file_is_type(datafiles, "ASCII"):
            all_datafiles = [datafile[:-1] for datafile in \
                             open(datafiles, "r").readlines()]
        else:
            all_datafiles = [datafiles]
        nchan = 0
        zap_channels = []
        for datafile in all_datafiles:
            try:
                data = load_data(datafile, dedisperse=False,
                                 dededisperse=False, tscrunch=tscrunch, pscrunch=True,
                                 fscrunch=False, rm_baseline=rm_baseline,
                                 flux_prof=False, refresh_arch=False, return_arch=False,
                                 quiet=True)
            except RuntimeError:
                if not quiet:
                    print("Cannot load_data(%s).  Skipping it." % datafile)
                continue
            nchan += np.array(list(map(len, data.ok_ichans))).sum()
            if norm is not None:
                for isub in data.ok_isubs:
                    data.subints[isub, 0] = normalize_portrait(
                        data.subints[isub, 0], method=norm,
                        weights=data.weights[isub], return_norms=False)
                    data.noise_stds[isub, 0] = get_noise(data.subints[isub, 0],
                                                         chans=True)
            zap_channels.append(get_zap_channels(data, nstd=nstd))
        print_paz_cmds(all_datafiles, zap_channels, all_subs=tscrunch,
                       modify=modify, outfile=outfile, quiet=quiet)

        nzap = 0
        for iarch in range(len(zap_channels)):
            for isub in range(len(zap_channels[iarch])):
                nzap += len(zap_channels[iarch][isub])

        if not quiet:
            print("ppzap.py found %d channels to zap out of a total %d channels (=%.2f%%) in %s." % (
            nzap, nchan, 100 * float(nzap) / nchan, datafiles))
