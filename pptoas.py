#!/usr/bin/env python

#Calls PulsePortraiture to generate TOAs and DM corrections

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
                  help=".model file to which the data are fit")
parser.add_option("-t", "--modeltype",
                  action="store", metavar="mtype", dest="mtype",
                  help='Must be either "gauss" (created by ppgauss.py) or "smooth" (created by psrsmooth).')
parser.add_option("-o", "--outfile",
                  action="store", metavar="timfile", dest="outfile", default=None,
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
                  action="store", metavar="errfile", dest="errfile", default=None,
                  help="If specified, will write the fitted DM errors to errfile. Will append.")
parser.add_option("--showplot",
                  action="store_true", dest="showplot", default=False,
                  help="Plot fit results. Only useful if nsubint > 1.")
parser.add_option("--quiet",
                  action="store_true", dest="quiet", default=False,
                  help="Minimal to stdout.")

(options, args) = parser.parse_args()

if options.datafile is None and options.metafile is None or options.modelfile is None or options.mtype is None:
    print "\npptoas.py - least-squares fit of pulsar phase-frequency portrait\n \
           to 2D template to produce TOAs and DM.\n"
    parser.print_help()
    parser.exit()

from PulsePortraiture import *

datafile = options.datafile
metafile = options.metafile
modelfile = options.modelfile
mtype = options.mtype
if options.DM0: DM0 = float(options.DM0)
else: DM0 = None
bary_DM = options.bary_DM
one_DM = options.one_DM
outfile = options.outfile
errfile = options.errfile
showplot = options.showplot
quiet = options.quiet

if not metafile:
    gt = GetTOAs(datafile,modelfile,mtype,DM0,bary_DM,one_DM,outfile,errfile,quiet=quiet)
    if showplot: gt.show_results()
else:
    datafiles = open(metafile,"r").readlines()
    for datafile in datafiles:
        gt = GetTOAs(datafile[:-1],modelfile,mtype,DM0,bary_DM,one_DM,outfile,errfile,quiet=quiet)
        if showplot: gt.show_results()
