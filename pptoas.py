#!/usr/bin/env python

#Calls PulsePortraiture to generate TOAs and DM corrections

from optparse import OptionParser,sys

usage = "usage: %prog [options]"
parser = OptionParser(usage)
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
parser.add_option("--showplot",
                  action="store_true", dest="showplot", default=False,
                  help="Plot fit results. Only useful if nsubint > 1. [default=False]")
parser.add_option("--quiet",
                  action="store_true", dest="quiet", default=False,
                  help="Minimal to stdout.")

(options, args) = parser.parse_args()

if options.datafile is None and options.metafile is None or options.modelfile is None or options.mtype is None:
    print "\npptoas.py - least-squares fit of pulsar phase-frequency portrait\n \
           to 2D template to generate TOAs and DM correction.\n"
    parser.print_help()
    sys.exit()

from PulsePortraiture import *

#Make template
datafile = options.datafile
metafile = options.metafile
modelfile = options.modelfile
mtype = options.mtype
outfile = options.outfile
showplot = options.showplot
quiet = options.quiet
#quiet = options.quiet

if not metafile:
    gt = GetTOAs(datafile,modelfile,mtype,outfile,quiet=quiet)
    if showplot: gt.show_results()
else:
    datafiles = open(metafile,"r").readlines()
    for datafile in datafiles:
        gt = GetTOAs(datafile[:-1],modelfile,mtype,outfile,quiet=quiet)
        if showplot: gt.show_results()
