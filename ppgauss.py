#!/usr/bin/env python

#Calls PulsePortraiture to generate analytic Gaussian template portrait

from optparse import OptionParser

usage = "usage: %prog [options]"
parser = OptionParser(usage)
#parser.add_option("-h", "--help",
#                  action="store_true", dest="help", default=False,
#                  help="Show this help message and exit.")
parser.add_option("-d", "--datafile",
                  action="store", metavar="archive", dest="datafile",
                  help="PSRCHIVE archive from which to generate Gaussian portrait.")
parser.add_option("-o", "--outfile",
                  action="store", metavar="outfile", dest="outfile",
                  help="Name of output model file name. [default=archive.model]")
parser.add_option("--nsubfit",
                  action="store", metavar="int", dest="nsubfit", default=8,
                  help="Number of subfits across the band. [default=8]")
parser.add_option("--niter",
                  action="store", metavar="int", dest="niter", default=0,
                  help="Number of iterations to loop over generating better model. [default=0]")
parser.add_option("--showplots",
                  action="store_true", dest="showplots", default=False,
                  help="Show residual plot after each iteration. [default=False]")
#parser.add_option("--quiet",
#                  action="store_false", dest="quiet", default=False,
#                  help="Nothing to stdout.")

(options, args) = parser.parse_args()

if options.datafile is None:
    print "\nppgauss.py - generate a 2D phase-frequency model portrait from Gaussian components.\n"
    parser.print_help()
    parser.exit()

from PulsePortraiture import DataPortrait,ModelPortrait_Gaussian

datafile = options.datafile
outfile = options.outfile
nsubfit = int(options.nsubfit)
niter = int(options.niter)
showplots = options.showplots
if showplots: shownone = False
else: shownone = True
#quiet = options.quiet

dp = DataPortrait(datafile)
dp.fit_profile()
dp.make_Gaussian_model_portrait(nsubfit=nsubfit,niter=niter,outfile=outfile,shownone=shownone)
