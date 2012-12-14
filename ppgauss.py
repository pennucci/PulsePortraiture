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
parser.add_option("--fixloc",
                  action="store_true", dest="fixloc", default=False,
                  help="Fix locations of gaussians across frequency. [default=True]")
parser.add_option("--fixwid",
                  action="store_true", dest="fixwid", default=False,
                  help="Fix widths of gaussians across frequency. [default=True]")
parser.add_option("--fixamp",
                  action="store", dest="fixamp", default=False,
                  help="Fix amplitudes of gaussians across frequency. [default=False]")
parser.add_option("--niter",
                  action="store", metavar="int", dest="niter", default=0,
                  help="Number of iterations to loop over generating better model. [default=0]")
parser.add_option("--showplots",
                  action="store_true", dest="showplots", default=False,
                  help="Show residual plot after each iteration. [default=False]")
parser.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="More to stdout.")

(options, args) = parser.parse_args()

if options.datafile is None:
    print "\nppgauss.py - generate a 2D phase-frequency model portrait from Gaussian components.\n"
    parser.print_help()
    parser.exit()

from PulsePortraiture import DataPortrait,ModelPortrait_Gaussian

datafile = options.datafile
outfile = options.outfile
fixloc = options.fixloc
fixwid = options.fixwid
fixamp = options.fixamp
niter = int(options.niter)
showplots = options.showplots
quiet = not options.verbose

dp = DataPortrait(datafile)
dp.make_gaussian_model_portrait(locparams=0.0,fixloc=fixloc,widparams=0.0,fixwid=fixwid,ampparams=0.0,fixamp=fixamp,nu_ref=None,niter=niter,outfile=outfile,residplot=showplots,quiet=quiet)
