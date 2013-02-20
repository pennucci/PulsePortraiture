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
parser.add_option("--freq",
                  action="store", metavar="freq", dest="nu_ref", default=None,
                  help="Reference frequency [MHz] for the gaussian model; the initial profile to fit will be centered on this freq. [default=PSRCHIVE weighted center frequency]")
parser.add_option("--bw",
                  action="store", metavar="bw", dest="bw_ref", default=None,
                  help="Used with --freq; amount of bandwidth [MHz] centered on nu_ref to average for the initial profile fit. [default=Full bandwidth]")
parser.add_option("--fixloc",
                  action="store_true", dest="fixloc", default=False,
                  help="Fix locations of gaussians across frequency. [default=False]")
parser.add_option("--fixwid",
                  action="store_true", dest="fixwid", default=False,
                  help="Fix widths of gaussians across frequency. [default=False]")
parser.add_option("--fixamp",
                  action="store_true", dest="fixamp", default=False,
                  help="Fix amplitudes of gaussians across frequency. [default=False]")
parser.add_option("--niter",
                  action="store", metavar="int", dest="niter", default=0,
                  help="Number of iterations to loop over generating better model. [default=0]")
parser.add_option("--figure", metavar="figurename",
                  action="store", dest="figure", default=None,
                  help="Save PNG figure of final fit to figurename. [default=Not saved]")
parser.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="More to stdout.")

(options, args) = parser.parse_args()

if options.datafile is None:
    print "\nppgauss.py - generate a 2D phase-frequency model portrait from Gaussian components.\n"
    parser.print_help()
    parser.exit()

from PulsePortraiture import DataPortrait

datafile = options.datafile
outfile = options.outfile
if options.nu_ref: nu_ref = float(options.nu_ref)
else: nu_ref = options.nu_ref
if options.bw_ref: bw_ref = float(options.bw_ref)
else: bw_ref = options.bw_ref
fixloc = options.fixloc
fixwid = options.fixwid
fixamp = options.fixamp
niter = int(options.niter)
figure = options.figure
quiet = not options.verbose

dp = DataPortrait(datafile)
dp.make_gaussian_model_portrait(ref_prof=(nu_ref,bw_ref),locparams=0.0,fixloc=fixloc,widparams=0.0,fixwid=fixwid,ampparams=0.0,fixamp=fixamp,niter=niter,outfile=outfile,residplot=figure,quiet=quiet)
