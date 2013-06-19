#!/usr/bin/env python

import os
from pplib import *

#Use example.gmodel and example.par to design a fake pulsar
modelfile = "example.gmodel"
ephemfile = "example.par"

#Generate some fake datafiles
#These files will be homogenous, even though they don't need to be

nfiles = 1      #Number of datafiles/epochs
MJD0 = 50000.00 #Start day [MJD]
days = 20.0     #Days between epochs
nsub = 1       #Number of subintegrations
npol = 1        #Number of polarization (can be 4, but will only use total I)
nchan = 128     #Number of frequency channels
nbin = 128      #Number of phase bins
nu0 = 1500.0    #Center of the band [MHz]
bw = 800.0      #Bandwidth [MHz]
tsub = 120.0    #Length of subintegration [s]
noise_std = 2.77#Noise level of the band, per subintegration [flux units]
dDMs = np.zeros(nfiles) #Uncomment and set dDM_mean and dDM_std to zero for no injected dDMs
weights = np.ones([nsub, nchan]) #Change if you want to have an "RFI" mask
                                 #eg. band edges zapped:
                                 #weights[:,:10] = 0 ; weights[:,-10:] = 0
                                 #eg. first and last subints zapped:
                                 #weights[0] = 0 ; weights[-1] = 0
print "Making fake data..."
for ifile in range(nfiles):
    if ifile == 0: quiet=False
    else: quiet = True
    start_MJD = MJD0 + ifile*days
    make_fake_pulsar(modelfile, ephemfile, outfile="example-%d.fits"%(ifile+1),
            nsub=nsub, npol=npol, nchan=nchan, nbin=nbin, nu0=nu0, bw=bw,
            tsub=tsub, phase=0.0, dDM=dDMs[ifile], start_MJD=None,
            weights=weights, noise_std=noise_std, t_scat=None, bw_scint=None,
            state="Coherence", obs="GBT", quiet=quiet)
    #NB: t_scat, bw_scint not yet implemented
    #NB: the input parfile cannot yet have binary parameters

#Now we want to "build" our gaussian model from the data
print "Running ppgauss.py to fit a gaussian model..."
import ppgauss as pg
datafile = "example-1.fits"
#Initiate Class instance
dp = pg.DataPortrait(datafile)
#Have a look at the data you're fitting
dp.show_data_portrait()
#Fit a model; see ppgauss.py for all options
dp.make_gaussian_model(ref_prof=(nu0, bw/4), niter=5, writemodel=True,
        outfile="example-fit.gmodel", model_name="Example_Fit",
        residplot="example.png", quiet=False)
