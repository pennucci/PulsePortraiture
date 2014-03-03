#!/usr/bin/env python

import os
from pplib import *

#Use example.gmodel and example.par to design a fake pulsar
modelfile = "example.gmodel"
ephemeris = "example.par"

#Generate some fake datafiles
#These files will be homogenous, even though they don't need to be

nfiles = 3      #Number of datafiles/epochs
MJD0 = 50000.00 #Start day [MJD]
days = 20.0     #Days between epochs
nsub = 10       #Number of subintegrations
npol = 1        #Number of polarization (can be 4, but will only use total I)
nchan = 64      #Number of frequency channels
nbin = 512      #Number of phase bins
nu0 = 1500.0    #Center of the band [MHz]
bw = 800.0      #Bandwidth [MHz]
tsub = 120.0    #Length of subintegration [s]
noise_std = 2.77#Noise level of the band, per subintegration [flux units] (switch to snr)
dDM_mean = 3e-4 #Add in random dispersion measure offsets with this mean value
dDM_std = 2e-4  #Add in random dispersion measure offsets with this std
dDMs = np.random.normal(dDM_mean, dDM_std, nfiles)
#dDMs = np.zeros(nfiles) #Uncomment and set dDM_mean and dDM_std to zero for no injected dDMs
                #Adding scattering may slow down the fit
t_scat = 50e-6  #Add scattering with this timescale [s] w.r.t. nu0
scint = True    #Add random scintillation
alpha = -4.4    #t_scat will follow a powerlaw with this spectral index

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
    make_fake_pulsar(modelfile, ephemeris, outfile="example-%d.fits"%(ifile+1),
            nsub=nsub, npol=npol, nchan=nchan, nbin=nbin, nu0=nu0, bw=bw,
            tsub=tsub, phase=0.0, dDM=dDMs[ifile], start_MJD=None,
            weights=weights, noise_std=noise_std, scale=1.0, dedisperse=False,
            t_scat=t_scat, alpha=alpha, scint=scint, state="Coherence",
            obs="GBT", quiet=quiet)
    #NB: the input parfile cannot yet have binary parameters

os.system("ls example-*.fits > example.meta")
metafile = "example.meta"
#If you wanted to add a bunch of datafiles together, you would make a metafile
#containing the filenames, and feed it to quick_add_archs.  I recommend you 
#use PSRCHIVE's psradd instead, or just a single long-integration observation.
print "Adding data archives..."
outfile = "example.port"
quick_add_archs(metafile, outfile, quiet=False)

#Now we want to "build" our gaussian model from the data
print "Running ppgauss.py to fit a gaussian model..."
import ppgauss as pg
datafile = "example.port"
#Initiate Class instance
dp = pg.DataPortrait(datafile)
#Have a look at the data you're fitting
dp.show_data_portrait()
#Fit a model; see ppgauss.py for all options
dp.make_gaussian_model(ref_prof=(nu0, bw/4), tau=(t_scat * dp.nbin) / dp.Ps[0],
        fixloc=True, fixscat= not bool(t_scat), niter=3,
        fiducial_gaussian=True, writemodel=True, outfile="example-fit.gmodel",
        model_name="Example_Fit", residplot="example.png", quiet=False)
#You can always then continue iterations using:
#niter = # 
#modelfile = example-fit.gmodel
#dp.make_gaussian_model(modelfile, niter=niter)
#You can check this fitted model against the "input" true model example.gmodel,
#assuming the reference frequencies are the same.

#Now we would measure TOAs and DMs
print "Running pptoas.py to fit TOAs and DMs..."
import pptoas as pt
#Set the DM to which the offsets are referenced (eg. from the input ephemeris)
i,o = os.popen4("grep DM example.par")
DM0 = float(o.readline().split()[1])
#Initiate Class instance; one could also use a smoothed average of the data
#as a model instead of the analytic gaussian model
gt = pt.GetTOAs(metafile, "example-fit.gmodel")
gt.get_TOAs(DM0=DM0)
#Show results from first datafile
gt.show_results()
#Show typical fit
gt.show_fit()
#Write TOAs
gt.write_TOAs(outfile="example.tim")
#See fitted versus injected DMs
#print ""
#print "Injected DMs, mean, std:"
#print dDMs, dDM_mean, dDM_std
#print "Measured average DM offsets, mean, std:"
dDM_fit = pt.np.array(gt.DeltaDM_means)
#print dDM_fit, dDM_fit.mean(), dDM_fit.std()
diff = dDMs - dDM_fit
#print "Difference, mean, std:"
#print diff, diff.mean(), diff.std()
