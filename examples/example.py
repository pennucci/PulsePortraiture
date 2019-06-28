#!/usr/bin/env python

import os
from pplib import *


#Use example.gmodel and example.par to design a fake pulsar
modelfile = "example.gmodel"
ephemeris = "example.par"


#Choose which modeling program to use
model_routine = "ppspline" #"ppspline" or "ppgauss"


#Generate some fake datafiles
#These files will be homogenous, even though they don't need to be
nfiles = 5       #Number of datafiles/epochs
MJD0 = 57202.00  #Start day [MJD]
days = 20.0      #Days between epochs
nsub = 10        #Number of subintegrations
npol = 1         #Number of polarization (can be 4, but will only use total I)
nchan = 64       #Number of frequency channels
nbin = 512       #Number of phase bins
nu0 = 1500.0     #Center of the band [MHz]
bw = 800.0       #Bandwidth [MHz]
tsub = 60.0      #Length of subintegration [s]
noise_std = 1.5  #Noise level of the band, per subintegration [flux units]
dDM_mean = 3e-4  #Add in random dispersion measure offsets with this mean value
dDM_std = 2e-4   #Add in random dispersion measure offsets with this std
dDMs = np.random.normal(dDM_mean, dDM_std, nfiles)
#dDMs = np.zeros(nfiles) #Uncomment this line and comment previous line for \
                         #no injected dDMs
scint = True     #Add random scintillation
#Scattering parameters will be read from the modelfile.
#Adding/fitting scattering to the fake data may slow down the fit:
fitscat = True   #Fit scattering timescale
fitalpha = False #Fit the scattering index
weights = np.ones([nsub, nchan]) #Change if you want to have an "RFI" mask
                                 #e.g. band edges zapped:
                                 #weights[:,:10] = 0 ; weights[:,-10:] = 0
                                 #e.g. first and last subints zapped:
                                 #weights[0] = 0 ; weights[-1] = 0

print "Making fake data..."
for ifile in range(nfiles):
    if ifile == 0: quiet=False
    else: quiet = True
    start_MJD = pr.MJD(MJD0 + ifile*days)
    make_fake_pulsar(modelfile, ephemeris, outfile="example-%d.fits"%(ifile+1),
            nsub=nsub, npol=npol, nchan=nchan, nbin=nbin, nu0=nu0, bw=bw,
            tsub=tsub, phase=0.0, dDM=dDMs[ifile], start_MJD=start_MJD,
            weights=weights, noise_stds=noise_std, scales=1.0,
            dedispersed=False, scint=scint, state="Stokes", obs="GBT",
            quiet=quiet)
    #NB: the input parfile for fake data cannot yet have binary parameters
os.system('psredit -q -m -c rcvr:name="fake_rx" -c be:name="fake_be" example-*.fits')


#Here we build an average portrait from the data
if nfiles > 1:
    import ppalign as pa
    metafile = "example.meta"
    os.system("ls example-*.fits > %s"%metafile)
    outfile = "example.port"
    print "Adding data archives..."
    pa.align_archives(metafile=metafile, initial_guess="example-1.fits",
            tscrunch=True, pscrunch=True, outfile=outfile, niter=1, quiet=True)
    #...or you could use PSRCHIVE's psradd to get a high SNR portrait.
    #os.system("psradd -T -P -E %s -M %s -o %s"%(ephemeris, metafile, outfile))


#Now we want to "build" our model from the data...
if nfiles > 1: datafile = "example.port"
else: datafile = "example-1.fits"
norm = "prof" #Normalization method (None, mean, max, prof, rms, abs)

#...with ppspline...
if not model_routine == "ppgauss":
    print "Running ppspline.py to fit a PCA/B-spline model..."
    import ppspline as ppi
    fitted_modelfile = "example-fit.spl"
    #Initial Class instance
    dp = ppi.DataPortrait(datafile)
    dp.normalize_portrait(norm)
    #Have a look at the data you're fitting
    print "Have a look at the average data you're fitting..."
    dp.show_data_portrait()
    dp.make_spline_model(max_ncomp=3, smooth=True, snr_cutoff=150.0,
            rchi2_tol=0.1, k=3, sfac=1.0, max_nbreak=None, model_name=None,
            quiet=False)
    print "Have a look at the mean profile and eigenprofiles..."
    dp.show_eigenprofiles()
    print "Have a look at the spline curve model of profile evolution..."
    dp.show_spline_curve_projections()
    dp.write_model(fitted_modelfile, quiet=False)

#...or using ppgauss...
else:
    print "Running ppgauss.py to fit a gaussian model..."
    import ppgauss as ppg
    fitted_modelfile = "example-fit.gmodel"
    #Initiate Class instance
    dp = ppg.DataPortrait(datafile)
    dp.normalize_portrait(norm)
    #Have a look at the data you're fitting
    print "Have a look at the average data you're fitting..."
    dp.show_data_portrait()
    #Fit a model; see ppgauss.py for all options
    dp.make_gaussian_model(ref_prof=(nu0, bw/4), fixloc=True,
            fixscat=not(fitscat), fixalpha=not(fitalpha), niter=3,
            fiducial_gaussian=True, writemodel=True, outfile=fitted_modelfile,
            writeerrfile=True, model_name="example-fit",
            residplot=None, quiet=False)
    #You can always then continue iterations using the ppgauss option -I or by:
    #niter = #
    #modelfile = example-fit.gmodel
    #dp.make_gaussian_model(modelfile, niter=niter)
    #You can check this fitted model against the "input" true model
    #example.gmodel, assuming the reference frequencies are the same.
print "Have a look at the model portrait you've made..."
dp.show_model_fit()

#Now we would measure TOAs and DMs
print "Running pptoas.py to fit TOAs and DMs..."
import pptoas as ppt
#Set the DM to which the offsets are referenced (e.g. from the input ephemeris)
i,o = os.popen4("grep DM %s"%ephemeris)
DM0 = float(o.readline().split()[1])
#Initiate Class instance; one could also use a smoothed average of the data
#as a model instead of the analytic gaussian model
gt = ppt.GetTOAs(metafile, fitted_modelfile)
gt.get_TOAs(DM0=DM0)
#Show results from first datafile
#gt.show_results()
#Show typical fit
print "Have a look at how one subintegration was fit by the model..."
gt.show_fit()
#Write TOAs
write_TOAs(gt.TOA_list, SNR_cutoff=0.0, outfile="example.tim", append=False)
#See fitted versus injected DMs
#print ""
#print "Injected DMs, mean, std:"
#print dDMs, dDM_mean, dDM_std
#print "Measured average DM offsets, mean, std:"
dDM_fit = ppt.np.array(gt.DeltaDM_means)
#print dDM_fit, dDM_fit.mean(), dDM_fit.std()
diff = dDMs - dDM_fit
#print "Difference, mean, std:"
#print diff, diff.mean(), diff.std()
