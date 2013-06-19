#!/usr/env python

import pptoas as pt
import ppgauss as pg
from pplib import *
import pickle

DM_inj = 5e-4
DMerr_inj = 1e-3
def DM_sample():
    #return np.random.normal(DM_inj, DMerr_inj)
    return np.random.uniform()*2*DM_inj - DM_inj
#phase_inj = 0.23456789
#phase_inj = 0.0
#phaseerr_inj = 1.66e-6 #~5 us precision
def phase_sample():
    #return np.random.normal(phase_inj, phaseerr_inj)
    return np.random.uniform() - 0.5

truemodel = "true.model"    #true model, written, not fit
#guessedmodel = "guessed.model"   #this is the guessed model from "data", no phase or DM added
ephemfile = "true.par"
outfile = "fake.fits"   #true model with noise and injected phase, DM
modelfile = truemodel

nu0 = 1500.0
DM0 = 0.0  #From true.par; things get tricky if DM0 is large, because nu_fit and nu_ref/nu0 are different, and phi_prime will have an integer number of rotations

def iteration():
    ps = phase_sample()
    ds = DM_sample()
    make_fake_pulsar(modelfile, ephemfile, outfile, nsub=1, npol=1, nchan=512,
            nbin=512, nu0=nu0, bw=800.0, tsub=300.0, phase=ps, dDM=ds,
            start_MJD=None, weights=None, noise_std=noise_std, t_scat=None,
            bw_scint=None, state="Coherence", obs="GBT", quiet=True)
    #fp = pg.DataPortrait("fake.fits")
    #fp.show_data_portrait()
    gt = pt.GetTOAs(outfile, modelfile, nu_ref=nu0, DM0=DM0, one_DM=False,
            bary_DM=False, common=True, quiet=True)
    gt.get_TOAs(quiet=True)
    #gt.show_fit(quiet=True)
    #phis.append(gt.phis[0][0])            #CHECK!!!!!
    phi = gt.phis[0][0]
    phi_prime = phase_transform(phi, gt.DMs[0][0], nu0, gt.nu_fits[0][0], gt.Ps[0][0])
    phis.append(phi_prime)            #CHECK!!!!!
    phierrs.append(gt.phi_errs[0][0])
    phase_samples.append(ps)
    DMs.append(gt.DMs[0][0])              #CHECK!!!!!
    DMerrs.append(gt.DM_errs[0][0])
    DM_samples.append(ds)
    print niter

def histo(data, fgnm, xlabel, nbin, true=(None,None), normed=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(data, bins=nbin, normed=normed)
    plt.xlabel(xlabel)
    ax.text(0.8,0.9,"Mean %.2e\n Std. %.2e\n"%(data.mean(), data.std()),
            ha="center", va="center", transform=ax.transAxes)
    if true[0] is not None:
        plt.title("True: %.5e +/- %.5e"%true)
        #plt.plot(np.linspace(plt.xlim()[0],plt.xlim()[1],nbin),
        #        gaussian_profile(bins, true[0], 2.35482*true[1], norm=True),
        #        "r--")
        ax.vlines(true[0], plt.ylim()[0], plt.ylim()[1], "r", "solid")
        ax.vlines(true[0] - true[1], plt.ylim()[0], plt.ylim()[1], "r",
                "dashed")
        ax.vlines(true[0] + true[1], plt.ylim()[0], plt.ylim()[1], "r",
                "dashed")
    plt.savefig("%s.png"%fgnm)
    plt.close("all")

niter = 1000
#bins = 5
bins = niter/10
ntr = niter

P = 345.67890123456789**-1  #From true.par
phis = []
phierrs = []
phase_samples = []
DMs = []
DMerrs = []
DM_samples = []

noise_std = 1.0

start = time.time()
while(niter):
    #print niter
    iteration()
    niter -= 1
    if niter == 0:
        print time.time()-start, ntr, (time.time()-start)/ntr
        phis = np.array(phis)
        phierrs = np.array(phierrs)
        phase_samples = np.array(phase_samples)
        DMs = np.array(DMs)
        DMerrs = np.array(DMerrs)
        DM_samples = np.array(DM_samples)
        histo(phis, "phis_%.2f_%d"%(noise_std, ntr), "Phi [phase]", bins,
                (None, None))
        histo(phis, "phis_%.2f_%d"%(noise_std, ntr), "Phi [phase]", bins,
                (None, None))
        print "Avg phi: %.5f"%phis.mean()
        print "Avg phi: %.5f"%(phis.mean()*P*1e6)
        print "Std phi: %.5f"%phis.std()
        print "Std phi: %.2f"%(phis.std()*P*1e6)
        histo(phierrs, "phierrs_%.2f_%d"%(noise_std, ntr),
                "Phi Errors [phase]", bins,(None, None))
        print "Avg phierr: %.5f"%phierrs.mean()
        print "Avg phierr: %.5f"%(phierrs.mean()*P*1e6)
        print "Std phierr: %.5f"%phierrs.std()
        print "Std phierr: %.5f"%(phierrs.std()*P*1e6)
        histo(phis-phase_samples, "phi_diff_%.2f_%d"%(noise_std, ntr),
                "Fitted phi - True phi [phase]", bins)
        print "Avg phi diff: %.5f"%(phis-phase_samples).mean()
        print "Avg phi diff: %.5f"%((phis-phase_samples).mean()*P*1e6)
        print "Std phi diff: %.5f"%((phis-phase_samples).std())
        print "Std phi diff: %.5f"%((phis-phase_samples).std()*P*1e6)
        histo(DMs, "DMs_%.2f_%d"%(noise_std, ntr), "DM [pc cm**-3]", bins,
                (DM_inj+DM0,DMerr_inj))
        print "Avg DM: %.5f"%DMs.mean()
        print "Std DM: %.5f"%DMs.std()
        histo(DMerrs, "DMerrs_%.2f_%d"%(noise_std, ntr),
                "DM Errors [pc cm**-3]", bins,(None, None))
        print "Avg DMerr: %.5f"%DMerrs.mean()
        print "Std DMerr: %.5f"%DMerrs.std()
        histo(DMs-(DM_samples+DM0), "DM_diff_%.2f_%d"%(noise_std, ntr),
                "Fitted DM - True DM [pc cm**-3]", bins)
        print "Avg DM diff: %.5f"%((DMs-DM_samples).mean())
        print "Std DM diff: %.5f"%((DMs-DM_samples).std())
        plt.plot(phis, DMs, 'k+', ms=10)
        plt.savefig("phi_dm_hist.png")
        plt.close("all")
        pickfile = open("ppMC_results.pick","wb")
        pickle.dump([phis, phierrs, phase_samples, DMs, DMerrs, DM_samples],
                pickfile, protocol=2)
        pickfile.close()
