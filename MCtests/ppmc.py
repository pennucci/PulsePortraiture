nodes = True

if nodes: from pplib_dist import *
else: from pplib import *
from ppgauss import DataPortrait
import os, sys, time, pickle

(niter, nchan, nbin) = map(int, sys.argv[-8:-5])
(nu0, bw, DM_inj, noise_std) = map(float, sys.argv[-5:-1])
scratchdir = sys.argv[-1]
ntr = niter

if nodes:
    vnodenum = int(os.getenv("PBS_VNODENUM"))
    workdir = os.getenv("PBS_O_WORKDIR")
else:
    vnodenum = 0
    workdir = "./"
MCname = "DM%.1e_%d_%d"%(DM_inj, niter, vnodenum)

modelfile = "%s/model"%scratchdir
ephemfile = "%s/ephemeris"%scratchdir
outfile = "%s/fake.fits"%scratchdir

o = os.popen("grep DM %s"%ephemfile)
DM0 = o.readlines()
DM0 = float(DM0[0].split()[1])

phis = []
DMs = []
phierrs = []
DMerrs = []
dphis = []
dDMs = []
covars = []

def DM_sample():
    #return np.random.normal(DM_inj, DMerr_inj)
    #return DM_inj
    return np.random.uniform()*2*DM_inj - DM_inj

def phi_sample():
    #return np.random.normal(phi_inj, phierr_inj)
    #return 0.33
    return np.random.uniform() - 0.5

def iteration():
    ps = phi_sample()
    ds = DM_sample()
    subint = port + np.random.normal(0.0, noise_std, nchan*nbin).reshape(nchan,
            nbin)
    channel_SNRs = subint.std(axis=1) / get_noise(subint, chans=True)
    nu_fit = guess_fit_freq(freqs, channel_SNRs)
    #should not have reassigned same variable name!
    rot_subint = rotate_portrait(subint, -ps, 0.0, P, freqs, nu_fit)
    phase_guess = fit_phase_shift(rot_subint.mean(axis=0),
            model.mean(axis=0)).phase
    pg0 = phase_guess
    phase_guess %= 1.0
    if phase_guess > 0.5:
        phase_guess -= 1.0
    if abs(phase_guess) > 0.5:
        print "GUESS", pg0, phase_guess
    rot_subint = rotate_portrait(rot_subint, 0.0, -ds, P, freqs, nu_fit)
    if not nodes: show_portrait(rot_subint, phases, freqs)
    DM_guess = DM0
    (phi, DM, scalex, param_errs, nu_zero, covariance, red_chi2,
            duration, nfeval, rc) = fit_portrait(rot_subint, model,
                    np.array([phase_guess, DM_guess]), P, freqs,
                    nu_fit, bounds=[(None, None), (None, None)],
                    id = "mc_iter_%d"%niter, quiet=True)
    if abs(phi) > 0.5:
        print "PHI", phi
    phi_err, DM_err = param_errs[0], param_errs[1]
    if not nodes: print phase_guess, DM_guess, phi, DM, phi_err*P*1e9
    phis.append(phi)
    DMs.append(DM)
    phierrs.append(phi_err)
    DMerrs.append(DM_err)
    dphis.append(phi-ps)
    dDMs.append(DM-ds)
    covars.append(covariance)

make_fake_pulsar(modelfile, ephemfile, outfile, nsub=1, npol=1, nchan=nchan,
        nbin=nbin, nu0=nu0, bw=bw, tsub=60.0, phase=0.0, dDM=0.0,
        start_MJD=None, weights=None, noise_std=0.0, t_scat=None,
        bw_scint=None, state="Coherence", obs="GBT", quiet=True)

fp = DataPortrait(outfile, quiet=True)
phases = fp.phases
freqs = fp.freqs
P = fp.Ps[0]
port = fp.port
modelname, ngauss, model = read_model(modelfile, fp.phases, fp.freqs,
        quiet=True)

start = time.time()
while(ntr):
    iteration()
    ntr -= 1

#if vnodenum == 0:
inffile = open("%s/DM%.1e_%d.inf"%(scratchdir, DM_inj, niter), "w")
inffile.write("niter_per_node %d\n"%niter)
inffile.write("nchan          %d\n"%nchan)
inffile.write("nbin           %d\n"%nbin)
inffile.write("nu0            %f\n"%nu0)
inffile.write("bw             %f\n"%bw)
inffile.write("P0             %.2f\n"%(P*1e3))
inffile.write("DM0            %.4f\n"%DM0)
inffile.write("DM_inj         %.4f\n"%DM_inj)
inffile.write("noise_std      %.2f\n"%noise_std)
inffile.close()

print "MC %s took %.1f sec"%(MCname, (time.time() - start))

phis = np.array(phis)
DMs = np.array(DMs)
phierrs = np.array(phierrs)
DMerrs = np.array(DMerrs)
dphis = np.array(dphis)
dDMs = np.array(dDMs)
covars = np.array(covars)

pickfile = open("%s/%s_results.pick"%(scratchdir, MCname), "wb")
pickle.dump([phis, DMs, phierrs, DMerrs, dphis, dDMs, covars], pickfile,
        protocol=2)
pickfile.close()
