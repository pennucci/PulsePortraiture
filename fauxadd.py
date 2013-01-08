from PulsePortraiture import *

archs = open(sys.argv[-2],"r").readlines()
totarch = sys.argv[-1]

for aa in range(len(archs)):
    datafile = archs[aa].split()[0]
    source,arch,port,portx,noise_stdev,fluxprof,fluxprofx,prof,nbin,phases,nu0,bw,nchan,freqs,freqsx,nsub,P,MJD,weights,normweights,maskweights,portweights = load_data(datafile,dedisperse=True,tscrunch=True,pscrunch=True,quiet=True,rm_baseline=(0,0),Gfudge=1.0)
    if aa == 0:
        totport = np.zeros([nchan,nbin])
    maxbin = prof.argmax()
    rotport = rotate(port.transpose(),maxbin-(nbin/2)).transpose()
    totport += rotport

totport /= len(archs)

arch = pr.Archive_load(totarch)
arch.tscrunch()
arch.pscrunch()
I = arch.get_Integration(0)
for nn in range(nchan):
    for bb in range(nbin):
        arch.get_Integration(0).get_Profile(0,nn)[bb] = totport[nn,bb]

arch.set_dispersion_measure(0.0)
#arch.set_dedispersed(True)
arch.unload()
