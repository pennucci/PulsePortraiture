from pplib_dist import *
import os, sys, glob, pickle

def histo(data, label, xlabel, title=None, covar=False, normed=False,
        save=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(data, bins=bins, normed=normed)
    plt.xlabel(xlabel)
    mu = data.mean()
    std = data.std()
    muerr = std/np.sqrt(len(data))
    ax.text(0.7, 0.9, r"Mean %.2e $\pm$ %.2e"%(mu, muerr) +"\n" +
            r"Std. %.2e"%std, ha="center", va="center",
            transform=ax.transAxes)
    if title is not None: plt.title(title)
    if label == "covar" and covar is True:
        covar = np.mean((phis-phis.mean())*(DMs-DMs.mean()))
        plt.vlines(covar, plt.ylim()[0], plt.ylim()[1], colors='r')
        print "MC %s Covar and norm'd covar: %.3e %.3e"%(stem, covar,
                covar/(phis.std()*DMs.std()))
    if save:
        plt.savefig("%s_%s.png"%(stem, label))
        plt.close("all")
    else:
        plt.show()

def plot_big(fig, nn, data, label, xlabel, covar=False, normed=False):
    ax = fig.add_subplot(4,2,nn)
    ax.hist(data, bins=bins, normed=normed)
    plt.xlabel(xlabel)
    mu = data.mean()
    std = data.std()
    muerr = std/np.sqrt(len(data))
    ax.text(0.7, 0.9, r"Mean %.2e $\pm$ %.2e"%(mu, muerr) +"\n" +
            r"Std. %.2e"%std, ha="center", va="center",
            transform=ax.transAxes)
    if label == "covar" and covar is True:
        covar = np.mean((phis-phis.mean())*(DMs-DMs.mean()))
        plt.vlines(covar, plt.ylim()[0], plt.ylim()[1], colors='r')
        print "MC %s Covar and norm'd covar: %.3e %.3e"%(stem, covar,
                covar/(phis.std()*DMs.std()))

ephemfile = sys.argv[-1]
workdir = sys.argv[-2]
DM_inj = float(sys.argv[-3])
niter = int(sys.argv[-4])

stem = "%s/DM%.1e_%d"%(workdir, DM_inj, niter)

o = os.popen("grep F0 %s/%s"%(workdir, ephemfile))
P = o.readlines()
P = float(P[0].split()[1])**-1

phis = []
DMs = []
phierrs = []
DMerrs = []
dphis = []
dDMs = []
covars = []

pickfiles = glob.glob(stem+"*.pick")

for pf in pickfiles:
    data = pickle.load(open(pf, "r"))
    phis += list(data[0])
    DMs += list(data[1])
    phierrs += list(data[2])
    DMerrs += list(data[3])
    dphis += list(data[4])
    dDMs += list(data[5])
    covars += list(data[6])

phis = np.array(phis)
DMs = np.array(DMs)
phierrs = np.array(phierrs)
DMerrs = np.array(DMerrs)
dphis = np.array(dphis)
print "MC %s has %d dphi values >1."%(stem, len(np.where(dphis>1)[0]))
dphis %= 1
dphi_primes = np.where(dphis > 0.999, dphis-1, dphis)
dDMs = np.array(dDMs)
covars = np.array(covars)

bins = len(phis)/10

covar = False

if __name__ == "__main__":

    pickfile = open("%s_results.pick"%stem, "wb")
    pickle.dump([phis, DMs, phierrs, DMerrs, dphis, dDMs, covars], pickfile,
            protocol=2)
    pickfile.close()

    inf = open("%s.inf"%stem, "r").readlines()
    inf = [inf[xx].split() for xx in xrange(len(inf))]
    inf = dict(inf)
    for key in inf.keys():
        exec(key + " = float(inf['" + key + "'])")
    title = "nchanxnbin = %dx%d, nu0 = %.2f MHz, BW =%.2f MHz\n P = %.2f ms, DM0 = %.4f, DM_inj = %.5f, noise_std = %.2f, niter/node = %d"%(int(nchan), int(nbin), nu0, bw, P, DM0, DM_inj, noise_std, int(niter_per_node))

    histo(phis, "phis", "phi [rot]", title)
    histo(DMs, "DMs", "DM [cm**-3 pc]", title)
    histo(phierrs*P*1e9, "phierr", "Precision [ns]", title)
    histo(DMerrs, "DMerr", "DM err [cm**-3 pc]", title)
    histo(dphi_primes*P*1e9, "phidiff", "Phase Accurracy [ns]", title)
    histo(dDMs, "DMdiff", "DM Accuracy [cm**-3 pc]", title)
    histo(covars, "covar", "Phi-DM Covariance", title, covar=covar)

    plt.plot(phis, DMs, 'k+', ms=10)
    plt.xlabel("phi [rot]")
    plt.ylabel("DM [cm**-3 pc]")
    plt.title(title)
    plt.savefig("%s_phiDM.png"%stem)
    plt.close("all")

    fig = plt.figure(figsize=(17, 24))
    plot_big(fig, 1, phis, "phis", "phi [rot]")
    plot_big(fig, 2, DMs, "DMs", "DM [cm**-3 pc]")
    plot_big(fig, 3, phierrs*P*1e9, "phierr", "Precision [ns]")
    plot_big(fig, 4, DMerrs, "DMerr", "DM err [cm**-3 pc]")
    plot_big(fig, 5, dphi_primes*P*1e9, "phidiff", "Phase Accurracy [ns]")
    plot_big(fig, 6, dDMs, "DMdiff", "DM Accuracy [cm**-3 pc]")
    plot_big(fig, 7, covars, "covar", "Phi-DM Covariance", covar=covar)
    ax = fig.add_subplot(4,2,8)
    ax.plot(phis, DMs, 'k+', ms=10)
    plt.xlabel("phi [rot]")
    plt.ylabel("DM [cm**-3 pc]")
    plt.suptitle(title)
    plt.savefig("%s_results.png"%stem)
    plt.show()
    plt.close("all")

    os.system("rm %s_*_results.pick"%stem)
    os.system("mkdir %s"%stem)
    os.system("mv %s* %s/"%(stem, stem))
