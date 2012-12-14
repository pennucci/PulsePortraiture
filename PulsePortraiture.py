###
### This software lays on the bed of Procrustes all too comfortably.
###

import sys,time
from PPlib import *

class DataPortrait:
    """
    """
    def __init__(self,datafile,Gfudge=1.0):
        ""
        ""
        #Reading the data
        self.datafile = datafile
        self.Gfudge = Gfudge
        self.initial_model_run = False
        (self.source,self.arch,self.port,self.portx,self.noise_stdev,self.fluxprof,self.fluxprofx,self.prof,self.nbin,self.phases,self.nu0,self.bw,self.nchan,self.freqs,self.freqsx,self.nsub,self.P,self.MJD,self.weights,self.normweights,self.maskweights,self.portweights) = load_data(datafile,dedisperse=True,tscrunch=True,pscrunch=True,quiet=False,rm_baseline=(0,0),Gfudge=self.Gfudge)
        self.lofreq = self.freqs[0]-(self.bw/(2*self.nchan))
        self.init_params = []

    def fit_profile(self):
        """
        """
        fig = plt.figure()
        profplot = fig.add_subplot(211)
        interactor = GaussianSelector(profplot,self.prof,self.noise_stdev,self.datafile,minspanx=None,minspany=None,useblit=True)   #FIX self.noise_stdev is not the number you want
        plt.show()
        self.init_params = interactor.fit_params
        self.ngauss = (len(self.init_params) - 1)/3

    def show_data_portrait(self,fignum=None):
        """
        """
        if fignum: portfig = plt.figure(fignum)
        else: portfig = plt.figure()
        #plt.subplot(111)
        plt.xlabel("Phase [rot]")
        plt.ylabel("Frequency [MHz]")
        plt.title("%s Portrait"%self.source)
        plt.imshow(self.port,aspect="auto",origin="lower",extent=(0.0,1.0,self.freqs[0],self.freqs[-1]))
        #Need to make pretty, axes and what not
        plt.show()

    def flux_profile(self,guessA=1.0,guessalpha=0.0,fit=True,plot=True,quiet=False):
        """
        Will fit a power law across frequency in a portrait by bin scrunching.  This should be the usual average pulsar power-law spectrum.  The plot will show obvious scintles.
        """
        if fit:
            (params,param_errs,chi2,dof,residuals) = fit_powlaws(self.fluxprofx,self.freqsx,self.nu0,np.ones(len(self.fluxprofx)),np.array([guessA,guessalpha]),self.noise_stdev)
            if not quiet: print "Initial flux density power-law fit has residual mean %.2f and standard deviation %.2f for A = %.3f (flux at %.2f MHz) and alpha = %.3f"%(residuals.mean(),residuals.std(),params[0],self.nu0,params[1])
            if plot:
                if fit: plt.subplot(211)
                else: plt.subplot(111)
                plt.xlabel("Frequency [MHz]")
                plt.ylabel("Flux Units")
                plt.title("Average Flux Profile for %s"%self.source)
                if fit:
                    plt.plot(self.freqs,powlaw(self.freqs,self.nu0,params[0],params[1]),'k-')
                    plt.plot(self.freqsx,self.fluxprofx,'r+')
                    plt.subplot(212)
                    plt.xlabel("Frequency [MHz]")
                    plt.ylabel("Flux Units")
                    plt.title("Residuals")
                    plt.plot(self.freqsx,residuals,'r+')
                plt.show()
            return params[0],params[1]

    def show_residual_plot(self):
        """
        """
        try:            #FIX make smarter
            junq = self.model.shape
        except(AttributeError):
            print "No model portrait. Use make_gaussian_model_portrait()."
            return 0
        modelfig = plt.figure()
        plt.subplot(221)
        plt.title("Data Portrait")
        plt.imshow(self.port,aspect="auto",origin="lower",extent=(0.0,1.0,self.freqs[0],self.freqs[-1]))
        plt.subplot(222)
        plt.title("Model Portrait")
        plt.imshow(self.model,aspect="auto",origin="lower",extent=(0.0,1.0,self.freqs[0],self.freqs[-1]))
        plt.subplot(223)
        plt.title("Residuals")
        plt.imshow(self.port-self.modelmasked,aspect="auto",origin="lower",extent=(0.0,1.0,self.freqs[0],self.freqs[-1]))
        plt.colorbar()
        plt.subplot(224)
        plt.title(r"Log$_{10}$(abs(Residuals/Data))")
        plt.imshow(np.log10(abs(self.port-self.model)/self.port),aspect="auto",origin="lower",extent=(0.0,1.0,self.freqs[0],self.freqs[-1]))
        plt.colorbar()
        print "Residuals mean: %.3f"%(self.portx-self.modelx).mean()
        print "Residuals std:  %.3f"%(self.portx-self.modelx).std()
        print "Data std:       %.3f"%self.noise_stdev
        plt.show()

    def set_model_run(self):
        self.initial_model_run = True

    def make_gaussian_model_portrait(self,locparams=0.0,fixloc=True,widparams=0.0,fixwid=False,ampparams=0.0,fixamp=False,nu_ref=None,niter=0,writemodel=True,outfile=None,residplot=True,quiet=True):
        """
        """
        self.fix_params = np.array([fixloc,fixwid,fixamp])
        if not len(self.init_params): self.fit_profile()
        if type(locparams) is not np.ndarray:
            try: locparams = np.ones(self.ngauss)*locparams
            except ValueError:
                print "Not enough parameters for ngauss = %d."%self.ngauss
                return 0
        if type(widparams) is not np.ndarray:
            try: widparams = np.ones(self.ngauss)*widparams
            except ValueError:
                print "Not enough parameters for ngauss = %d."%self.ngauss
                return 0
        if type(ampparams) is not np.ndarray:
            try: ampparams = np.ones(self.ngauss)*ampparams
            except ValueError:
                print "Not enough parameters for ngauss = %d."%self.ngauss
                return 0
        if nu_ref is None: self.nu_ref = self.nu0
        if outfile is None: outfile=self.datafile+".model"
        self.init_model_params = np.empty([self.ngauss,6])
        for nn in range(self.ngauss):
            self.init_model_params[nn] = np.array([self.init_params[1::3][nn],locparams[nn],self.init_params[2::3][nn],widparams[nn],self.init_params[3::3][nn],ampparams[nn]])
        self.init_model_params = np.array([self.init_params[0]]+list(np.ravel(self.init_model_params)))
        itern = niter
        if niter < 0: niter = 0
        portx_noise = np.outer(get_noise(self.portx,chans=True),np.ones(self.nbin))
        if not self.initial_model_run:
            start = time.time()
            self.fit_params, self.fit_errs, self.chi_sq, self.dof = fit_gaussian_portrait(self.portx, portx_noise, self.init_model_params, self.fix_params, self.phases, self.freqsx, self.nu_ref, quiet=quiet)
            print "Fit took %.2f min"%((time.time()-start)/60.)
            niter += 1
        while(niter):
            if niter and self.initial_model_run:
                start = time.time()
                self.fit_params, self.fit_errs, self.chi_sq, self.dof = fit_gaussian_portrait(self.portx, portx_noise, self.model_params, self.fix_params, self.phases, self.freqsx, self.nu_ref, quiet=quiet)
                print "Fit took %.2f min"%((time.time()-start)/60.)
            self.model_params = self.fit_params
            self.model = gaussian_portrait(self.model_params,self.phases,self.freqs,self.nu_ref)
            self.modelmasked,self.modelx = screen_portrait(self.model,self.portweights)
            if residplot: self.show_residual_plot()      #FIX    Have it also show statistics of residuals
            niter -= 1
            dofit = 1
            if dofit == 1:
                phaseguess = first_guess(self.portx,self.modelx,nguess=5000)
                DMguess = 0.0
                phi,DM,nfeval,rc,scalesx,param_errs,red_chi2 = fit_portrait(self.portx,self.modelx,np.array([phaseguess,DMguess]),self.P,self.freqsx,self.nu0,scales=True,test=False)
                phierr = param_errs[0]
                DMerr = param_errs[1]
                print "Fit has phase offset of %.2e +/- %.2e [rot], DM of %.2e +/- %.2e [pc cm**-3], and red. chi**2 of %.2f."%(phi,phierr,DM,DMerr,red_chi2)
                if min(abs(phi),abs(1-phi)) < abs(phierr):
                    if abs(DM) < abs(DMerr):
                        print "Iteration converged."
                        phi = 0.0
                        DM = 0.0
                        niter = 0
                if niter:
                    print "Rotating data portrait by above values for iteration %d."%(itern-niter+1)
                    self.port = rotate_portrait(self.port,phi,DM,self.P,self.freqs,self.nu0)
                    self.portx = rotate_portrait(self.portx,phi,DM,self.P,self.freqsx,self.nu0)
                    self.set_model_run()
        if writemodel: write_model(outfile,self.source,self.model_params,self.nu_ref)    #FIX do not overwrite model file if exists...

class ModelPortrait_Gaussian:
    """
    """
    def __init__(self,modelfile,nbin,freqs,portweights=None,quiet=False,Gfudge=1.0):  #FIX make smarter to query DataPortrait...?
        """
        """
        self.modelfile = modelfile
        self.nbin = nbin
        self.freqs = freqs
        self.portweights = portweights
        self.Gfudge = Gfudge
        self.phases = np.arange(nbin, dtype='d')/nbin
        self.source,self.ngauss,self.model = make_model(modelfile,self.phases,self.freqs,quiet=quiet)
        if portweights is not None: self.modelmasked,self.modelx = screen_portrait(self.model,portweights)
        else: self.modelmasked,self.modelx = self.model,self.model

    def show_model_portrait(self,fignum=None):
        """
        """
        if fignum: portfig = plt.figure(fignum)
        else: portfig = plt.figure()
        #plt.subplot(111)
        plt.xlabel("Phase [rot]")
        plt.ylabel("Frequency [MHz]")
        plt.title("%s Model Portrait"%self.source)
        plt.imshow(self.model,aspect="auto",origin="lower",extent=(0.0,1.0,self.freqs[0],self.freqs[-1]))
        #Need to make pretty, axes and what not
        plt.show()

class ModelPortrait_Smoothed:
    """
    """
    def __init__(self,modelfile,quiet=False,Gfudge=1.0):    #FIX need to interpolate, or somehow account for missing channels in model...
        """
        """
        self.Gfudge = Gfudge
        self.modelfile = modelfile
        (self.source,self.arch,self.port,self.portx,self.noise_stdev,self.fluxprof,self.fluxprofx,self.prof,self.nbin,self.phases,self.nu0,self.bw,self.nchan,self.freqs,self.freqsx,self.nsub,self.P,self.MJD,self.weights,self.normweights,self.maskweights,self.portweights) = load_data(modelfile,dedisperse=True,tscrunch=True,pscrunch=True,quiet=False,rm_baseline=(0,0),Gfudge=self.Gfudge)
        self.model = self.port
        self.modelx = self.portx

class GetTOAs:
    """
    """
    def __init__(self,datafile,modelfile,mtype=None,DM0=None,bary_DM=True,one_DM=False,pam_cmd=False,outfile=None,errfile=None,mcmc=False,iters=20000,burn=10000,thin=100,starti=0,lsfit=True,write_TOAs=True,quiet=False,Gfudge=1.0):    #How much to thin? Burn?
        """
        """
        self.datafile=datafile
        self.modelfile=modelfile
        self.mtype=mtype
        self.outfile=outfile
        self.Gfudge=Gfudge
        (self.source,self.arch,self.ports,self.portxs,self.noise_stdev,self.fluxprof,self.fluxprofx,self.prof,self.nbin,self.phases,self.nu0,self.bw,self.nchan,self.freqs,self.freqsx,self.nsub,self.Ps,self.epochs,self.weights,self.normweights,self.maskweights,self.portweights) = load_data(datafile,dedisperse=False,tscrunch=False,pscrunch=True,quiet=False,rm_baseline=(0,0),Gfudge=self.Gfudge)
        self.MJDs = np.array([self.epochs[ii].in_days() for ii in xrange(self.nsub)],dtype=np.double)
        if DM0: self.DM0 = DM0
        else: self.DM0 = self.arch.get_dispersion_measure()
        print '\n'
        if self.mtype == "gauss": self.modelportrait=ModelPortrait_Gaussian(modelfile,self.nbin,self.freqs,portweights=None,quiet=False,Gfudge=1.0)
        elif self.mtype == "smooth": self.modelportrait=ModelPortrait_Smoothed(modelfile,quiet=False,Gfudge=self.Gfudge)
        else:
            print 'Model type must be either "gauss" or "smooth".'
            sys.exit()
        mp=self.modelportrait
        print '\n'
        #self.TOAs = np.empty(self.nsub,dtype=np.double)
        self.phis = np.empty(self.nsub,dtype=np.double)
        #self.TOAs_std = np.empty(self.nsub,dtype=np.double)
        self.phis_std = np.empty(self.nsub,dtype=np.double)
        self.DMs = np.empty(self.nsub,dtype=np.float)
        self.DMs_std = np.empty(self.nsub,dtype=np.float)
        self.scalesx = []
        if mcmc:
            import pymc as pm
            self.mcmc_params = dict(iters=iters,burn=burn,thin=thin,starti=starti)
            #self.TOAs_95 = np.empty([self.nsub,2],dtype=np.double)
            self.phis_95 = np.empty([self.nsub,2],dtype=np.double)
            self.DMs_95 = np.empty([self.nsub,2],dtype=np.float)
            self.phis_trace = np.empty([self.nsub,((iters-burn)/float(thin))-starti],dtype=np.double)     #FIX pymc db option
            self.DMs_trace = np.empty([self.nsub,((iters-burn)/float(thin))-starti],dtype=np.float)
            print "Doing the MCMC fit..."
            for nn in range(self.nsub):
                start = time.time()
                dataportrait = self.portxs[nn]
                noise0 = get_noise(dataportrait,frac=4,tau=True)     #FIX get_noise inccorect
                portx_fft = np.fft.rfft(dataportrait,axis=1)
                pw = self.portweights[nn]
                model,modelx = screen_portrait(mp.model,pw)
                model_fft = np.fft.rfft(modelx,axis=1)
                freqsx = ma.masked_array(self.freqs,mask=self.maskweights[nn]).compressed()
                if nn == 0:
                    phaseguess = first_guess(dataportrait,modelx,nguess=5000)    #FIX how does it tell the diff between say, +0.85 and -0.15
                    #if phaseguess > 0.5: phaseguess = 1-phaseguess  #FIX good fix?
                    print "Phase guess: %.8f"%phaseguess
                P = self.Ps[nn]
                MJD = self.MJDs[nn]
                #noise = pm.Normal('noise',mu=noise0,tau=noise0/4.0,value=noise0,plot=True)    #Noise prior
                #phi = pm.Uniform('phi',lower=-0.5,upper=0.5,value=phaseguess,plot=True)       #Phase prior      FIX (maybe -0.5--0.5??)
                phi = pm.Uniform('phi',lower=0.0,upper=1.0,value=phaseguess,plot=True)       #Phase prior      FIX (maybe -0.5--0.5??)
                #DMBeta = pm.Uniform('DM-Beta',lower=0.0,upper=10.0,value=5.0,plot=False)
                #DMBeta = pm.Uninformative('DM-Beta',value=5.0,plot=False)                   #Hyper parameter prior
                #DM = pm.Gamma('DM',alpha=1.0,beta=DMBeta,value=0.0,plot=True)              #Dispersion correction parameter prior
                upper_DM = 0.5*P/((self.lofreq**-2-(self.lofreq+self.bw)**-2)*Dconst)      #Let's say we can't be off by more than 50% in phase across band due to DM
                DM = pm.Uniform('DM',lower=0.0,upper=upper_DM,value=0.0,plot=True)                #Dispersion correction parameter prior
                #scales = pm.Uniform('scales',lower=0.0,upper=10.0,size=len(freqsx),value=ampguess*np.ones(len(freqsx)),plot=False)    #Scaling params prior
                scales = pm.Uniform('scales',lower=0.0,upper=10.0,size=len(freqsx),plot=False)    #Scaling params prior
                @pm.deterministic(plot=False)
                def portraitfitter_fft(modelportrait_fft=model_fft,phi=phi,DM=DM,scales=scales,freqs=freqsx,P=P):
                    Cdm = Dconst*DM/P
                    phasor = np.exp(np.transpose(np.transpose(np.array([np.arange(len(modelportrait_fft[0])) for x in xrange(len(freqs))]))*np.complex(0.0,-2*np.pi)*(phi+(Cdm/freqs**2))))     #NEGATIVE 2pi?
                    return np.transpose(scales*np.transpose(phasor*modelportrait_fft))
                fittedportrait = pm.Normal('fittedportrait',mu=portraitfitter_fft,tau=noise0,value=portx_fft,observed=True)
                #M = pm.MCMC([noise,phi,DMBeta,DM,scales,portraitfitter_fft,fittedportrait])
                #M = pm.MCMC([noise,phi,DMBeta,DM,portraitfitter_fft,fittedportrait])
                #M = pm.MCMC([phi,DMBeta,DM,scales,portraitfitter_fft,fittedportrait])
                M = pm.MCMC([phi,DM,scales,portraitfitter_fft,fittedportrait])
                M.sample(iter=iters,burn=burn,thin=thin)
                duration = time.time()-start
                self.M = M
                phinode = M.get_node("phi")
                DMnode = M.get_node("DM")
                scalesnode = M.get_node("scales")
                phi_med = np.median(phinode.trace[starti:])
                phi_95 = phinode.stats(start=starti)['95% HPD interval']
                DM_med = np.median(DMnode.trace[starti:])
                DM_95 = DMnode.stats(start=starti)['95% HPD interval']
                print "Finished TOA %d.  Took %.2f sec\t Median phase offset = %.8f rot ; Median DM = %.5f pc cm**-3"%(nn+1,duration,phi_med,DM_med)
                #self.TOAs[nn] = MJD + (phi_med*P)
                self.phis[nn] = phi_med
                #self.TOAs_95[nn] = phi_95*P
                self.phis_95[nn] = phi_95
                #self.TOAs_std[nn] = phinode.stats(start=starti)['standard deviation']
                self.phis_std[nn] = phinode.stats(start=starti)['standard deviation']
                self.phis_trace[nn] = phinode.trace()
                self.DMs[nn] = DM_med
                self.DMs_95[nn] = DM_95
                self.DMs_std[nn] = DMnode.stats(start=starti)['standard deviation']
                self.DMs_trace[nn] = DMnode.trace()
                self.scales.append(np.median(scalesnode.trace()))
        elif lsfit:
            if write_TOAs:      #FIX
                obs = self.arch.get_telescope()
                obs_codes = ["@","0","1","2"]
                obs = "1"
            print "Each of the %d TOAs are approximately %.2f s"%(self.nsub,self.arch.integration_length()/self.nsub)
            print "Doing Fourier-domain least-squares fit via chi_2 minimization...\n"  #FIX
            start = time.time()
            self.phis = np.empty(self.nsub)
            self.phi_errs = np.empty(self.nsub)
            self.DMs = np.empty(self.nsub)
            self.DM_errs = np.empty(self.nsub)
            self.nfevals = np.empty(self.nsub,dtype='int')
            self.rcs = np.empty(self.nsub,dtype='int')
            self.scales = np.empty([self.nsub,self.nchan])
            #These next two are lists becuase in principle, the subints could have different numbers of zapped channels.
            self.scalesx = []
            self.scale_errs = []
            self.red_chi2s = np.empty(self.nsub)
            for nn in range(self.nsub):
                dataportrait = self.portxs[nn]
                portx_fft = np.fft.rfft(dataportrait,axis=1)
                pw = self.portweights[nn]
                model,modelx = screen_portrait(mp.model,pw)
                freqsx = ma.masked_array(self.freqs,mask=self.maskweights[nn]).compressed()
                nu0 = self.nu0
                P = self.Ps[nn]
                MJD = self.MJDs[nn]
                ####################
                #DOPPLER CORRECTION#
                ####################
                #df = self.arch.get_Integration(nn).get_doppler_factor()
                #freqsx = correct_freqs_doppler(freqsx,df)
                #nu0 = correct_freqs_doppler(self.nu0,df)
                ####################
                if nn == 0:
                    #phaseguess,ampguess = first_guess(dataportrait,modelx,nguess=20)    #FIX how does it tell the diff between say, +0.85 and -0.15
                    #print "Phase and amplitude guesses %.5f %.5f"%(phaseguess, ampguess)
                    rot_dataportrait = rotate_portrait(self.portxs.mean(axis=0),0.0,self.DM0,P,freqsx,nu0)
                    #PSRCHIVE Dedisperses w.r.t. center of band...??
                    #if phaseguess > 0.5: phaseguess = phaseguess - 1    #FIX good fix?
                    phaseguess = first_guess(rot_dataportrait,modelx,nguess=5000)
                    #phaseguess = first_guess(dataportrait,modelx,nguess=5000)
                    #self.DM0 = 0.0
                    DMguess = self.DM0
                    if not quiet: print "Phase guess: %.8f ; DM guess: %.5f"%(phaseguess,DMguess)
                #else:   #To first order this only speeds things up marginally, same answers found, unless it breaks...
                #    phaseguess = self.phis[nn-1]    #FIX Might not be a good idea if RFI or something throws it completely off, whereas first phaseguess only depends on pulse profile...
                #    DMguess = self.DMs[nn-1]
                #if not quiet: print "Phase guess: %.8f ; DM guess: %.5f"%(phaseguess,DMguess)
                #NEED status bar?
                print "Fitting for TOA %d...put more info here"%(nn+1)      #FIX
                phi,DM,nfeval,rc,scalex,param_errs,red_chi2 = fit_portrait(self.portxs[nn],modelx,np.array([phaseguess,DMguess]),P,freqsx,nu0,scales=True,test=False)
                self.phis[nn] = phi
                self.phi_errs[nn] = param_errs[0]
                ####################
                #DOPPLER CORRECTION#
                ####################
                if bary_DM:
                    #NB: the 'doppler factor' retrieved below seems to be the inverse of the convention nu_source/nu_observed
                    df = self.arch.get_Integration(nn).get_doppler_factor()
                    DM *= df
                self.DMs[nn] = DM
                self.DM_errs[nn] = param_errs[1]
                self.nfevals[nn] = nfeval
                self.rcs[nn] = rc
                self.scalesx.append(scalex)
                self.scale_errs.append(param_errs[2:])
                scale = np.zeros(self.nchan)
                ss = 0
                for ii in range(self.nchan):
                    if self.normweights[nn,ii] == 1:
                        scale[ii] = scalex[ss]
                        ss += 1
                    else: pass
                self.scales[nn] = scale
                self.red_chi2s[nn] = red_chi2
            self.DeltaDMs = self.DMs - self.DM0
            self.DeltaDM_mean,self.DeltaDM_var = np.average(self.DeltaDMs,weights=self.DM_errs**-2,returned=True)   #Returns the weighted mean and the sum of the weights, need to do better than this in case of small-error outliers from RFI, etc.  Last TOA may mess things up...Median...then...
            self.DeltaDM_var = self.DeltaDM_var**-1
            if self.nsub > 1: self.DeltaDM_var *= np.sum(((self.DeltaDMs-self.DeltaDM_mean)**2)/(self.DM_errs**2))/(len(self.DeltaDMs)-1)    #Multiplying by the chi-squared...
            self.DeltaDM_err = self.DeltaDM_var**0.5
            if write_TOAs:
                toas = [self.epochs[nn] + pr.MJD((self.phis[nn]*self.Ps[nn])/(3600*24.)) for nn in xrange(self.nsub)]
                #toas = self.epochs + pr.MJD((self.phis*self.Ps)/(3600*24.))
                #toa_errs = [np.array(self.param_errs[nn,0])*self.Ps[nn]*1e6 for nn in xrange(self.nsub)]
                toa_errs = self.phi_errs*self.Ps*1e6
                if self.outfile: sys.stdout = open(self.outfile,"a")
                #Should write which freqs? topo or bary?
                #Have option for different kinds of TOA output
                for nn in range(self.nsub):
                    if one_DM:
                        write_princeton_toa(toas[nn].intday(),toas[nn].fracday(),toa_errs[nn],self.nu0,self.DeltaDM_mean,obs=obs)
                    else:
                        write_princeton_toa(toas[nn].intday(),toas[nn].fracday(),toa_errs[nn],self.nu0,self.DeltaDMs[nn],obs=obs)
            sys.stdout = sys.__stdout__
            duration = time.time()-start
            print "\nFitting took %.1f min, ~%.3f min/TOA, mean TOA error is %.3f us"%(duration/60.,duration/(60*self.nsub),self.phi_errs.mean()*self.Ps.mean()*1e6)
            if pam_cmd:
                pc = open("pam_cmds","a")
                pam_ext = self.datafile[-self.datafile[::-1].find("."):]+".rot"
                self.phi_mean,self.phi_var = np.average(self.phis,weights=self.phi_errs**-2,returned=True)   #Returns the weighted mean and the sum of the weights, need to do better than this in case of small-error outliers from RFI, etc.  Last TOA may mess things up...Median...then...
                self.phi_var = self.phi_var**-1
                pc.write("pam -e %s -r %.7f -d %.5f %s\n"%(pam_ext,self.phi_mean,self.DeltaDM_mean+self.DM0,self.datafile))
                pc.close()
            if errfile:
                ef = open(errfile,"a")
                for nn in range(self.nsub):
                    ef.write("%.5e\n"%self.DM_errs[nn])
                    #if nn != self.nsub-1: ef.write("%.5e\n"%self.DM_errs[nn])
                    #else: ef.write("%.5e"%self.DM_errs[nn])
        else:
            print "Invalid."
    def show_subint(self,subint,fignum=None):
        """
        subint 0 = python index 0
        """
        if fignum: portfig = plt.figure(fignum)
        else: portfig = plt.figure()
        #plt.subplot(111)
        ii = subint
        plt.xlabel("Phase [rot]")
        plt.ylabel("Frequency [MHz]")
        plt.title("Subint %d"%(subint))
        plt.imshow(self.ports[ii],aspect="auto",origin="lower",extent=(0.0,1.0,self.freqs[0],self.freqs[-1]))
        #Need to make pretty, axes and what not
        plt.show()

    def show_fit(self,subint=0,fignum=None):
        """
        subint 0 = python index 0
        """
        if fignum: portfig = plt.figure(fignum)
        else: fitfig = plt.figure()
        ii = subint
        phi = self.phis[ii]
        DM = self.DMs[ii]
        scales = self.scales[ii]
        scalesx = self.scalesx[ii]
        freqs = self.freqs
        freqsx = self.freqsx
        nu0 = self.nu0
        P = self.Ps[ii]
        port = self.ports[ii]
        portx = self.portxs[ii]
        model,modelx = screen_portrait(self.modelportrait.model,self.portweights[ii])
        #modelmasked = 
        fitmodel = np.transpose(self.scales[ii]*np.transpose(rotate_portrait(model,-phi,-DM,P,freqs,nu0)))
        fitmodelx = np.transpose(self.scalesx[ii]*np.transpose(rotate_portrait(modelx,-phi,-DM,P,freqsx,nu0)))
        #fitmodelmasked = rotate_portrait(modelmasked,-np.mean((M.phi.trace()[starti:])),-10**np.mean((M.DM.trace()[starti:])),freqs,nu0)
        plt.subplot(221)
        plt.title("Data Portrait")
        plt.imshow(port,aspect="auto",origin="lower",extent=(0.0,1.0,self.freqs[0],self.freqs[-1])) #WRONG EXTENT
        plt.subplot(222)
        plt.title("Fitted Model Portrait")
        plt.imshow(fitmodel,aspect="auto",origin="lower",extent=(0.0,1.0,self.freqs[0],self.freqs[-1]))
        plt.subplot(223)
        plt.title("Residuals")
        plt.imshow(port-fitmodel,aspect="auto",origin="lower",extent=(0.0,1.0,self.freqs[0],self.freqs[-1]))
        plt.colorbar()
        plt.subplot(224)
        plt.title(r"Log$_{10}$(abs(Residuals/Data))")
        plt.imshow(np.log10(abs(port-fitmodel)/port),aspect="auto",origin="lower",extent=(0.0,1.0,self.freqs[0],self.freqs[-1]))
        plt.colorbar()
        plt.show()

    def show_results(self,fignum=None):
        """
        """
        cols = ['b','k','g','b','r']
        if fignum: fig = plt.figure(fignum)
        else: fig = plt.figure()
        pf = np.polynomial.polynomial.polyfit
        fit_results = pf(self.MJDs,self.phis*self.Ps*1e3,1,full=True,w=self.phi_errs**-2)      #FIX not sure weighting works...
        resids = (self.phis*self.Ps*1e3)-(fit_results[0][0]+(fit_results[0][1]*self.MJDs))
        resids_mean,resids_var = np.average(resids,weights=self.phi_errs**-2,returned=True)
        resids_var = resids_var**-1
        if self.nsub > 1: resids_var *= np.sum(((resids-resids_mean)**2)/(self.phi_errs**2))/(len(resids)-1)
        resids_err = resids_var**0.5
        RMS = resids_err
        #RMS = np.sum(resids**2/len(resids))**0.5        #FIX  check this, RMS seems too high
        ax1 = fig.add_subplot(311)
        #ax1.errorbar(self.MJDs,self.phis,[self.phi_errs[xx] for xx in xrange(len(self.phis))],color='k',fmt='+')
        for nn in range(len(self.phis)):
            ax1.errorbar(self.MJDs[nn],self.phis[nn]*self.Ps[nn]*1e6,self.phi_errs[nn]*self.Ps[nn]*1e6,color='%s'%cols[self.rcs[nn]],fmt='+')
        plt.plot(self.MJDs,(fit_results[0][0]+(fit_results[0][1]*self.MJDs))*1e3,"m--")
        plt.xlabel("MJD")
        plt.ylabel(r"Offset [$\mu$s]")
        ax1.text(0.1,0.9,"%.2e ms/s"%(fit_results[0][1]/(3600*24)),ha='center',va='center',transform=ax1.transAxes)
        ax2 = fig.add_subplot(312)
        for nn in range(len(self.phis)):
            ax2.errorbar(self.MJDs[nn],resids[nn]*1e3,self.phi_errs[nn]*self.Ps[nn]*1e6,color='%s'%cols[self.rcs[nn]],fmt='+')
        plt.plot(self.MJDs,np.ones(len(self.MJDs))*resids_mean*1e3,"m--")
        xverts=np.array([self.MJDs[0],self.MJDs[0],self.MJDs[-1],self.MJDs[-1]])
        yverts=np.array([resids_mean-resids_err,resids_mean+resids_err,resids_mean+resids_err,resids_mean-resids_err])*1e3
        plt.fill(xverts,yverts,"m",alpha=0.25,ec='none')
        plt.xlabel("MJD")
        plt.ylabel(r"Offset [$\mu$s]")
        ax2.text(0.1,0.9,r"$\sim$weighted RMS = %d ns"%int(resids_err*1e6),ha='center',va='center',transform=ax2.transAxes)
        ax3 = fig.add_subplot(313)
        #ax3.errorbar(self.MJDs,self.DMs,[self.DM_errs[xx] for xx in xrange(len(self.DMs))],color='k',fmt='+')
        for nn in range(len(self.phis)):
            ax3.errorbar(self.MJDs[nn],self.DMs[nn],self.DM_errs[nn],color='%s'%cols[self.rcs[nn]],fmt='+')
        if abs(self.DeltaDM_mean)/self.DeltaDM_err < 10: plt.plot(self.MJDs,np.ones(len(self.MJDs))*self.DM0,"r-")
        plt.plot(self.MJDs,np.ones(len(self.MJDs))*(self.DeltaDM_mean+self.DM0),"m--")
        xverts=[self.MJDs[0],self.MJDs[0],self.MJDs[-1],self.MJDs[-1]]
        yverts=[self.DeltaDM_mean+self.DM0-self.DeltaDM_err,self.DeltaDM_mean+self.DM0+self.DeltaDM_err,self.DeltaDM_mean+self.DM0+self.DeltaDM_err,self.DeltaDM_mean+self.DM0-self.DeltaDM_err]
        plt.fill(xverts,yverts,"m",alpha=0.25,ec='none')
        plt.xlabel("MJD")
        plt.ylabel(r"DM [pc cm$^{3}$]")
        ax3.text(0.15,0.9,r"$\Delta$ DM = %.2e $\pm$ %.2e"%(self.DeltaDM_mean,self.DeltaDM_err),ha='center',va='center',transform=ax3.transAxes)
        plt.show()

    def show_hists(self):
        cols = ['b','k','g','b','r']
        bins = self.nfevals.max()
        binmin = self.nfevals.min()
        rc1=np.zeros(bins-binmin+1)
        rc2=np.zeros(bins-binmin+1)
        rc4=np.zeros(bins-binmin+1)
        rc5=np.zeros(bins-binmin+1)
        for nn in range(len(self.nfevals)):
            nfeval = self.nfevals[nn]
            if self.rcs[nn] == 1: rc1[nfeval-binmin] += 1
            elif self.rcs[nn] == 2: rc2[nfeval-binmin] += 1
            elif self.rcs[nn] == 4: rc4[nfeval-binmin] += 1
            else:
                print "rc %d discovered!"%self.rcs[nn]
                rc5[nfevals-binmin] += 1
        width = 1
        b1 = plt.bar(np.arange(binmin-1,bins)+0.5,rc1,width,color=cols[1])
        b2 = plt.bar(np.arange(binmin-1,bins)+0.5,rc2,width,color=cols[2],bottom=rc1)
        b4 = plt.bar(np.arange(binmin-1,bins)+0.5,rc4,width,color=cols[4],bottom=rc1+rc2)
        if rc5.sum() != 0:
            b5 = plt.bar(np.arange(binmin-1,bins)+0.5,rc5,width,color=cols[3],bottom=rc1+rc2+rc4)
            plt.legend((b1[0],b2[0],b4[0],b5[0]),('rc=1','rc=2','rc=4','rc=#'))
        else: plt.legend((b1[0],b2[0],b4[0]),('rc=1','rc=2','rc=4'))
        plt.xlabel("nfevals")
        plt.ylabel("counts")
        plt.xticks(np.arange(1,bins+1))
        if rc5.sum() != 0: plt.axis([binmin-width/2.0,bins+width/2.0,0,max(rc1+rc2+rc4+rc5)])
        else: plt.axis([binmin-width/2.0,bins+width/2.0,0,max(rc1+rc2+rc4)])
        plt.figure(2)
        urcs = np.unique(self.rcs)
        rchist = np.histogram(self.rcs,bins=len(urcs))[0]
        for uu in range(len(urcs)):
            rc = urcs[uu]
            plt.hist(np.ones(rchist[uu])*rc,bins=1,color=cols[rc])
            plt.xlabel("rc")
            plt.ylabel("counts")
        plt.show()

#class MakeModelPortrait:
#    """
#    """
#    def __init__(self,datafiles=None,metafile=None,nu_ref=np.inf):
#        if datafiles and metafile:
#            print "Too many file options specified."
#            sys.exit()
#        if metafile:
#            datafiles = open(metafile,"r").readlines()
#        self.nepoch = len(datafiles)
#        freqs = []
#        ports = []
#        phaseguess = []
#        DMguess = []
#        scalesguess = []
#        for nn in range(self.nepoch):
#            (source,arch,port,portx,noise_stdev,fluxprof,fluxprofx,prof,nbin,phases,nu0,bw,nchan,freqs,freqsx,nsub,P,MJD,weights,normweights,maskweights,portweights) = load_data(datafiles[nn],dedisperse=False,tscrunch=True,pscrunch=True,quiet=False,rm_baseline=(0,0),Gfudge=self.Gfudge)
#            freqs.append(freqsx)
#            ports.append(portx)
#            if nn == 0:
#                port_ref = portx
#                phaseguess.append(0.0)
#            else:
#                phaseguess.append(first_guess(portx,port_ref,nguess=1000))
#            DMguess.append(arch.get_dispersion_measure())
#            if nn == 0:
#                scales_ref = prof.max()
#            scalesguess.append(np.ones(len(freqsx))*prof.max()/scales_ref)

