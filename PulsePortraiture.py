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
        (self.source,self.arch,self.port,self.portx,self.noise_stdev,self.fluxprof,self.fluxprofx,self.prof,self.nbin,self.phases,self.nu0,self.bw,self.nchan,self.chanwidth,self.lofreq,self.freqs,self.freqsx,self.nsub,self.P,self.MJD,self.weights,self.normweights,self.maskweights,self.portweights) = load_data(datafile,dedisperse=True,tscrunch=True,pscrunch=True,quiet=False,rm_baseline=(0,0),Gfudge=self.Gfudge)

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

    def show_evol_plots(self):
        """
        """
        try:            #FIX make smarter, or in case evol plots are needed for just modelfile
            print "Plotting component evolution for %d fitted subbands..."%self.nsubfit
        except(AttributeError):     #__getattr__ __getattribute__?
            print "Model portrait has not been made yet. Use make_model_portrait()."
            return 0
        subfitparams = self.subfitparams
        subfreqs = self.subfreqs
        nsubfit = self.nsubfit
        dcfig = plt.figure(1)
        plt.title("DC parameter")
        plt.plot(subfitparams[:,0],'k')
        phasefig = plt.figure(2)
        for gg in xrange(self.ngauss):
            plt.errorbar(subfitparams[:,1::3][:,gg],subfreqs,fmt='%s-'%cols[gg],xerr=subfitparams[:,2::3][:,gg]/2,ecolor='%s'%cols[gg])
            for hh in xrange(nsubfit):
                plt.plot(subfitparams[:,1::3][:,gg][hh],subfreqs[hh],'%so'%cols[gg],ms=30*subfitparams[:,3::3][:,gg][hh]/subfitparams[:,3::3].max())
        plt.title("Component Evolution")
        plt.xlabel("Phase [rot]")
        plt.ylabel("Frequency [MHz]")
        fwhmfig = plt.figure(3)
        for gg in xrange(self.ngauss):
            plt.plot(subfitparams[:,2::3][:,gg],subfreqs,'%s-'%cols[gg])
        plt.title("Component Width Evolution")
        plt.xlabel("Width [rot]")
        plt.ylabel("Frequency [MHz]")
        amplfig = plt.figure(4)
        for gg in xrange(self.ngauss):
            plt.plot(subfitparams[:,3::3][:,gg],subfreqs,'%s-'%cols[gg])
        plt.title("Component Height Evolution")
        plt.xlabel("Flux Density [mJy-ish]")
        plt.ylabel("Frequency [MHz]")
        plt.show()

    def show_PL_plots(self):
        """
        """
        try:            #FIX make smarter, or in case PL plots are needed with just modelfile
            print "Plotting component power-law evolution for %d fitted subbands..."%self.nsubfit
        except(AttributeError):
            print "Model portrait has not been made yet. Use make_model_portrait()."
            return 0
        subfitparams = self.subfitparams
        subfreqs = self.subfreqs
        As = self.As
        alphas = self.alphas
        for gg in xrange(self.ngauss):
            plt.plot(subfreqs,subfitparams[:,3::3][:,gg],'%s+'%cols[gg])
            plt.plot(subfreqs,powlaw(subfreqs,self.nu0,As[gg],alphas[gg]),'%s'%cols[gg])
            plt.title("Component Height Power-Law Fits")
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("Flux Density [mJy-ish]")
        plt.show()

    def show_residual_plot(self):
        """
        """
        try:            #FIX make smarter
            junq = self.model.shape
            print "Plotting portrait residuals..."
        except(AttributeError):
            print "No model portrait.  Use make_model_portrait() or make_model()."
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
        plt.show()

    def make_Gaussian_model_portrait(self,nsubfit=8,guessA=1.0,guessalpha=0.0,niter=0,nuspacing=True,makemovie=False,writemodel=True,outfile=None,subfitplots=False,evolplots=False,PLplots=False,residplot=True,showall=False,shownone=False):
        """
        """
        if outfile is None: outfile=self.datafile+".model"
        itern = niter
        if niter < 0: niter=0
        niter += 1
        while(niter):
            niter -= 1
            print "Fitting %i components across each of %i subbands..."%(self.ngauss,nsubfit)
            if showall:
                subfitplots=True
                evolplots=True
                PLplots=True
                residplot=True
            if shownone:
                subfitplots=False
                evolplots=False
                PLplots=False
                residplot=False
            if nuspacing:                   #FIX this should probably be double checked
                subfreqs = powlawnus(self.lofreq,self.lofreq+self.bw,nsubfit,self.flux_profile(plot=False,quiet=True)[1],mid=False)
                subfreqbins = np.zeros(len(subfreqs))
                for ii in xrange(len(subfreqs)):
                    subfreqbins[ii] = np.argmin(np.abs(subfreqs[ii]-self.freqs))
                scrport = np.zeros([nsubfit,self.nbin])
                for ii in xrange(nsubfit):
                    norm = self.normweights[subfreqbins[ii]:subfreqbins[ii+1]].sum()
                    scrport[ii] = np.sum(self.port[subfreqbins[ii]:subfreqbins[ii+1]],axis=0)/norm
                subfreqs = powlawnus(self.lofreq,self.lofreq+self.bw,nsubfit,self.flux_profile(plot=False,quiet=True)[1],mid=True)
            else:
                subfreqs = powlawnus(self.lofreq,self.lofreq+self.bw,nsubfit,0.0,mid=False)
                subfreqbins = np.zeros(len(subfreqs))
                for ii in xrange(len(subfreqs)):
                    subfreqbins[ii] = np.argmin(np.abs(subfreqs[ii]-self.freqs))
                scrport = np.zeros([nsubfit,self.nbin])
                for ii in xrange(nsubfit):
                    norm = self.normweights[subfreqbins[ii]:subfreqbins[ii+1]].sum()
                    scrport[ii] = np.sum(self.port[subfreqbins[ii]:subfreqbins[ii+1]],axis=0)/norm
                subfreqs = powlawnus(self.lofreq,self.lofreq+self.bw,nsubfit,0.0,mid=True)
                #subfreqs = powlawnus(self.lofreq,self.lofreq+self.bw,nsubfit,0.0,mid=True)
                #self.arch.fscrunch_to_nchan(nsubfit)
                #scrport = self.arch.get_data()[0][0]*self.Gfudge
            ymax = scrport.ravel().max()*1.15
            ymin = scrport.ravel().min()*1.15
            subfitparams = np.zeros([nsubfit,len(self.init_params)])
            if makemovie: moviefiles=[]
            for ii in xrange(nsubfit):
                #Need better noise estimate
                (params,param_errs,chi2,dof,residuals) = fit_gaussians(scrport[ii],self.init_params,self.noise_stdev,self.datafile,quiet=1)
                subfitparams[ii] = params
                #To convert Scott's amplitude parameter to the amplitude at the mean...     #FIX needed? already done?
                #subfitparams[ii][3::3] *= (2*np.pi*((subfitparams[ii][2::3]/(2*np.sqrt(2*np.log(2))))**2))**-0.5     #FIX needed? already done?
                plt.figure(ii)
                plt.subplot(211)
                plt.title("%s Frequency Evolution"%self.source)
                plt.ylabel("Flux Density-ish [mJy]")
                plt.ylim(ymin,ymax)
                plt.text(0.75,ymax*0.8,'%.2f Hz'%subfreqs[ii])
                plt.plot(self.phases,scrport[ii],'k')
                DC = params[0]
                for gg in xrange(self.ngauss):
                    phase, FWHM, amp = params[1+gg*3:4+gg*3]
                    plt.plot(self.phases, DC + amp*gaussian_profile(self.nbin,phase,FWHM),'%s'%cols[gg])
                fitprof = gen_gaussians(params,self.nbin)
                plt.plot(self.phases,fitprof,'r--',lw=1)
                plt.subplot(212)
                plt.plot(self.phases,residuals,'k')
                plt.xlabel("Phase [rot]")
                plt.ylabel("Data-Fit Residuals")
                print "Subband %d fit done..."%(ii+1)
                if makemovie:
                    fname = '_tmp%03d.png'%(ii+1)
                    plt.savefig(fname)
                    moviefiles.append(fname)
            if makemovie:
                import os
                os.system("mencoder -really-quiet 'mf://_tmp*.png' -mf type=png:fps=1 -o %s_evol.avi -ovc lavc -lavcopts vcodec=mpeg4"%self.source)
                for fname in moviefiles:
                    os.remove(fname)
            if subfitplots: plt.show()
            else: plt.close('all')
            self.nsubfit = nsubfit
            self.subfreqs = subfreqs
            self.subfreqbins = subfreqbins
            self.scrport = scrport
            self.subfitparams = subfitparams
            if evolplots: self.show_evol_plots()
            As = []
            alphas = []
            for gg in xrange(self.ngauss):
                (params,param_errs,chi2,dof,residuals) = fit_powlaws(subfitparams[:,3::3][:,gg],subfreqs,self.nu0,np.ones(nsubfit),[guessA,guessalpha],self.noise_stdev)
                A,alpha = params[0],params[1]
                As.append(A)
                alphas.append(alpha)
            self.As = As
            self.alphas = alphas
            if PLplots: self.show_PL_plots()
            modelparams = np.empty(self.ngauss*3+1)
            modelparams[0] = np.median(subfitparams[:,0])      #FIX make smarter...way smarter...
            for gg in xrange(self.ngauss):
                modelparams[1::3][gg] = np.median(subfitparams[:,1::3][:,gg])
                modelparams[2::3][gg] = np.median(subfitparams[:,2::3][:,gg])
                modelparams[3::3][gg] = np.median(subfitparams[:,3::3][:,gg])
            model = make_model(self.phases,self.freqs,None,modelparams,As,alphas,self.nu0)
            modelmasked,modelx = screen_portrait(model,self.portweights)
            self.modelparams = modelparams
            self.model = model
            self.modelmasked = modelmasked
            self.modelx = modelx
            if residplot: self.show_residual_plot()      #FIX    Have it also show statistics of residuals
            if writemodel: write_model(outfile,self.source,modelparams,As,alphas,self.nu0)    #FIX do not overwrite model file if exists...
            dofit = 1
            if dofit == 1:
                phaseguess = first_guess(self.portx,self.modelx,nguess=5000)
                Ddmguess = 0.0
                phi,Ddm,nfeval,rc,scalesx,param_errs,red_chi2 = fit_portrait(self.portx,self.modelx,np.array([phaseguess,Ddmguess]),self.P,self.freqsx,self.nu0,scales=True)
                phierr = param_errs[0]
                Ddmerr = param_errs[1]
                print "Fit has phase offset of %.2e +/- %.2e [rot], Ddm of %.2e +/- %.2e [pc cm**-3], and red. chi**2 of %.2f."%(phi,phierr,Ddm,Ddmerr,red_chi2)
                if min(abs(phi),abs(1-phi)) < abs(phierr):
                    if abs(Ddm) < abs(Ddmerr):
                        print "Iteration converged."
                        phi = 0.0
                        Ddm = 0.0
                        niter = 0
                if niter:
                    print "Rotating portrait by above values for iteration %d."%(itern-niter+1)
                    self.port = rotate_portrait(self.port,phi,Ddm,self.P,self.freqs,self.nu0)
                    self.portx = rotate_portrait(self.portx,phi,Ddm,self.P,self.freqsx,self.nu0)

class ModelPortrait_Gaussian:
    """
    """
    def __init__(self,modelfile,nchan,nbin,bw,lofreq,portweights=None,quiet=False,Gfudge=1.0):  #FIX make smarter to query DataPortrait...?
        """
        """
        self.modelfile = modelfile
        self.nchan =  nchan
        self.nbin = nbin
        self.bw = bw
        self.lofreq = lofreq
        self.portweights = portweights
        self.Gfudge = Gfudge
        self.chanwidth = bw/float(nchan)
        self.freqs = np.linspace(lofreq+(self.chanwidth/2.0),lofreq+bw-(self.chanwidth/2.0),nchan)     #Centers of frequency channels
        self.phases = np.arange(nbin, dtype='d')/nbin
        self.source,self.ngauss,self.refparams,self.As,self.alphas,self.nu0,self.model = make_model(self.phases,self.freqs,modelfile,quiet=quiet)
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
        (self.source,self.arch,self.port,self.portx,self.noise_stdev,self.fluxprof,self.fluxprofx,self.prof,self.nbin,self.phases,self.nu0,self.bw,self.nchan,self.chanwidth,self.lofreq,self.freqs,self.freqsx,self.nsub,self.P,self.MJD,self.weights,self.normweights,self.maskweights,self.portweights) = load_data(modelfile,dedisperse=True,tscrunch=True,pscrunch=True,quiet=False,rm_baseline=(0,0),Gfudge=self.Gfudge)
        self.model = self.port
        self.modelx = self.portx

class GetTOAs:
    """
    """
    def __init__(self,datafile,modelfile,mtype=None,outfile=None,mcmc=False,iters=20000,burn=10000,thin=100,starti=0,lsfit=True,write_TOAs=True,quiet=False,Gfudge=1.0):    #How much to thin? Burn?
        """
        """
        self.datafile=datafile
        self.modelfile=modelfile
        self.mtype=mtype
        self.outfile=outfile
        self.Gfudge=Gfudge
        (self.source,self.arch,self.ports,self.portxs,self.noise_stdev,self.fluxprof,self.fluxprofx,self.prof,self.nbin,self.phases,self.nu0,self.bw,self.nchan,self.chanwidth,self.lofreq,self.freqs,self.freqsx,self.nsub,self.Ps,self.epochs,self.weights,self.normweights,self.maskweights,self.portweights) = load_data(datafile,dedisperse=True,tscrunch=False,pscrunch=True,quiet=False,rm_baseline=(0,0),Gfudge=self.Gfudge)
        self.MJDs = np.array([self.epochs[ii].in_days() for ii in xrange(self.nsub)],dtype=np.double)
        print '\n'
        if self.mtype == "gauss": self.modelportrait=ModelPortrait_Gaussian(modelfile,self.nchan,self.nbin,self.bw,self.lofreq,portweights=None,quiet=False,Gfudge=1.0)
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
        self.Ddms = np.empty(self.nsub,dtype=np.float)
        self.Ddms_std = np.empty(self.nsub,dtype=np.float)
        self.scalesx = []
        if mcmc:
            import pymc as pm
            self.mcmc_params = dict(iters=iters,burn=burn,thin=thin,starti=starti)
            #self.TOAs_95 = np.empty([self.nsub,2],dtype=np.double)
            self.phis_95 = np.empty([self.nsub,2],dtype=np.double)
            self.Ddms_95 = np.empty([self.nsub,2],dtype=np.float)
            self.phis_trace = np.empty([self.nsub,((iters-burn)/float(thin))-starti],dtype=np.double)     #FIX pymc db option
            self.Ddms_trace = np.empty([self.nsub,((iters-burn)/float(thin))-starti],dtype=np.float)
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
                #DdmBeta = pm.Uniform('Ddm-Beta',lower=0.0,upper=10.0,value=5.0,plot=False)
                #DdmBeta = pm.Uninformative('Ddm-Beta',value=5.0,plot=False)                   #Hyper parameter prior
                #Ddm = pm.Gamma('Ddm',alpha=1.0,beta=DdmBeta,value=0.0,plot=True)              #Dispersion correction parameter prior
                upper_Ddm = 0.5*P/((self.lofreq**-2-(self.lofreq+self.bw)**-2)*Dconst)      #Let's say we can't be off by more than 50% in phase across band due to Ddm
                Ddm = pm.Uniform('Ddm',lower=0.0,upper=upper_Ddm,value=0.0,plot=True)                #Dispersion correction parameter prior
                #scales = pm.Uniform('scales',lower=0.0,upper=10.0,size=len(freqsx),value=ampguess*np.ones(len(freqsx)),plot=False)    #Scaling params prior
                scales = pm.Uniform('scales',lower=0.0,upper=10.0,size=len(freqsx),plot=False)    #Scaling params prior
                @pm.deterministic(plot=False)
                def portraitfitter_fft(modelportrait_fft=model_fft,phi=phi,Ddm=Ddm,scales=scales,freqs=freqsx,P=P):
                    Cdm = Dconst*Ddm/P
                    phasor = np.exp(np.transpose(np.transpose(np.array([np.arange(len(modelportrait_fft[0])) for x in xrange(len(freqs))]))*np.complex(0.0,-2*np.pi)*(phi+(Cdm/freqs**2))))     #NEGATIVE 2pi?
                    return np.transpose(scales*np.transpose(phasor*modelportrait_fft))
                fittedportrait = pm.Normal('fittedportrait',mu=portraitfitter_fft,tau=noise0,value=portx_fft,observed=True)
                #M = pm.MCMC([noise,phi,DdmBeta,Ddm,scales,portraitfitter_fft,fittedportrait])
                #M = pm.MCMC([noise,phi,DdmBeta,Ddm,portraitfitter_fft,fittedportrait])
                #M = pm.MCMC([phi,DdmBeta,Ddm,scales,portraitfitter_fft,fittedportrait])
                M = pm.MCMC([phi,Ddm,scales,portraitfitter_fft,fittedportrait])
                M.sample(iter=iters,burn=burn,thin=thin)
                duration = time.time()-start
                self.M = M
                phinode = M.get_node("phi")
                Ddmnode = M.get_node("Ddm")
                scalesnode = M.get_node("scales")
                phi_med = np.median(phinode.trace[starti:])
                phi_95 = phinode.stats(start=starti)['95% HPD interval']
                Ddm_med = np.median(Ddmnode.trace[starti:])
                Ddm_95 = Ddmnode.stats(start=starti)['95% HPD interval']
                print "Finished TOA %d.  Took %.2f sec\t Median phase offset = %.8f rot ; Median deltaDM = %.5f pc cm**-3"%(nn+1,duration,phi_med,Ddm_med)
                #self.TOAs[nn] = MJD + (phi_med*P)
                self.phis[nn] = phi_med
                #self.TOAs_95[nn] = phi_95*P
                self.phis_95[nn] = phi_95
                #self.TOAs_std[nn] = phinode.stats(start=starti)['standard deviation']
                self.phis_std[nn] = phinode.stats(start=starti)['standard deviation']
                self.phis_trace[nn] = phinode.trace()
                self.Ddms[nn] = Ddm_med
                self.Ddms_95[nn] = Ddm_95
                self.Ddms_std[nn] = Ddmnode.stats(start=starti)['standard deviation']
                self.Ddms_trace[nn] = Ddmnode.trace()
                self.scales.append(np.median(scalesnode.trace()))
        elif lsfit:
            if write_TOAs:      #FIX
                obs = self.arch.get_telescope()
                obs_codes = ["@","0","1","2"]
                obs = "1"
            print "Each of the %d TOAs are approximately %.2f s"%(self.nsub,self.arch.integration_length()/self.nsub)
            print "Doing Fourier-domain least-squares fit via chi_2 minimization...\n"  #FIX
            start = time.time()
            self.phis = []
            self.Ddms = []
            self.nfevals = []
            self.rcs = []
            self.scalesx = []
            self.scales = []
            self.param_errs = []
            self.red_chi2s = []
            for nn in range(self.nsub):
            #for nn in range(5):
                dataportrait = self.portxs[nn]
                portx_fft = np.fft.rfft(dataportrait,axis=1)
                pw = self.portweights[nn]
                model,modelx = screen_portrait(mp.model,pw)
                freqsx = ma.masked_array(self.freqs,mask=self.maskweights[nn]).compressed()
                nu0 = self.nu0
                ####################
                #DOPPLER CORRECTION#
                ####################
                dc = self.arch.get_Integration(nn).get_doppler_factor()
                freqsx = correct_freqs_doppler(freqsx,dc)
                nu0 = correct_freqs_doppler(self.nu0,dc)
                ####################
                if nn == 0:
                    #phaseguess,ampguess = first_guess(dataportrait,modelx,nguess=20)    #FIX how does it tell the diff between say, +0.85 and -0.15
                    #print "Phase and amplitude guesses %.5f %.5f"%(phaseguess, ampguess)
                    phaseguess = first_guess(dataportrait,modelx,nguess=5000)
                    #if phaseguess > 0.5: phaseguess = phaseguess - 1    #FIX good fix?
                    #self.DM = self.arch.get_dispersion_measure()
                    self.DM = 0.0
                    #Ddmguess = self.DM
                    Ddmguess = 0.0
                    if not quiet: print "Phase guess: %.8f ; Ddm guess: %.5f"%(phaseguess,Ddmguess)
                #else:   #To first order this only speeds things up marginally, same answers found, unless it breaks...
                #    phaseguess = self.phis[nn-1]    #FIX Might not be a good idea if RFI or something throws it completely off, whereas first phaseguess only depends on pulse profile...
                #    Ddmguess = self.Ddms[nn-1]
                #if not quiet: print "Phase guess: %.8f ; Ddm guess: %.5f"%(phaseguess,Ddmguess)
                P = self.Ps[nn]
                MJD = self.MJDs[nn]
                #NEED status bar?
                print "Fitting for TOA %d...put more info here"%(nn+1)      #FIX
                phi,Ddm,nfeval,rc,scalex,param_errs,red_chi2 = fit_portrait(self.portxs[nn],modelx,np.array([phaseguess,Ddmguess]),P,freqsx,nu0,scales=True)
                self.phis.append(phi)
                self.Ddms.append(Ddm)
                self.nfevals.append(nfeval)
                self.rcs.append(rc)
                self.scalesx.append(scalex)
                scale = np.zeros(self.nchan)
                ss = 0
                for ii in range(self.nchan):
                    if self.normweights[nn,ii] == 1:
                        scale[ii] = scalex[ss]
                        ss += 1
                    else: pass
                self.scales.append(scale)
                self.param_errs.append(param_errs)
                self.red_chi2s.append(red_chi2)
                #duration = time.time()-start
                #print nn,duration,phi,Ddm,scalesx,param_errs,red_chi2 #Also have it print red_chi_2
                ###mark rc=1,2,4 points in different colors###
                #self.show_fit(nn)
            if write_TOAs:
                toas = [self.epochs[nn] + pr.MJD((self.phis[nn]*self.Ps[nn])/(3600*24.)) for nn in xrange(self.nsub)]
                toa_errs = [np.array(self.param_errs)[nn,0]*self.Ps[nn]*1e6 for nn in xrange(self.nsub)]
                if self.outfile: sys.stdout = open(self.outfile,"a")
                for nn in range(self.nsub):
                    Ddm = self.Ddms[nn] - self.DM
                    #Should write which freqs? topo or bary?
                    write_princeton_toa(toas[nn].intday(),toas[nn].fracday(),toa_errs[nn],self.nu0,Ddm,obs=obs)
            sys.stdout = sys.__stdout__
            duration = time.time()-start
            #print self.red_chi2s
            print "\nFitting took %.1f min, ~%.3f min/TOA, mean TOA error is %.3f us"%(duration/60.,duration/(60*self.nsub),np.array(self.param_errs)[:,0].mean()*self.Ps.mean()*1e6)
        show_Ddmerr = 0
        if show_Ddmerr:
            for nn in range(self.nsub):
                print "%.2e"%np.array(self.param_errs)[nn,1]

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
        Ddm = self.Ddms[ii]
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
        fitmodel = np.transpose(self.scales[ii]*np.transpose(rotate_portrait(model,-phi,-Ddm,P,freqs,nu0)))
        fitmodelx = np.transpose(self.scalesx[ii]*np.transpose(rotate_portrait(modelx,-phi,-Ddm,P,freqsx,nu0)))
        #fitmodelmasked = rotate_portrait(modelmasked,-np.mean((M.phi.trace()[starti:])),-10**np.mean((M.Ddm.trace()[starti:])),freqs,nu0)
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
        fit_results = pf(self.MJDs,self.phis*self.Ps*1e3,1,full=True,w=np.array(self.param_errs)[:,0]**-2)      #FIX not sure weighting works...
        resids = np.array(self.phis*self.Ps*1e3)-(fit_results[0][0]+(fit_results[0][1]*self.MJDs))
        RMS = np.sum(resids**2/len(resids))**0.5        #FIX  check this, RMS seems too high
        ax1 = fig.add_subplot(311)
        #ax1.errorbar(self.MJDs,self.phis,[self.param_errs[xx][0] for xx in xrange(len(self.phis))],color='k',fmt='+')
        for nn in range(len(self.phis)):
            ax1.errorbar(self.MJDs[nn],self.phis[nn]*self.Ps[nn]*1e6,self.param_errs[nn][0]*self.Ps[nn]*1e6,color='%s'%cols[self.rcs[nn]],fmt='+')
        plt.plot(self.MJDs,(fit_results[0][0]+(fit_results[0][1]*self.MJDs))*1e3,"m--")
        plt.xlabel("MJD")
        plt.ylabel(r"Offset [$\mu$s]")
        ax1.text(0.1,0.9,"%.2e ms/s"%(fit_results[0][1]/(3600*24)),ha='center',va='center',transform=ax1.transAxes)
        ax2 = fig.add_subplot(312)
        for nn in range(len(self.phis)):
            ax2.errorbar(self.MJDs[nn],resids[nn]*1e3,self.param_errs[nn][0]*self.Ps[nn]*1e6,color='%s'%cols[self.rcs[nn]],fmt='+')
        plt.xlabel("MJD")
        plt.ylabel(r"Offset [$\mu$s]")
        ax2.text(0.1,0.9,"RMS = %d ns"%int(RMS*1e6),ha='center',va='center',transform=ax2.transAxes)
        #FIX below; no need to fit...just get average and std
        fit_results = pf(self.MJDs,self.Ddms,0,full=True,w=np.array(self.param_errs)[:,1]**-2)      #FIX not sure weighting works...
        #fit_results = pf(self.MJDs,self.Ddms,1,full=True,w=np.array(self.param_errs)[:,1]**-2)      #FIX not sure weighting works...
        resids = np.array(self.Ddms)-fit_results[0][0]
        #resids = np.array(self.Ddms)-(fit_results[0][0]+(fit_results[0][1]*self.MJDs))
        RMS = np.sum(resids**2/len(resids))**0.5
        ax3 = fig.add_subplot(313)
        #ax3.errorbar(self.MJDs,self.Ddms,[self.param_errs[xx][1] for xx in xrange(len(self.Ddms))],color='k',fmt='+')
        for nn in range(len(self.phis)):
            ax3.errorbar(self.MJDs[nn],self.Ddms[nn],self.param_errs[nn][1],color='%s'%cols[self.rcs[nn]],fmt='+')
        plt.plot(self.MJDs,np.ones(len(self.MJDs))*fit_results[0][0],"m--")
        plt.xlabel("MJD")
        plt.ylabel(r"DM [pc cm$^{3}$]")
        ax3.text(0.15,0.9,"dDM = %.2e ; RMS = %.2e"%(fit_results[0][0],RMS),ha='center',va='center',transform=ax3.transAxes)
        plt.show()

    def show_hists(self):
        cols = ['b','k','g','b','r']
        bins = np.array(self.nfevals).max()
        binmin = np.array(self.nfevals).min()
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
