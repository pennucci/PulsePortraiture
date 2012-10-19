#!/usr/bin/env python

# Gaussian fit + GUI stuff stolen from Scott Ransom's pygaussfit.py and subsequently HACKED
# To be used with PSRCHIVE Archive files

# This software lays on the bed of Procrustes all too comfortably.

#Next two lines needed for dispatching on nodes
#import matplotlib
#matplotlib.use("Agg")
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from psr_utils import rotate, fft_rotate
import numpy as np
import psrchive as pr
import numpy.fft as fft
import numpy.ma as ma
import scipy.optimize as opt
import mpfit as mp

cols = ['b','g','r','c','m','y','b','g','r','c','m','y','b','g','r','c','m','y','b','g','r','c','m','y','b','g','r','c','m','y']
Dconst = 4.148808e3     #Dispersion constant (e**2/(2*pi*m_e*c)) [MHz**2 pc**-1 cm**3 s]

def gaussian_profile(N, phase, fwhm, norm=0):
    """
    gaussian_profile(N, phase, fwhm):
        Return a gaussian pulse profile with 'N' bins and
        an integrated 'flux' of 1 unit (if norm=1; default norm=0 and peak ampltiude = 1).
            'N' = the number of points in the profile
            'phase' = the pulse phase (0-1)
            'fwhm' = the gaussian pulses full width at half-max
        Note:  The FWHM of a gaussian is approx 2.35482 sigma (exactly 2*sqrt(2*ln(2)))
    """
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    mean = phase % 1.0
    phsval = np.arange(N, dtype='d') / float(N)
    if (mean < 0.5):
        phsval = np.where(np.greater(phsval, mean+0.5),
                           phsval-1.0, phsval)
    else:
        phsval = np.where(np.less(phsval, mean-0.5),
                           phsval+1.0, phsval)
    try:
        zs = (phsval-mean)/sigma
        okzinds = np.compress(np.fabs(zs)<20.0, np.arange(N))
        okzs = np.take(zs, okzinds)
        retval = np.zeros(N, 'd')
        np.put(retval, okzinds, np.exp(-0.5*(okzs)**2.0)/(sigma*np.sqrt(2*np.pi)))
        if norm: return retval
        else: return retval/np.max(retval)
    except OverflowError:
        print "Problem in gaussian prof:  mean = %f  sigma = %f" % \
              (mean, sigma)
        return np.zeros(N, 'd')

class GaussianSelector:
    def __init__(self, ax, profile, errs, profnm, minspanx=None,
                 minspany=None, useblit=True):
        print "============================================="
        print "Left mouse click to draw a Gaussian component"
        print "Middle mouse click to fit components to data"
        print "Right mouse click to remove a component"
        print "============================================="
        print "Press 'q' or close window when done fitting"
        print "============================================="
        self.ax = ax.axes
        self.profile = profile
        self.proflen = len(profile)
        self.profnm = profnm
        self.phases = np.arange(self.proflen, dtype='d')/self.proflen
        self.errs = errs
        self.visible = True
        self.DCguess = sorted(profile)[len(profile)/10+1]
        self.init_params = [self.DCguess]
        self.numgaussians = 0
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.canvas.mpl_connect('draw_event', self.update_background)
        self.canvas.mpl_connect('key_press_event',self.keypress)
        self.background = None
        self.rectprops = dict(facecolor='white', edgecolor = 'black',
                              alpha=0.5, fill=False)
        self.to_draw = Rectangle((0,0), 0, 1, visible=False, **self.rectprops)
        self.ax.add_patch(self.to_draw)
        self.useblit = useblit
        self.minspanx = minspanx
        self.minspany = minspany
        # will save the data (position at mouseclick)
        self.eventpress = None
        # will save the data (pos. at mouserelease)
        self.eventrelease = None
        self.plot_gaussians(self.init_params)

    def update_background(self, event):
        'force an update of the background'
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def ignore(self, event):
        'return True if event should be ignored'
        # If no button was pressed yet ignore the event if it was out
        # of the axes
        if self.eventpress == None:
            return event.inaxes!= self.ax
        # If a button was pressed, check if the release-button is the
        # same.
        return (event.inaxes!=self.ax or
                event.button != self.eventpress.button)

    def press(self, event):
        'on button press event'
        # Is the correct button pressed within the correct axes?
        if self.ignore(event): return
        # make the drawed box/line visible get the click-coordinates,
        # button, ...
        self.eventpress = event
        if event.button==1:
            self.to_draw.set_visible(self.visible)
            self.eventpress.ydata = self.DCguess

    def release(self, event):
        'on button release event'
        if self.eventpress is None or self.ignore(event): return
        # release coordinates, button, ...
        self.eventrelease = event
        if event.button==1:
            # make the box/line invisible again
            self.to_draw.set_visible(False)
            self.canvas.draw()
            xmin, ymin = self.eventpress.xdata, self.eventpress.ydata
            xmax, ymax = self.eventrelease.xdata, self.eventrelease.ydata
            # calculate dimensions of box 
            if xmin>xmax: xmin, xmax = xmax, xmin
            if ymin>ymax: ymin, ymax = ymax, ymin
            spanx = xmax - xmin
            spany = ymax - ymin
            xproblems = self.minspanx is not None and spanx<self.minspanx
            yproblems = self.minspany is not None and spany<self.minspany
        # call desired function
        self.onselect()
        self.eventpress = None                # reset the variables to their
        self.eventrelease = None              #   inital values

    def update(self):
        'draw using newfangled blit or oldfangled draw depending on useblit'
        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.to_draw)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def onmove(self, event):
        if self.eventpress is None or self.ignore(event): return
        x,y = event.xdata, event.ydata # actual position with button still pressed
        minx, maxx = self.eventpress.xdata, x # click-x and actual mouse-x
        miny, maxy = self.eventpress.ydata, y # click-y and actual mouse-y
        if minx>maxx: minx, maxx = maxx, minx # get them in the right order
        if miny>maxy: miny, maxy = maxy, miny
        self.to_draw.set_x(minx)             # set lower left of box
        self.to_draw.set_y(miny)
        self.to_draw.set_width(maxx-minx)     # set width and height of box
        self.to_draw.set_height(maxy-miny)
        self.update()

    def keypress(self, event):
        'on key press event'
        if self.ignore(event): return
        self.eventpress = event
        if event.key=='q':
            self.eventpress = None
            self.close()

    def plot_gaussians(self, params):
        plt.subplot(211)
        plt.cla()
        # Re-plot the original profile
        plt.plot(self.phases, self.profile, c='black', lw=3, alpha=0.3)
        plt.xlabel('Pulse Phase')
        plt.ylabel('Pulse Amplitude')
        DC = params[0]
        # Plot the individual gaussians
        for ii in xrange(self.numgaussians):
            phase, FWHM, amp = params[1+ii*3:4+ii*3]
            plt.plot(self.phases, DC + amp*gaussian_profile(self.proflen, phase, FWHM),'%s'%cols[ii])

    def onselect(self):
        event1 = self.eventpress
        event2 = self.eventrelease
        # Left mouse button = add a gaussian
        if event1.button == event2.button == 1:
            x1, y1 = event1.xdata, event1.ydata
            x2, y2 = event2.xdata, event2.ydata
            phase = 0.5*(x1+x2)
            FWHM = np.fabs(x2-x1)
            #amp = np.fabs(1.05*(y2-self.init_params[0])*(x2-x1))
            amp = np.fabs(1.05*(y2-self.init_params[0]))
            self.init_params += [phase, FWHM, amp]
            self.numgaussians += 1
            self.plot_gaussians(self.init_params)
            plt.draw()
        # Middle mouse button = fit the gaussians
        elif event1.button == event2.button == 2:
            fit_params, fit_errs, chi_sq, dof, residuals = \
                        fit_gaussians(self.profile, self.init_params,
                                      np.zeros(self.proflen)+self.errs,
                                      self.profnm)
            self.fit_params = fit_params
            # scaled uncertainties
            #scaled_fit_errs = fit_errs * np.sqrt(chi_sq / dof)

            # Plot the best-fit profile
            self.plot_gaussians(fit_params)
            fitprof = gen_gaussians(fit_params, self.proflen)
            plt.plot(self.phases, fitprof, c='black', lw=1)
            plt.draw()

            # Plot the residuals
            plt.subplot(212)
            plt.cla()
            residuals = self.profile - fitprof
            plt.plot(self.phases, residuals,'k')
            plt.xlabel('Pulse Phase')
            plt.ylabel('Data-Fit Residuals')
            plt.draw()
        # Right mouse button = remove last gaussian
        elif event1.button == event2.button == 3:
            if self.numgaussians:
                self.init_params = self.init_params[:-3]
                self.numgaussians -= 1
                self.plot_gaussians(self.init_params)
                plt.draw()
                plt.subplot(212)
                plt.cla()
                plt.xlabel('Pulse Phase')
                plt.ylabel('Data-Fit Residuals')
                plt.draw()

    def close(self):
        plt.close(1)
        plt.close(2)

def gen_gaussians(params, N):
    """
    gen_gaussians(params, N):
        Return a model of a DC-component + M gaussians
            params is a sequence of 1+M*3 values
                the first value is the DC component.  Each remaining
                group of three represents the gaussians phase (0-1),
                FWHM (0-1), and amplitude (>0.0).
            N is the number of points in the model.
    """
    numgaussians = (len(params)-1)/3
    model = np.zeros(N, dtype='d') + params[0]
    for ii in xrange(numgaussians):
        phase, FWHM, amp = params[1+ii*3:4+ii*3]
        model += amp * gaussian_profile(N, phase, FWHM)
    return model

def powlaw(nu,nu0,A,alpha):
    """
    Power-law spectrum given by:
    F(nu) = A*(nu/nu0)**(alpha)
    """
    return A*((nu/nu0)**(alpha))

def powlawint(nu2,nu1,nu0,A,alpha):
    """
    Returns the integral over a powerlaw of form A*(nu/nu0)**(alpha)
    from nu1 to nu2
    """
    alpha = np.float(alpha)
    if alpha == -1.0:
        return A*nu0*np.log(nu2/nu1)
    else:
        C = A*(nu0**-alpha)/(1+alpha)
        diff = ((nu2**(1+alpha))-(nu1**(1+alpha)))
        return C*diff

def powlawnus(lo,hi,N,alpha,mid=False):
    """
    Returns frequencies such that a bandwidth from lo to hi frequencies
    split into N chunks contains the same amount of power in each chunk
    given a power-law across the band with spectral index alpha.  Default
    behavior return N+1 frequencies (includes both lo and hi freqs); if
    mid=True, it returns N frequencies, corresponding to the middle frequency
    in each chunk.
    """
    alpha = np.float(alpha)
    nus = np.zeros(N+1)
    if alpha == -1.0:
        nus = np.exp(np.linspace(np.log(lo),np.log(hi),N+1))
    else:
        nus = np.power(np.linspace(lo**(1+alpha),hi**(1+alpha),N+1),(1+alpha)**-1)
        #Equivalently:
        #for nn in xrange(N+1):
        #    nus[nn] = ((nn/np.float(N))*(hi**(1+alpha)) + (1-(nn/np.float(N)))*(lo**(1+alpha)))**(1/(1+alpha))
    if mid:
        midnus = np.zeros(N)
        for nn in xrange(N):
            midnus[nn] = 0.5*(nus[nn]+nus[nn+1])
        nus = midnus
    return nus

def fit_gauss_function(params, fjac=None, data=None, errs=None):
    """
    """
    return [0, (data - gen_gaussians(params, len(data))) / errs]

def fit_pl_function(params, freqs, nu0, weights=None, fjac=None, data=None, errs=None):     #NO NEED FOR WEIGHTS HERE?
    """
    """
    A = params[0]
    alpha = params[1]
    d = []
    f = []
    for ii in xrange(len(weights)):
        if weights[ii]:
            d.append(data[ii])
            f.append(freqs[ii])
        else: pass
    d=np.array(d)
    f=np.array(f)
    return [0, (d - powlaw(f,nu0,A,alpha)) / errs]

def fit_portrait_function(params, model=None, p=None, data=None, d=None, errs=None, P=None, freqs=None, nu_ref=np.inf):
    phase = params[0]
    m = 0.0
    if P == None or freqs == None:
        Cdm = 0.0
        freqs = np.inf*np.ones(len(model))
    else: Cdm = params[1]*Dconst/P
    for nn in xrange(len(freqs)):
        err = errs[nn]
        freq = freqs[nn]
        phasor = np.exp(np.arange(len(model[nn])) * 2.0j*np.pi*(phase+(Cdm*(freq**-2.0 - nu_ref**-2.0))))
        mm = np.real(data[nn,:] * np.conj(model[nn,:]) * phasor).sum()
        m += (mm**2.0)*err/p[nn]
    #print phase,params[1]
    return d-m

def fit_portrait_function_deriv(params, model=None, p=None, data=None, d=None, errs=None, P=None, freqs=None, nu_ref=np.inf):
    """
    """
    phase = params[0]
    Cdm = params[1]*Dconst/P
    d_phi,d_Ddm = 0.0,0.0
    for nn in xrange(len(freqs)):
        err = errs[nn]
        freq = freqs[nn]
        harmind = np.arange(len(model[nn]))
        phasor = np.exp(harmind * 2.0j*np.pi*(phase+(Cdm*(freq**-2.0 - nu_ref**-2.0))))
        g1 = np.real(data[nn,:]*np.conj(model[nn,:]) * phasor).sum()
        gp2 = np.real(2j*np.pi*harmind * data[nn,:] * np.conj(model[nn,:]) * phasor).sum()
        gd2 = np.real(2j*np.pi*harmind * (freq**-2.0 - nu_ref**-2.0)*(Dconst/P) *data[nn,:] * np.conj(model[nn,:]) * phasor).sum()
        d_phi += -2*g1*gp2*err/p[nn]
        d_Ddm += -2*g1*gd2*err/p[nn]
    #print d_phi,d_Ddm
    return np.array([d_phi,d_Ddm])

def fit_portrait_function_2deriv(params, model=None, p=None, data=None, d=None, errs=None, P=None, freqs=None, nu_ref=np.inf):      #Covariance matrix...??
    """
    """
    phase = params[0]
    Cdm = params[1]*Dconst/P
    d2_phi,d2_Ddm = 0.0,0.0
    for nn in xrange(len(freqs)):
        err = errs[nn]
        freq = freqs[nn]
        harmind = np.arange(len(model[nn]))
        phasor = np.exp(harmind * 2.0j*np.pi*(phase+(Cdm*(freq**-2.0 - nu_ref**-2.0))))
        g1 = np.real(data[nn,:]*np.conj(model[nn,:]) * phasor).sum()
        gp2 = np.real(2.0j*np.pi*harmind * data[nn,:] * np.conj(model[nn,:]) * phasor).sum()
        gd2 = np.real(2.0j*np.pi*harmind * (freq**-2.0 - nu_ref**-2.0)*(Dconst/P) *data[nn,:] * np.conj(model[nn,:]) * phasor).sum()
        gp3 = np.real(pow(2.0j*np.pi*harmind,2.0)*data[nn,:] * np.conj(model[nn,:])* phasor).sum()
        gd3 = np.real(pow(2.0j*np.pi*harmind*(freq**-2.0 - nu_ref**-2.0)*(Dconst/P),2) * data[nn,:] * np.conj(model[nn,:]) * phasor).sum()
        d2_phi += -2.0*err*(pow(gp2,2.0)+(g1*gp3))/p[nn]
        d2_Ddm += -2.0*err*(pow(gd2,2.0)+(g1*gd3))/p[nn]
    return np.array([d2_phi,d2_Ddm])

def wiener_filter(prof,noise):      #FIX does not work
    """
    prof is noisy template
    noise is standard deviation of the gaussian noise in the data
    """
    FFT = fft.rfft(prof)
    pows = np.real(FFT*np.conj(FFT)) / len(prof)        #Check Normalization
    return pows/(pows+(noise**2))
    #return (pows - (noise**2)) / pows

def brickwall_filter(n,kc):
    """
    """
    fk = np.zeros(n)
    fk[:kc] = 1.0
    return fk

def find_kc(prof,noise):
    """
    """
    wf = wiener_filter(prof,noise)
    N = len(wf)
    X2 = np.zeros(N)
    for ii in xrange(N):
        X2[ii] = np.sum((wf-brickwall_filter(N,ii))**2)
    return X2.argmin()

def fit_gaussians(data, initial_params, errs, profnm, quiet=0):
    """
    """
    numparams = len(initial_params)
    numgaussians = (len(initial_params)-1)/3
    # Generate the parameter structure
    params0 = []
    parinfo = []
    for ii in xrange(numparams):
        params0.append(initial_params[ii])
        if ii in range(numparams)[1::3]:
            parinfo.append({'value':initial_params[ii], 'fixed':0,
                            'limited':[0,0], 'limits':[0.,0.]})
        elif ii in range(numparams)[2::3]:
            parinfo.append({'value':initial_params[ii], 'fixed':0,
                            'limited':[1,0], 'limits':[0.,0.]})
        elif ii in range(numparams)[3::3]:
            parinfo.append({'value':initial_params[ii], 'fixed':0,
                            'limited':[1,0], 'limits':[0.,0.]})     #Not sure these limits are working...
        else:
            parinfo.append({'value':initial_params[ii], 'fixed':0,
                            'limited':[0,0], 'limits':[0.,0.]})
    other_args = {'data':data, 'errs':errs}
    # Now fit it
    mpfit_out = mp.mpfit(fit_gauss_function, params0, functkw=other_args,
                            parinfo=parinfo, quiet=1)
    print mpfit_out.errmsg
    fit_params = mpfit_out.params
    fit_errs = mpfit_out.perror
    # degrees of freedom
    dof = len(data) - len(fit_params)
    # chi-squared for the model fit
    chi_sq = mpfit_out.fnorm
    residuals = data - gen_gaussians(fit_params, len(data))
    #outfile = open("modelparams.dat","w")
    #outfile.write("%.5f\n"%fit_params[0])
    #for ii in xrange(numgaussians):
    #    outfile.write("%.5f\n"%fit_params[1+ii*3])
    #    outfile.write("%.5f\n"%fit_params[2+ii*3])
    #    outfile.write("%.5f\n"%fit_params[3+ii*3])
    #outfile.close()
    if not quiet:
        print "------------------------------------------------------------------"
        print "Multi-Gaussian Fit of '%s'"%profnm
        print "------------------------------------------------------------------"
        print "mpfit status:", mpfit_out.status
        print "gaussians:", numgaussians
        print "DOF:", dof
        print "chi-sq: %.2f" % chi_sq
        print "reduced chi-sq: %.2f" % (chi_sq/dof)
        print "residuals mean: %.3g" % np.mean(residuals)
        print "residuals stdev: %.3g" % np.std(residuals)
        print "--------------------------------------"
        print " const = %.5f +/- %.5f" % (fit_params[0], fit_errs[0])
        for ii in xrange(numgaussians):
            print " phas%d = %.5f +/- %.5f" % (ii+1, fit_params[1+ii*3], fit_errs[1+ii*3])
            print " fwhm%d = %.5f +/- %.5f" % (ii+1, fit_params[2+ii*3], fit_errs[2+ii*3])
            print " ampl%d = %.5f +/- %.5f" % (ii+1, fit_params[3+ii*3], fit_errs[3+ii*3])
        print "--------------------------------------"
    return fit_params, fit_errs, chi_sq, dof, residuals

def fit_powlaws(data, freqs, nu0, weights, initial_params, errs):
    """
    """
    numparams = len(initial_params)
    # Generate the parameter structure
    params0 = []
    parinfo = []
    for ii in xrange(numparams):
        params0.append(initial_params[ii])
        parinfo.append({'value':initial_params[ii], 'fixed':0,
                        'limited':[0,0], 'limits':[0.,0.]})
    other_args = {'freqs':freqs, 'nu0':nu0, 'weights':weights, 'data':data, 'errs':errs}
    # Now fit it
    mpfit_out = mp.mpfit(fit_pl_function, params0, functkw=other_args,
                         parinfo=parinfo, quiet=1)
    fit_params = mpfit_out.params
    fit_errs = mpfit_out.perror
    # degrees of freedom
    dof = len(data) - len(fit_params)
    # chi-squared for the model fit
    chi_sq = mpfit_out.fnorm
    residuals = data - powlaw(freqs,nu0,fit_params[0],fit_params[1])
    return fit_params, fit_errs, chi_sq, dof, residuals

def fit_portrait(data,model,initial_params,P=None,freqs=None,nu_ref=np.inf,scales=False):        #COMMENTS EVERYWHERE! #get_scales function
    """
    """
    errs = get_noise(data,tau=True,chans=True,fd=True,frac=4) #Precision, 1/variance.  FIX Need to use better filtering instead of frac        #WRONG WRONG WRONG EVERYWHERE!  FOURIER VARIANCE
    dFFT = fft.rfft(data,axis=1)
    mFFT = fft.rfft(model,axis=1)
    d = np.real(np.sum(np.transpose(errs*np.transpose(dFFT*np.conj(dFFT)))))
    p = np.real(np.sum(mFFT*np.conj(mFFT),axis=1))
    other_args = (mFFT,p,dFFT,d,errs,P,freqs,nu_ref)
    #minimize = opt.fmin #same as others, 14s
    #minimize = opt.fmin_powell  #+1 phase, off in Ddm, 11s
    #minimize = opt.fmin_l_bfgs_b #same answer as cg, ~10s,7.6s

    #minimize = opt.fmin_bfgs #doesn't work
    #minimize = opt.fmin_cg #~two minutes,75s
    #minimize = opt.fmin_ncg #almost the same as the other two, ~50s,~50s
    minimize = opt.fmin_tnc #same as other two, 10s,6s

    #results = minimize(fit_portrait_function,initial_params,args=other_args)
    #If the fit fails...
    results = minimize(fit_portrait_function,initial_params,fprime=fit_portrait_function_deriv,args=other_args,messages=0)
    #results = minimize(fit_portrait_function,initial_params,fprime=fit_portrait_function_deriv,args=other_args,bounds=[(0,1),(None)],messages=0)   #FIX bounds on phase?
    phi = results[0][0]
    Ddm = results[0][1]
    nfeval = results[1]
    return_code = results[2]
    if return_code != 1:
        print "Fit failed for some reason.  Return code is %d; consult RCSTRINGS dictionary"%return_code
    param_errs = list(pow(fit_portrait_function_2deriv(np.array([phi,Ddm]),mFFT,p,dFFT,d,errs,P,freqs,nu_ref),-0.5))
    DoF = len(data.ravel()) - (len(freqs)+2)    #minus 1?
    red_chi2 = fit_portrait_function(np.array([phi,Ddm]),mFFT,p,dFFT,d,errs,P,freqs,nu_ref) / DoF
    if scales:
        scales = get_scales(data,model,phi,Ddm,P,freqs,nu_ref)
        param_errs += list(pow(2*p*errs,-0.5))
        return phi, Ddm, nfeval, return_code, scales, np.array(param_errs),red_chi2
    else: return phi, Ddm, nfeval, return_code, np.array(param_errs),red_chi2

def first_guess(data,model,nguess=1000):       #is phaseguess/fit for phase the left/right shift for model/data?  #FIX FOR NEW VERSION!
    """
    """
    #Get initial guesses for phase, and amplitudes...
    #guessparams = []
    #guesschi2s = []
    #for ii in np.linspace(-1.0,1.0,nguess):
    crosscorr = np.empty(nguess)
    #phaseguess = np.linspace(0,1.0,nguess)
    phaseguess = np.linspace(-0.5,0.5,nguess)
    for ii in range(nguess):
        phase = phaseguess[ii]
        crosscorr[ii] = np.correlate(fft_rotate(np.sum(data,axis=0),phase*len(np.sum(data,axis=0))),np.sum(model,axis=0))
    phaseguess = phaseguess[crosscorr.argmax()]
    #results = fit_portrait(np.array([data.mean(0)]),np.array([model.mean(0)]),np.array([phaseguess]))
    #    guessparams.append(output[0])
    #    guesschi2s.append(output[2])
    #phaseguess = guessparams[np.argmin(np.array(guesschi2s))][0]%1
    #ampguess = guessparams[np.argmin(np.array(guesschi2s))][2]
    return phaseguess

def make_model(phases,freqs,modelfile=None,refparams=None,As=None,alphas=None,nu0=None,quiet=False):
    """
    """
    nbin = len(phases)
    nchan = len(freqs)
    if modelfile:
        modeldata = open(modelfile,"r").readlines()
        ngauss = len(modeldata)-3
        refparams = np.zeros(ngauss*3+1)
        As = np.zeros(ngauss)
        alphas = np.zeros(ngauss)
        source = str(modeldata.pop(0)[:-1])
        nu0 = float(modeldata.pop(0))   #Extraneous here?
        refparams[0] = float(modeldata.pop(0))
        for gg in xrange(ngauss):
            comp = map(float,modeldata[gg].split())
            refparams[1+gg*3:1+(gg*3)+3] = comp[:3]
            As[gg] = comp[3]
            alphas[gg] = comp[4]
    model=np.zeros([nchan,nbin]) + refparams[0]
    for ff in xrange(nchan):
        freq = freqs[ff]
        for nn in xrange(len(refparams-1)/3):
            G = refparams[(nn*3)+1:(nn*3)+4]
            mu = G[0]
            fwhm = G[1]
            A = As[nn]
            alpha = alphas[nn]
            amp = powlaw(freq,nu0,A,alpha)
            model[ff] += amp*gaussian_profile(nbin,mu,fwhm)
    if modelfile:
        if not quiet: print "Made model for %s with %d frequency channels, %d profile bins, %.0f MHz bandwidth centered on %.2f MHz"%(source,nchan,nbin,(freqs[-1]-freqs[0])+((freqs[-1]-freqs[-2])),nu0)
        return source,ngauss,refparams,As,alphas,nu0,model
    else:
        if not quiet: print "Made model with %d frequency channels, %d profile bins, %.0f MHz bandwidth centered on %.2f MHz"%(nchan,nbin,(freqs[-1]-freqs[0])+((freqs[-1]-freqs[-2])),nu0)
        return model

def get_noise(data,frac=4,tau=False,chans=False,fd=False):     #FIX: Make sure to use on portraits w/o zapped freq. channels, i.e. portxs     FIX: MAKE SIMPLER!!!    FIX: Implement k_max from wiener/brick-wall filter fit
    """
    """
    shape = data.shape
    if len(shape) == 1:
        prof = data
    elif shape[0] == 1:
        prof = data[0]
    elif shape[1] == 1:
        prof = data[:,0]
    else: pass
    try:
        FFT = fft.rfft(prof)
        if fd:
            if tau: return np.std(np.real(FFT)[-len(FFT)/frac:])**-2
            #if tau: return (np.std(np.real(FFT)[-len(FFT)/frac:])**-2,np.std(np.imag(FFT)[-len(FFT)/frac:])**-2)
            else: return np.std(np.real(FFT)[-len(FFT)/frac:])
            #else: return (np.std(np.real(FFT)[-len(FFT)/frac:]),np.std(np.imag(FFT)[-len(FFT)/frac:]))
        else:
            pows = np.real(FFT*np.conj(FFT))/len(prof)    #!!!CHECK NORMALIZATION
            if tau: return (np.mean(pows[-len(pows)/frac:]))**-1
            else: return np.sqrt(np.mean(pows[-len(pows)/frac:]))
    except(NameError):
        noise = np.zeros(len(data))
        if fd:
            for nn in range(len(noise)):
                    prof = data[nn]
                    FFT = fft.rfft(prof)
                    noise[nn] = np.std(np.real(FFT)[-len(FFT)/frac:])
            if chans:
                if tau: return noise**-2
                else: return noise
            else:
                if tau: return np.median(noise)**-2     #not statistically rigorous
                else: return np.median(noise)
        else:
            for nn in range(len(noise)):
                prof = data[nn]
                FFT = fft.rfft(prof)
                pows = np.real(FFT*np.conj(FFT))/len(prof)    #!!!CHECK NORMALIZATION
                noise[nn] = np.sqrt(np.mean(pows[-len(pows)/frac:]))
            if chans:
                if tau: return noise**-2
                else: return noise
            else:
                if tau: return np.median(noise)**-2     #not statistically rigorous
                else: return np.median(noise)

def get_scales(data,model,phase,Ddm,P,freqs,nu_ref):
    """
    """
    scales = np.zeros(len(model))
    dFFT = fft.rfft(data,axis=1)
    mFFT = fft.rfft(model,axis=1)
    p = np.real(np.sum(mFFT*np.conj(mFFT),axis=1))
    Cdm = Ddm*Dconst/P
    for kk in range(len(mFFT[0])):
        scales += np.real(dFFT[:,kk]*np.conj(mFFT[:,kk])*np.exp(2j*np.pi*kk*(phase+(Cdm*(pow(freqs,-2)-pow(nu_ref,-2))))))/p
    return scales

def rotate_portrait(port,phase,Ddm=None,P=None,freqs=None,nu_ref=np.inf):
    """
    Positive values of phase and Ddm rotate to earlier phase.
    """
    pFFT = fft.rfft(port,axis=1)
    for nn in xrange(len(pFFT)):
        for kk in xrange(len(pFFT[nn])):
            if Ddm is None and freqs is None: pFFT[nn,kk] *= np.exp(2j*np.pi*kk*phase)
            else:
                Cdm = Ddm*Dconst/P
                pFFT[nn,kk] *= np.exp(2j*np.pi*kk*(phase+(Cdm*(pow(freqs[nn],-2)-pow(nu_ref,-2)))))
    return fft.irfft(pFFT)

def plot_PL_results(M,withprof=1,witherrors=1, negative=0):
    """
    M is a pymc MCMC object (a chain).
    """
    nbin = M.nbin
    alphamu = M.alphas.stats()['mean']
    alpha95 = M.alphas.stats()['95% HPD interval']
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if witherrors:
        if negative: ax1.errorbar((np.arange(nbin)+0.5)/nbin,-alphamu,yerr=[alpha95.transpose()[1]-alphamu,alphamu-alpha95.transpose()[0]],fmt='r+')
        else: ax1.errorbar((np.arange(nbin)+0.5)/nbin,alphamu,yerr=[alphamu-alpha95.transpose()[0],alpha95.transpose()[1]-alphamu],fmt='r+')
    else:
        if negative: ax1.plot((np.arange(nbin)+0.5)/nbin, -alphamu,'r+')
        else: ax1.plot((np.arange(nbin)+0.5)/nbin, alphamu,'r+')
    if withprof:
        plt.axes()
        plt.plot(np.arange(nbin, dtype='d')/nbin,prof,'k-')
    plt.ylim(alphamu.min()-5,plt.ylim()[1])
    plt.show()

#def plot_amps(M)

#def plot_PL_bin(M,bin)

def write_model(filenm,source,refparams,As,alphas,nu0):
    """
    """
    outfile = open(filenm,"w")
    outfile.write("%s\n"%source)
    outfile.write("%.8f\n"%nu0)
    outfile.write("%.8f\n"%refparams[0])
    ngauss = len(refparams[1:])/3
    for nn in xrange(ngauss):
        comp = refparams[1+nn*3:4+nn*3]
        outfile.write("%.8f\t %.8f\t %.8f\t %.8f\t %.8f\n"%(comp[0],comp[1],comp[2],As[nn],alphas[nn]))
    outfile.close()
    print "%s written."%filenm
    return 0

def load_data(filenm,tscrunch=False,pscrunch=False,quiet=False,rm_baseline=(0,0),Gfudge=1.0):
    """
    Will read data using PSRCHIVE.  If not already done, it will dedisperse and pscrunch.
    """
    #Load archive
    arch = pr.Archive_load(filenm)
    source = arch.get_source()
    if not quiet: print "Reading data from %s on source %s..."%(filenm,source)
    #Get some metadata
    nu0 = arch.get_centre_frequency()   #I'm taking this to be the center of the continuous band, not some bin edge value
    bw = arch.get_bandwidth()
    nchan = arch.get_nchan()
    chanwidth = bw/nchan
    lofreq = nu0-(bw/2)
    freqs = np.linspace(lofreq+(chanwidth/2.0),lofreq+bw-(chanwidth/2.0),nchan)     #Centers of frequency channels
    nbin = arch.get_nbin()
    phases = np.arange(nbin, dtype='d')/nbin
    #Dedisperse?
    if not arch.get_dedispersed(): arch.dedisperse()
    #This is where I think the bandpass is being removed (needs to be robust...)
    baseline_removed = 0
    if rm_baseline[0] + rm_baseline[1] == 0:
        arch.remove_baseline()
        baseline_removed = 1
    #pscrunchd?
    if pscrunch: arch.pscrunch()
    else:
        print "Full Stokes not ready.  Try again later.  Sorry"
        return 0
    #if arch.get_state != 'Intensity': arch.pscrunch()  #Stokes
    #tscrunch?
#    if tscrunch:
#        arch.tscrunch()
#        nsub = arch.get_nsubint()
#        P = arch.get_Integration(0).get_folding_period()
#        #Get data
#        port = arch.get_data()[0]*Gfudge    #Stokes
#        for ss in range(4):                 #Stokes
#            for ff in range(nchan):         #Stokes
#                port[ss,ff] -= port[ss,ff,260:298].mean()   #Stokes     ##UGLY HARDCODE!!
#                #port[ss,ff] -= port[ss,ff,260/6:298/6].mean()   #Stokes        ##UGLY HARDCODE!!
#        #Get weights !!!Careful about this!!!
#        #weights = arch.get_weights()[0]
#        weights = port[0].sum(axis=1)  #STOKES FAKE WEIGHTS
#        normweights = np.divide(map(int,weights),map(int,weights))
#        maskweights = (normweights+1)%2
#        portweights = np.array([np.ones(nbin)*maskweights[ii] for ii in xrange(nchan)])
#        portx = []    #Stokes
#        for ss in range(4):     #Stokes
#            sp = screen_portrait(port[ss],portweights)  #Does not change port in this case (already "masked")   #Stokes
#            portx.append(sp[1])
#        portx = np.array(portx)
#        #Estimate noise
#        noise_stdev = np.zeros(4)     #Stokes
#        for ss in range(4):
#            noise_stdev[ss] = get_noise(portx[ss])   #Stokes #FIX
#        #Make flux profile
#        #fluxprof = port.sum(1)/nbin  #This is about equal to bscrunch to ~6 places     #Stokes
#        #fluxprofx = ma.masked_array(fluxprof,mask=maskweights).compressed()            #Stokes
#        freqsx = ma.masked_array(freqs,mask=maskweights).compressed()
#        #Get pulse profile
#        arch.fscrunch()
#        prof = arch.get_data()[0][0][0]*Gfudge
#        if not quiet:
#            print "\tcenter freq. (MHz) = %.5f\n\
#            bandwidth (MHz)    = %.1f\n\
#            # bins in prof.    = %d\n\
#            # channels         = %d\n\
#            unzapped chan.     = %d"%(nu0,bw,nbin,nchan,len(portx[0]))
#        arch.refresh()
#        return source,arch,port,portx,noise_stdev,prof,nbin,phases,nu0,bw,nchan,chanwidth,lofreq,freqs,freqsx,nsub,P,weights,normweights,maskweights,portweights    #Stokes
#    else:
    #tscrunch?
    if tscrunch: arch.tscrunch()
    nsub = arch.get_nsubint()
    #Get data
    ports = arch.get_data()[:,0,:,:]*Gfudge     #FIX Here assumes pscrunched in second index
    Ps = np.array([arch.get_Integration(ii).get_folding_period() for ii in xrange(nsub)],dtype=np.double)
    MJDs = [arch.get_Integration(ii).get_epoch() for ii in xrange(nsub)]
    #Get weights !!!Careful about this!!!
    weights = arch.get_weights()
    normweights = np.array([np.divide(map(int,weights[ii]),map(int,weights[ii])) for ii in xrange(len(weights))])
    maskweights = (normweights+1)%2
    portweights = np.array([np.array([np.ones(nbin)*maskweights[ii,jj] for jj in xrange(nchan)]) for ii in xrange(nsub)])
    ports,portxs = np.array([screen_portrait(ports[ii],portweights[ii])[0] for ii in xrange(nsub)]),np.array([screen_portrait(ports[ii],portweights[ii])[1] for ii in xrange(nsub)])    #FIX latter part may not work if portxs have different sizes
    arch.tscrunch()
    port = ports.mean(axis=0)
    portx = portxs.mean(axis=0)
    #Estimate noise
    noise_stdev = np.zeros(nsub)
    for nn in range(nsub):
        noise_stdev[nn] = get_noise(portxs[nn])     #FIX Check
    fluxprof = port.sum(1)/nbin  #This is about equal to bscrunch to ~6 places
    fluxprofx = ma.masked_array(fluxprof,mask=np.array(map(round,maskweights.mean(axis=0)))).compressed()
    freqsx = ma.masked_array(freqs,mask=np.array(map(round,maskweights.mean(axis=0)))).compressed()     #FIX will not work if portxs have different sizes/different things zapped
    #Get pulse profile
    arch.fscrunch()
    prof = arch.get_data()[0][0][0]*Gfudge
    if not quiet:
        print "\tcenter freq. (MHz) = %.5f\n\
        bandwidth (MHz)    = %.1f\n\
        # bins in prof.    = %d\n\
        # channels         = %d\n\
        # unzapped chan.   ~ %d\n\
        # sub ints         = %d"%(nu0,bw,nbin,nchan,int(np.array(map(len,portxs)).mean()),nsub) #FIX might not work if subints masked differently
    arch.refresh()      #FIX return as is or as requested scrunched?
    if tscrunch: return source,arch,ports[0],portxs[0],noise_stdev[0],fluxprof,fluxprofx,prof,nbin,phases,nu0,bw,nchan,chanwidth,lofreq,freqs,freqsx,nsub,Ps[0],MJDs[0],weights[0],normweights[0],maskweights[0],portweights[0]
    else: return source,arch,ports,portxs,noise_stdev,fluxprof,fluxprofx,prof,nbin,phases,nu0,bw,nchan,chanwidth,lofreq,freqs,freqsx,nsub,Ps,MJDs,weights,normweights,maskweights,portweights

def screen_portrait(port,portweights):
    """
    """
    #nbin = portweights.sum(axis=1).max()
    try: nbin = len(port[0])
    except(TypeError): nbin = len(port)
    normweights = (portweights.sum(axis=1)+1)%(nbin+1)
    nchan = nbin-portweights.sum(axis=0).max()
    maskedport = np.transpose(normweights*np.transpose(port))
    portx = ma.masked_array(port,mask=portweights).compressed()
    portx = portx.reshape(len(portx)/nbin,nbin)
    return maskedport,portx

def plot_lognorm(mu,tau,lo=0.0,hi=5.0,npts=500,plot=1,show=0):
    """
    """
    pts = np.empty(npts)
    xs = np.linspace(lo,hi,npts)
    for ii in xrange(npts):
        pts[ii] = np.exp(pm.lognormal_like(xs[ii],mu,tau))
    if plot:
        plt.plot(xs,pts)
    if show:
        plt.show()
    return xs,pts

def plot_gamma(alpha,beta,lo=0.0,hi=5.0,npts=500,plot=1,show=0):
    """
    """
    pts = np.empty(npts)
    xs = np.linspace(lo,hi,npts)
    for ii in xrange(npts):
        pts[ii] = np.exp(pm.gamma_like(xs[ii],alpha,beta))
    if plot:
        plt.plot(xs,pts)
    if show:
        plt.show()
    return xs,pts

def DM_delay(DM,freq,freq2=None,P=None):
    """
    Calculates the delay [s] of emitted frequency freq [MHz] from dispersion measure DM [cm**-3 pc] relative to infinite frequency.  If freq2 is provided, the relative delay is caluclated.  If a period P [s] is provided, the delay is returned in phase.
    """
    if freq2:
        delay = Dconst*DM*((freq**-2)-(freq2**-2))
    else:
        delay = Dconst*DM*(freq**-2)
    if P:
        return delay/P
    else: return delay

def make_fake():
    noise = np.random.normal(size=64**2)
    noise = noise.reshape(64,64)
    fake = np.zeros([64,64])
    fake[:,25] += 5
    fake[:,26] += 3
    fake[:,27] += 1
    fake[:,24] += 3
    fake[:,23] += 1
    model = 12.1*rotate_portrait(fake,-0.17)
    fake += noise
    return fake,model

def write_princeton_toa(toa_MJDi, toa_MJDf, toaerr, freq, dm, obs='@', name=' '*13):
    """
    RIPPED FROM PRESTO

    Princeton Format

    columns     item
    1-1     Observatory (one-character code) '@' is barycenter
    2-2     must be blank
    16-24   Observing frequency (MHz)
    25-44   TOA (decimal point must be in column 30 or column 31)
    45-53   TOA uncertainty (microseconds)
    69-78   DM correction (pc cm^-3)
    """
    # Splice together the fractional and integer MJDs
    toa = "%5d"%int(toa_MJDi) + ("%.13f"%toa_MJDf)[1:]
    if dm!=0.0:
        print obs+" %13s %8.3f %s %8.3f              %9.4f" % \
              (name, freq, toa, toaerr, dm)
    else:
        print obs+" %13s %8.3f %s %8.3f" % \
              (name, freq, toa, toaerr)
