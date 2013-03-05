#!/usr/bin/env python


from matplotlib.patches import Rectangle
from pplib import *


class DataPortrait:
    """
    """
    def __init__(self, datafile, quiet=False):
        ""
        ""
        self.datafile = datafile
        self.initial_model_run = False
        self.data = load_data(datafile, dedisperse=True, dededisperse=False,
                tscrunch=True, pscrunch=True, rm_baseline=True, flux_prof=True,
                quiet=quiet)
        #Unpack the data dictionary into the local namespace; see load_data for
        #dictionary keys.
        for key in self.data.keys():
            exec("self." + key + " = self.data['" + key + "']")
        if self.source is None: self.source = "noname"
        self.port = (self.masks*self.subints)[0,0]
        self.portx = self.subintsx[0][0]
        self.lofreq = self.freqs[0]-(self.bw/(2*self.nchan))
        self.init_params = []

    def show_data_portrait(self):
        """
        """
        title = "%s Portrait"%self.source
        show_port(self.port, self.freqs, title=title)

    def fit_profile(self, profile):
        """
        """
        fig = plt.figure()
        profplot = fig.add_subplot(211)
        #Maybe can do better than self.noise_std below
        interactor = GaussianSelector(profplot, profile, self.noise_std,
                minspanx=None, minspany=None, useblit=True)
        plt.show()
        self.init_params = interactor.fit_params
        self.ngauss = (len(self.init_params) - 1) / 3

    def fit_flux_profile(self, guessA=1.0, guessalpha=0.0, fit=True, plot=True,
            quiet=False):
        """
        Will fit a power law across frequency in a portrait by bin scrunching.
        This should be the usual average pulsar power-law spectrum.  The plot
        will show obvious scintles.
        """
        if fit:
            params, param_errs, chi2,dof, residuals = fit_powlaw(
                    self.flux_profx, self.freqsxs[0], self.nu0,
                    np.ones(len(self.flux_profx)),
                    np.array([guessA,guessalpha]),self.noise_std)
            if not quiet:
                print ""
                print "Flux-density power-law fit"
                print "----------------------------------"
                print "residual mean = %.2f"%residuals.mean()
                print "residual std. = %.2f"%residuals.std()
                print "A = %.3f (flux at %.2f MHz)"%(params[0], self.nu0)
                print "alpha = %.3f "%params[1]
            if plot:
                if fit: plt.subplot(211)
                else: plt.subplot(111)
                plt.xlabel("Frequency [MHz]")
                plt.ylabel("Flux Units")
                plt.title("Average Flux Profile for %s"%self.source)
                if fit:
                    plt.plot(self.freqs, powlaw(self.freqs, self.nu0,
                        params[0], params[1]), 'k-')
                    plt.plot(self.freqsxs[0], self.flux_profx, 'r+')
                    plt.subplot(212)
                    plt.xlabel("Frequency [MHz]")
                    plt.ylabel("Flux Units")
                    plt.title("Residuals")
                    plt.plot(self.freqsxs[0], residuals, 'r+')
                plt.show()
            self.spect_index = params[1]

    def set_model_run(self):
        self.initial_model_run = True

    def make_gaussian_model_portrait(self, ref_prof=None, locparams=0.0,
            fixloc=False, widparams=0.0, fixwid=False, ampparams=0.0,
            fixamp=False, niter=0, writemodel=False, outfile=None,
            model_name=None, residplot=None, quiet=False):
        """
        """
        self.nu_ref = ref_prof[0]
        self.bw_ref = ref_prof[1]
        if self.nu_ref is None: self.nu_ref = self.nu0
        if self.bw_ref is None: self.bw_ref = abs(self.bw)
        okinds = np.compress(np.less(self.nu_ref - (self.bw_ref/2),
            self.freqs) * np.greater(self.nu_ref + (self.bw_ref/2),
            self.freqs) * self.weights[0], np.arange(self.nchan))
        #The below profile average gives a slightly different set of values for
        #the profile than self.profile, if given the full band and center 
        #frequency.  Unsure why; shouldn't matter.
        profile = np.take(self.port, okinds, axis=0).mean(axis=0)
        self.fix_params = (fixloc, fixwid, fixamp)
        if not len(self.init_params): self.fit_profile(profile)
        if type(locparams) is not np.ndarray:
            try:
                locparams = np.ones(self.ngauss) * locparams
            except ValueError:
                print "Not enough parameters for ngauss = %d."%self.ngauss
                return 0
        if type(widparams) is not np.ndarray:
            try:
                widparams = np.ones(self.ngauss) * widparams
            except ValueError:
                print "Not enough parameters for ngauss = %d."%self.ngauss
                return 0
        if type(ampparams) is not np.ndarray:
            try:
                ampparams = np.ones(self.ngauss) * ampparams
            except ValueError:
                print "Not enough parameters for ngauss = %d."%self.ngauss
                return 0
        if outfile is None: outfile = self.datafile + ".model"
        if model_name is None: model_name = self.source
        self.init_model_params = np.empty([self.ngauss, 6])
        for nn in range(self.ngauss):
            self.init_model_params[nn] = np.array([self.init_params[1::3][nn],
                locparams[nn], self.init_params[2::3][nn], widparams[nn],
                self.init_params[3::3][nn], ampparams[nn]])
        self.init_model_params = np.array([self.init_params[0]] +
                list(np.ravel(self.init_model_params)))
        itern = niter
        if niter < 0: niter = 0
        portx_noise = np.outer(get_noise(self.portx, chans=True),
                np.ones(self.nbin))
        print "Fitting gaussian model portrait..."
        if not self.initial_model_run:
            start = time.time()
            self.fit_params, self.chi_sq, self.dof = fit_gaussian_portrait(
                    self.portx, portx_noise, self.init_model_params,
                    self.fix_params, self.phases, self.freqsxs[0], self.nu_ref,
                    quiet=quiet)
            if not quiet:
                print "Fit took %.2f min"%((time.time() - start) /  60.)
            niter += 1
        while(niter):
           if niter and self.initial_model_run:
               start = time.time()
               self.fit_params, self.chi_sq, self.dof = fit_gaussian_portrait(
                       self.portx, portx_noise, self.model_params,
                       self.fix_params, self.phases, self.freqsxs[0],
                       self.nu_ref, quiet=quiet)
               if not quiet:
                   print "Fit took %.2f min"%((time.time() - start) / 60.)
           self.model_params = self.fit_params
           self.model = gen_gaussian_portrait(self.model_params, self.phases,
                   self.freqs, self.nu_ref)
           self.model_masked = np.transpose(self.weights[0] *
                   np.transpose(self.model))
           self.modelx = np.compress(self.weights[0], self.model, axis=0)
           niter -= 1
           dofit = 1
           if dofit == 1:
               phaseguess = first_guess(self.portx, self.modelx, nguess=1000)
               DMguess = 0.0
               phi, DM, nfeval, rc, scalesx, param_errs, red_chi2, duration = (
                       fit_portrait(self.portx, self.modelx,
                           np.array([phaseguess, DMguess]), self.Ps[0],
                           self.freqsxs[0], self.nu0, scales=True)
                       )
               phierr = param_errs[0]
               DMerr = param_errs[1]
               if not quiet:
                   print "Iter %d:"%(itern - niter)
                   print " phase offset of %.2e +/- %.2e [rot]"%(phi, phierr)
                   print " DM of %.2e +/- %.2e [pc cm**-3]"%(DM, DMerr)
                   print " red. chi**2 of %.2f."%red_chi2
               else:
                   if niter and (itern - niter) != 0:
                       print "Iter %d..."%(itern - niter)
               if min(abs(phi), abs(1 - phi)) < abs(phierr):
                   if abs(DM) < abs(DMerr):
                       print "\nIteration converged.\n"
                       phi = 0.0
                       DM = 0.0
                       niter = 0
               if niter:
                   if not quiet:
                       print "\nRotating data portrait for iteration %d."%(
                               itern - niter + 1)
                   self.port = rotate_portrait(self.port, phi,DM, self.Ps[0],
                           self.freqs, self.nu0)
                   self.portx = rotate_portrait(self.portx, phi, DM,
                           self.Ps[0], self.freqsxs[0], self.nu0)
                   self.set_model_run()
        if not quiet:
            print "Residuals mean: %.3f"%(self.portx - self.modelx).mean()
            print "Residuals std:  %.3f"%(self.portx - self.modelx).std()
            print "Data std:       %.3f\n"%self.noise_std
        if writemodel:
            write_model(outfile, model_name, self.model_params, self.nu_ref)
        if residplot:
            self.show_residual_plot(residplot)

    def show_residual_plot(self, savefig=None):
        """
        """
        try:    #Make smarter
            test_for_model = self.model.shape
        except(AttributeError):
            print "No model portrait. Use make_gaussian_model_portrait()."
            return 0
        modelfig = plt.figure()
        aspect = "auto"
        interpolation = "none"
        origin = "lower"
        extent = (0.0, 1.0, self.freqs[0], self.freqs[-1])
        plt.subplot(221)
        plt.title("Data Portrait")
        plt.imshow(self.port, aspect=aspect, interpolation=interpolation,
                origin=origin, extent=extent)
        plt.subplot(222)
        plt.title("Model Portrait")
        plt.imshow(self.model, aspect=aspect, interpolation=interpolation,
                origin=origin, extent=extent)
        plt.subplot(223)
        plt.title("Residuals")
        plt.imshow(self.port - self.model_masked, aspect=aspect,
                interpolation=interpolation, origin=origin, extent=extent)
        plt.colorbar()
        #plt.subplot(224)
        #plt.title(r"Log$_{10}$(abs(Residuals/Data))")
        #plt.imshow(np.log10(abs(self.port - self.model) / self.port),
        #        aspect=aspect, origin=origin, extent=extent)
        #plt.colorbar()

        if savefig:
            plt.savefig(savefig, format='png')
        else:
            plt.show()


class GaussianSelector:
    def __init__(self, ax, profile, errs, minspanx=None,
                 minspany=None, useblit=True):
        print ""
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
        self.phases = np.arange(self.proflen, dtype='d') / self.proflen
        self.errs = errs
        self.visible = True
        self.DCguess = sorted(profile)[len(profile)/10+1]
        self.init_params = [self.DCguess]
        self.ngauss = 0
        self.canvas = ax.figure.canvas
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.canvas.mpl_connect('draw_event', self.update_background)
        self.canvas.mpl_connect('key_press_event', self.keypress)
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
        return (event.inaxes != self.ax or
                event.button != self.eventpress.button)

    def press(self, event):
        'on button press event'
        # Is the correct button pressed within the correct axes?
        if self.ignore(event): return
        # make the drawed box/line visible get the click-coordinates,
        # button, ...
        self.eventpress = event
        if event.button == 1:
            self.to_draw.set_visible(self.visible)
            self.eventpress.ydata = self.DCguess

    def release(self, event):
        'on button release event'
        if self.eventpress is None or self.ignore(event): return
        # release coordinates, button, ...
        self.eventrelease = event
        if event.button == 1:
            # make the box/line invisible again
            self.to_draw.set_visible(False)
            self.canvas.draw()
            xmin, ymin = self.eventpress.xdata, self.eventpress.ydata
            xmax, ymax = self.eventrelease.xdata, self.eventrelease.ydata
            # calculate dimensions of box 
            if xmin > xmax: xmin, xmax = xmax, xmin
            if ymin > ymax: ymin, ymax = ymax, ymin
            spanx = xmax - xmin
            spany = ymax - ymin
            xproblems = self.minspanx is not None and spanx < self.minspanx
            yproblems = self.minspany is not None and spany < self.minspany
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
        x, y = event.xdata, event.ydata         # actual position 
                                                # with button still pressed
        minx, maxx = self.eventpress.xdata, x   # click-x and actual mouse-x
        miny, maxy = self.eventpress.ydata, y   # click-y and actual mouse-y
        if minx > maxx: minx, maxx = maxx, minx # get them in the right order
        if miny > maxy: miny, maxy = maxy, miny
        self.to_draw.set_x(minx)                # set lower left of box
        self.to_draw.set_y(miny)
        self.to_draw.set_width(maxx - minx)     # set width and height of box
        self.to_draw.set_height(maxy - miny)
        self.update()

    def keypress(self, event):
        'on key press event'
        if self.ignore(event): return
        self.eventpress = event
        if event.key == 'q':
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
        for ii in xrange(self.ngauss):
            loc, wid, amp = params[(1 + ii*3):(4 + ii*3)]
            plt.plot(self.phases, DC + amp*gaussian_profile(self.proflen, loc,
                wid), '%s'%cols[ii])

    def onselect(self):
        event1 = self.eventpress
        event2 = self.eventrelease
        # Left mouse button = add a gaussian
        if event1.button == event2.button == 1:
            x1, y1 = event1.xdata, event1.ydata
            x2, y2 = event2.xdata, event2.ydata
            loc = 0.5 * (x1 + x2)
            wid = np.fabs(x2 - x1)
            #amp = np.fabs(1.05 * (y2 - self.init_params[0]) * (x2 - x1))
            amp = np.fabs(1.05 * (y2 - self.init_params[0]))
            self.init_params += [loc, wid, amp]
            self.ngauss += 1
            self.plot_gaussians(self.init_params)
            plt.draw()
        # Middle mouse button = fit the gaussians
        elif event1.button == event2.button == 2:
            fit_params, chi_sq, dof, residuals = fit_gaussian_profile(
                    self.profile, self.init_params, np.zeros(self.proflen) +
                    self.errs, quiet=True)
            self.fit_params = fit_params
            # scaled uncertainties
            #scaled_fit_errs = fit_errs * np.sqrt(chi_sq / dof)

            # Plot the best-fit profile
            self.plot_gaussians(fit_params)
            fitprof = gen_gaussian_profile(fit_params, self.proflen)
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
            if self.ngauss:
                self.init_params = self.init_params[:-3]
                self.ngauss -= 1
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


if __name__ == "__main__":

    from optparse import OptionParser

    usage = "usage: %prog [Options]"
    parser = OptionParser(usage)
    #parser.add_option("-h", "--help",
    #                  action="store_true", dest="help", default=False,
    #                  help="Show this help message and exit.")
    parser.add_option("-d", "--datafile",
                      action="store", metavar="archive", dest="datafile",
                      help="PSRCHIVE archive from which to generate Gaussian portrait.")
    parser.add_option("-o", "--outfile",
                      action="store", metavar="outfile", dest="outfile",
                      help="Name of output model file name. [default=archive.model]")
    parser.add_option("-m", "--model_name",
                      action="store", metavar="model_name", dest="model_name",
                      help="Name given to model. [default=PSRCHIVE Source name]")
    parser.add_option("--freq",
                      action="store", metavar="freq", dest="nu_ref", default=None,
                      help="Reference frequency [MHz] for the gaussian model; the initial profile to fit will be centered on this freq. [default=PSRCHIVE weighted center frequency]")
    parser.add_option("--bw",
                      action="store", metavar="bw", dest="bw_ref", default=None,
                      help="Used with --freq; amount of bandwidth [MHz] centered on nu_ref to average for the initial profile fit. [default=Full bandwidth]")
    parser.add_option("--fixloc",
                      action="store_true", dest="fixloc", default=False,
                      help="Fix locations of gaussians across frequency. [default=False]")
    parser.add_option("--fixwid",
                      action="store_true", dest="fixwid", default=False,
                      help="Fix widths of gaussians across frequency. [default=False]")
    parser.add_option("--fixamp",
                      action="store_true", dest="fixamp", default=False,
                      help="Fix amplitudes of gaussians across frequency. [default=False]")
    parser.add_option("--niter",
                      action="store", metavar="int", dest="niter", default=0,
                      help="Number of iterations to loop for generating better model. [default=0]")
    parser.add_option("--figure", metavar="figurename",
                      action="store", dest="figure", default=None,
                      help="Save PNG figure of final fit to figurename. [default=Not saved]")
    parser.add_option("--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="More to stdout.")

    (options, args) = parser.parse_args()

    if options.datafile is None:
        print "\nppgauss.py - generates gaussian-component model portrait\n"
        parser.print_help()
        parser.exit()

    datafile = options.datafile
    outfile = options.outfile
    model_name = options.model_name
    if options.nu_ref: nu_ref = float(options.nu_ref)
    else: nu_ref = options.nu_ref
    if options.bw_ref: bw_ref = float(options.bw_ref)
    else: bw_ref = options.bw_ref
    fixloc = options.fixloc
    fixwid = options.fixwid
    fixamp = options.fixamp
    niter = int(options.niter)
    figure = options.figure
    quiet = not options.verbose

    dp = DataPortrait(datafile, quiet)
    dp.make_gaussian_model_portrait(ref_prof=(nu_ref, bw_ref), locparams=0.0,
            fixloc=fixloc, widparams=0.0, fixwid=fixwid, ampparams=0.0,
            fixamp=fixamp, niter=niter, writemodel=True, outfile=outfile,
            model_name=model_name, residplot=figure, quiet=quiet)
