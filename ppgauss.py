#!/usr/bin/env python


from matplotlib.patches import Rectangle
from pplib import *


class DataPortrait:
    """
    """
    def __init__(self, datafile=None, metafile=None, quiet=False):
        ""
        ""
        self.init_params = []
        self.datafile = datafile
        self.metafile = metafile
        if self.metafile is None:
            self.data = load_data(datafile, dedisperse=True,
                    dededisperse=False, tscrunch=True, pscrunch=True,
                    rm_baseline=True, flux_prof=True, norm_weights=True,
                    quiet=quiet)
            #Unpack the data dictionary into the local namespace;
            #see load_data for dictionary keys.
            for key in self.data.keys():
                exec("self." + key + " = self.data['" + key + "']")
            if self.source is None: self.source = "noname"
            self.port = (self.masks * self.subints)[0,0]
            self.portx = self.subintsxs[0][0]
        else:
            self.data = concatenate_ports(metafile, quiet=quiet)
            #Unpack the data dictionary into the local namespace;
            #see concatenate_ports for dictionary keys.
            for key in self.data.keys():
                exec("self." + key + " = self.data['" + key + "']")
            self.freqsxs = [self.freqsx]
            self.Ps = np.array([self.P])
            self.weights = np.array([self.weights])

    def fit_profile(self, profile):
        """
        """
        fig = plt.figure()
        profplot = fig.add_subplot(211)
        #Maybe can do better than self.noise_std below
        interactor = GaussianSelector(profplot, profile, self.noise_std,
                minspanx=None, minspany=None, useblit=True)
        plt.show()
        self.init_params = interactor.fitted_params
        self.ngauss = (len(self.init_params) - 1) / 3

    def fit_flux_profile(self, guessA=1.0, guessalpha=0.0, fit=True, plot=True,
            quiet=False):
        """
        Will fit a power law across frequency in a portrait by bin scrunching.
        This should be the usual average pulsar power-law spectrum.  The plot
        will show obvious scintles.
        """
        if fit:
            params, param_errs, chi2, dof, residuals = fit_powlaw(
                    self.flux_profx, np.array([guessA,guessalpha]),
                    self.noise_std, self.freqsxs[0], self.nu0)
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


    def make_gaussian_model(self, modelfile=None,
            ref_prof=(None, None), fixloc=False, fixwid=False, fixamp=False,
            niter=0, writemodel=False, outfile=None, model_name=None,
            residplot=None, quiet=False):
        """
        """
        if modelfile:
            (self.model_name, self.nu_ref, self.ngauss, self.init_model_params,
                    self.fit_flags) = read_model(modelfile)
        else:
            if model_name is None:
                self.model_name = self.source
            else:
                self.model_name = model_name
            #Fit the profile
            if not len(self.init_params):
                self.nu_ref = ref_prof[0]
                self.bw_ref = ref_prof[1]
                if self.nu_ref is None: self.nu_ref = self.nu0
                if self.bw_ref is None: self.bw_ref = abs(self.bw)
                okinds = np.compress(np.less(self.nu_ref - (self.bw_ref/2),
                    self.freqs) * np.greater(self.nu_ref + (self.bw_ref/2),
                    self.freqs) * self.weights[0], np.arange(self.nchan))
                #The below profile average gives a slightly different set of
                #values for the profile than self.profile, if given the full
                #band and center frequency.  Unsure why; shouldn't matter.
                profile = np.take(self.port, okinds, axis=0).mean(axis=0)
                self.fit_profile(profile)
            #All slopes, spectral indices start at 0.0
            locparams = widparams = ampparams = np.zeros(self.ngauss)
            self.init_model_params = np.empty([self.ngauss, 6])
            for igauss in xrange(self.ngauss):
                self.init_model_params[igauss] = np.array(
                        [self.init_params[1::3][igauss], locparams[igauss],
                            self.init_params[2::3][igauss], widparams[igauss],
                            self.init_params[3::3][igauss], ampparams[igauss]])
            self.init_model_params = np.array([self.init_params[0]] +
                list(np.ravel(self.init_model_params)))
            self.fit_flags = np.ones(len(self.init_model_params))
            self.fit_flags[2::6] *= not(fixloc)
            self.fit_flags[4::6] *= not(fixwid)
            self.fit_flags[6::6] *= not(fixamp)
        #The noise...
        self.portx_noise = np.outer(get_noise(self.portx, chans=True),
                np.ones(self.nbin))
        channel_SNRs = self.portx.std(axis=1) / self.portx_noise[:, 0]
        self.nu_fit = guess_fit_freq(self.freqsxs[0], channel_SNRs)
        #Here's the loop
        if niter < 0: niter = 0
        self.niter = niter
        self.itern = niter
        self.model_params = self.init_model_params
        self.total_time = 0.0
        self.start = time.time()
        #if not quiet:
        #    print "Fitting gaussian model portrait..."
        print "Fitting gaussian model portrait..."
        iterator = self.model_iteration(quiet)
        iterator.next()
        self.cnvrgnc = self.check_convergence(efac=1.0, quiet=quiet)
        while (self.niter and not self.cnvrgnc):
            if self.cnvrgnc:
                break
            else:
                if not quiet:
                    print "\nRotating data portrait for iteration %d."%(
                            self.itern - self.niter + 1)
                self.port = rotate_portrait(self.port, self.phi, self.DM,
                        self.Ps[0], self.freqs, self.nu_fit)
                self.portx = rotate_portrait(self.portx, self. phi, self.DM,
                        self.Ps[0], self.freqsxs[0], self.nu_fit)
            if not quiet:
                print "Fitting gaussian model portrait..."
            iterator.next()
            self.niter -= 1
            self.cnvrgnc = self.check_convergence(efac=1.0, quiet=quiet)
        if not quiet:
            print "Residuals mean: %.3f"%(self.portx - self.modelx).mean()
            print "Residuals std:  %.3f"%(self.portx - self.modelx).std()
            print "Data std:       %.3f\n"%self.noise_std
            print "Total fit time: %.2f min"%(self.total_time / 60.0)
            print "Total time:     %.2f min\n"%((time.time() - self.start) /
                    60.0)
        if writemodel:
            if outfile is None:
                if self.metafile is not None:
                    outfile = self.metafile + ".gmodel"
                else:
                    outfile = self.datafile + ".gmodel"
            write_model(outfile, self.model_name, self.nu_ref,
                    self.model_params, self.fit_flags)
        if residplot:
            resids = self.port - self.model_masked
            titles = ("%s"%self.datafile, "%s"%self.model_name, "Residuals")
            show_residual_plot(self.port, self.model, resids, self.phases,
                    self.freqs, titles, bool(self.bw < 0), savefig=residplot)

    def model_iteration(self, quiet=False):
        """
        """
        while (1):
            start = time.time()
            self.fitted_params, self.chi_sq, self.dof = (
                    fit_gaussian_portrait(self.portx, self.model_params,
                        self.portx_noise, self.fit_flags, self.phases,
                        self.freqsxs[0], self.nu_ref, quiet=quiet))
            self.model_params = self.fitted_params
            self.model = gen_gaussian_portrait(self.model_params,
                    self.phases, self.freqs, self.nu_ref)
            self.model_masked = np.transpose(self.weights[0] *
                    np.transpose(self.model))
            self.modelx = np.compress(self.weights[0], self.model, axis=0)
            #Currently, fit_phase_shift returns an unbounded phase
            phase_guess = fit_phase_shift(self.portx.mean(axis=0),
                    self.modelx.mean(axis=0)).phase
            phase_guess %= 1
            if phase_guess > 0.5:
                phase_guess -= 1.0
            DM_guess = 0.0
            (self.phi, self.DM, self.scalesx, param_errs, nu_zero, covar,
                    self.red_chi2, self.fit_duration, self.nfeval, self.rc) = (
                        fit_portrait(self.portx, self.modelx,
                            np.array([phase_guess, DM_guess]), self.Ps[0],
                            self.freqsxs[0], self.nu_fit,
                            bounds=[(None, None), (None, None)], id=None,
                            quiet=True))
            self.phierr = param_errs[0]
            self.DMerr = param_errs[1]
            self.duration = time.time() - start
            self.total_time += self.duration
            yield

    def check_convergence(self, efac=1.0, quiet=False):
        if not quiet:
            print "Iter %d:"%(self.itern - self.niter)
            print " duration of %.2f min"%(self.duration /  60.)
            print " phase offset of %.2e +/- %.2e [rot]"%(self.phi,
                self.phierr)
            print " DM of %.2e +/- %.2e [cm**-3 pc]"%(self.DM, self.DMerr)
            print " red. chi**2 of %.2f."%self.red_chi2
        else:
            if self.niter and (self.itern - self.niter) != 0:
                print "Iter %d..."%(self.itern - self.niter)
        if min(abs(self.phi), abs(1 - self.phi)) < abs(self.phierr)*efac:
            if abs(self.DM) < abs(self.DMerr)*efac:
                print "\nIteration converged.\n"
                return 1
        else:
            return 0

    def show_data_portrait(self):
        """
        """
        title = "%s Portrait"%self.source
        show_portrait(self.port, self.phases, self.freqs, title, True, True,
                bool(self.bw < 0))

    def show_model_fit(self):
        """
        """
        resids = self.port - self.model_masked
        titles = ("%s"%self.datafile, "%s"%self.model_name, "Residuals")
        show_residual_plot(self.port, self.model, resids, self.phases,
                self.freqs, titles, bool(self.bw < 0))

class GaussianSelector:
    def __init__(self, ax, profile, errs, minspanx=None,
                 minspany=None, useblit=True):
        """
        Ripped and altered from SMR's pygaussfit.py
        """
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
        self.DCguess = sorted(profile)[len(profile)/10 + 1]
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
        for igauss in xrange(self.ngauss):
            loc, wid, amp = params[(1 + igauss*3):(4 + igauss*3)]
            plt.plot(self.phases, DC + amp*gaussian_profile(self.proflen, loc,
                wid), '%s'%cols[igauss])

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
            print "Fitting reference gaussian profile..."
            fitted_params, chi_sq, dof, residuals = fit_gaussian_profile(
                    self.profile, self.init_params, np.zeros(self.proflen) +
                    self.errs, quiet=True)
            self.fitted_params = fitted_params
            # scaled uncertainties
            #scaled_fit_errs = fit_errs * np.sqrt(chi_sq / dof)

            # Plot the best-fit profile
            self.plot_gaussians(fitted_params)
            fitprof = gen_gaussian_profile(fitted_params, self.proflen)
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
                      help="PSRCHIVE archive from which to generate gaussian model.")
    parser.add_option("-M", "--metafile",
                      action="store", metavar="metafile", dest="metafile",
                      help="File containing list of archive file names, each of which represents a unique band; these files are concatenated, without rotation, for the fit. nbin must not differ. This does not yet work right.")
    parser.add_option("-o", "--outfile",
                      action="store", metavar="outfile", dest="outfile",
                      help="Name of output model file name. [default=archive.gmodel or metafile.gmodel]")
    parser.add_option("-m", "--model_name",
                      action="store", metavar="model_name", dest="model_name",
                      help="Name given to model. [default=PSRCHIVE Source name]")
    parser.add_option("--nu_ref",
                      action="store", metavar="nu_ref", dest="nu_ref",
                      default=None,
                      help="Reference frequency [MHz] for the gaussian model; the initial profile to fit will be centered on this freq. [default=PSRCHIVE center frequency]")
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
                      action="store", dest="figure", default=False,
                      help="Save PNG figure of final fit to figurename. [default=Not saved]")
    parser.add_option("--verbose",
                      action="store_false", dest="quiet", default=True,
                      help="More to stdout.")

    (options, args) = parser.parse_args()

    if options.datafile is None and options.metafile is None:
        print "\nppgauss.py - generates gaussian-component model portrait\n"
        parser.print_help()
        print ""
        parser.exit()

    datafile = options.datafile
    metafile = options.metafile
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
    quiet = options.quiet

    dp = DataPortrait(datafile=datafile, metafile=metafile, quiet=quiet)
    dp.make_gaussian_model(modelfile = None,
            ref_prof=(nu_ref, bw_ref), fixloc=fixloc, fixwid=fixwid,
            fixamp=fixamp, niter=niter, writemodel=True, outfile=outfile,
            model_name=model_name, residplot=figure, quiet=quiet)
