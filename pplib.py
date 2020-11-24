from __future__ import division
from __future__ import print_function

from builtins import map
from builtins import object
from builtins import range
# Unfortunately, this has to come first in the event nodes = True
from builtins import str
from builtins import zip

from past.utils import old_div

#########
# pplib #
#########
# pplib contains most of the necessary functions and definitions for the
#    other scripts in the PulsePortraiture package.  See pptoaslib for specific
#    functions for the latest version of pptoas.
# Written by Timothy T. Pennucci (TTP; tim.pennucci@nanograv.org).
# Contributions by Scott M. Ransom (SMR), Paul B. Demorest (PBD), and Emmanuel
#    Fonseca (EF).
###########
# imports #
###########
nodes = False  # Used when needing parallelized operation
if nodes:
    import matplotlib

    matplotlib.use('Agg')

import sys
import subprocess
import time
import pickle
import operator
import numpy as np
import numpy.fft as fft
import scipy.interpolate as si
import scipy.optimize as opt
import scipy.signal as ss

try:
    import lmfit as lm
except ImportError:
    print("No lmfit found.  You will not be able to use ppgauss.py or fit_powlaw().")
try:
    import pywt as pw
except ImportError:
    print(
        "No pywt found.  You will not be able to use wavelet_smooth() and will have limited, no-smoothing functionality in ppspline.py.")
import psrchive as pr
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
from telescope_codes import telescope_code_dict

############
# settings #
############

# Exact dispersion constant (e**2/(2*pi*m_e*c)) (used by PRESTO).
Dconst_exact = 4.148808e3  # [MHz**2 cm**3 pc**-1 s]

# "Traditional" dispersion constant (used by PSRCHIVE,TEMPO,PINT).
Dconst_trad = 0.000241 ** -1  # [MHz**2 cm**3 pc**-1 s]

# Fitted DM values will depend on this choice.  Choose wisely.
Dconst = Dconst_trad

# Power-law index for scattering law
scattering_alpha = -4.0

# Use get_noise and default_noise_method for noise levels instead of PSRCHIVE;
# see load_data.
use_get_noise = True

# Default get_noise method (see functions get_noise_*).
# _To_be_improved_.
default_noise_method = 'PS'

# Ignore 0-frequency (sum) component in Fourier fit if F0_fact == 0, else set
# F0_fact == 1.
F0_fact = 0

# Upper limit on the width of a Gaussian component to "help" in fitting.
# Should be either None or > 0.0.
wid_max = 0.25

# default_model is the default model_code used for generating Gaussian models.
# The value of a digit labels an evolutionary function that has one evolutionary
# parameter, in addition to the reference value.  0 = power-law, 1 = linear,
# ...add your own below (see evolve_parameter function).  The order of the
# digits corresponds to the Gaussian model parameter (loc,wid,amp).  Unless
# otherwise specified in the model file after CODE, this set of evolutionary
# functions will be used.  This will eventually be overhauled...!!!
default_model = '000'

# binshift is a fudge factor for scattering portrait functions; was -1;
# currently not used.
binshift = 1.0

###########
# display #
###########

# Set colormap preference
# Decent sequential colormaps: gist_heat, pink, copper, Oranges_r, gray, bone,
#    Blues_r, cubehelix, terrain, YlOrBr_r
# see plt.cm for list of available colormaps.
default_colormap = 'gist_heat'
if hasattr(plt.cm, default_colormap):
    plt.rc('image', cmap=default_colormap)
else:
    plt.rc('image', cmap='YlOrBr_r')

# List of colors; can do this better...
cols = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'brown', 'purple', 'pink',
        'b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'brown', 'purple', 'pink',
        'b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'brown', 'purple', 'pink',
        'b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'brown', 'purple', 'pink']

########
# misc #
########

# RCSTRINGS dictionary, for the return codes given by scipy.optimize.fmin_tnc.
# These are only needed for debugging.
RCSTRINGS = {'-1': 'INFEASIBLE: Infeasible (low > up).',
             '0': 'LOCALMINIMUM: Local minima reach (|pg| ~= 0).',
             '1': 'FCONVERGED: Converged (|f_n-f_(n-1)| ~= 0.)',
             '2': 'XCONVERGED: Converged (|x_n-x_(n-1)| ~= 0.)',
             '3': 'MAXFUN: Max. number of function evaluations reach.',
             '4': 'LSFAIL: Linear search failed.',
             '5': 'CONSTANT: All lower bounds are equal to the upper bounds.',
             '6': 'NOPROGRESS: Unable to progress.',
             '7': 'USERABORT: User requested end of minimization.'}


###########
# classes #
###########

class DataBunch(dict):
    """
    Create a simple class instance of DataBunch.

    db = DataBunch(a=1, b=2,....) has attributes a and b, which are callable
    and update-able using either syntax db.a or db['a'].
    """

    def __init__(self, **kwds):
        dict.__init__(self, kwds)
        self.__dict__ = self


class DataPortrait(object):
    """
    DataPortrait is a class that contains the data to which a model is fit.

    This class is also useful for the quick examining of PSRCHIVE archives in
    an interactive python environment.
    """

    def __init__(self, datafile=None, joinfile=None, quiet=False,
                 **load_data_kwargs):
        """
        Unpack all of the data and set initial attributes.

        If datafile is a metafile of PSRCHIVE archives, "join" attributes are
            set, which are used to align the archives.  A large (>3) number of
            archives signficiantly slows the fitting process, and it has only
            been tested for the case that each archive originates from a
            unique receiver.
        joinfile is a file like that which is output by write_join_parameters,
            containing optional parameters to align the multiple archives.
        quiet=True suppresses output.
        """
        self.init_params = []
        self.joinfile = joinfile
        if file_is_type(datafile, "ASCII"):
            self.join_params = []
            self.join_fit_flags = []
            self.join_nchans = [0]
            self.join_nchanxs = [0]
            self.join_ichans = []
            self.join_ichanxs = []
            self.nchans = []
            self.metafile = self.datafile = datafile
            self.datafiles = open(datafile, "r").readlines()
            self.datafiles = [self.datafiles[ifile][:-1] for ifile in
                              range(len(self.datafiles))]
            self.njoin = len(self.datafiles)
            self.Ps = 0.0
            self.nchan = 0
            self.nchanx = 0
            # self.nu0s = []
            self.lofreq = np.inf
            self.hifreq = 0.0
            self.freqs = []
            self.freqsxs = []
            self.port = []
            self.portx = []
            self.flux_prof = []
            self.flux_profx = []
            self.noise_stds = []
            self.noise_stdsxs = []
            self.SNRs = []
            self.SNRsxs = []
            self.weights = []
            self.weightsxs = []
            self.masks = []
            for ifile in range(len(self.datafiles)):
                datafile = self.datafiles[ifile]
                data = load_data(datafile, dedisperse=True, tscrunch=True,
                                 pscrunch=True, fscrunch=False, flux_prof=True,
                                 return_arch=True, quiet=quiet, **load_data_kwargs)
                self.nchan += data.nchan
                self.nchanx += len(data.ok_ichans[0])
                if ifile == 0:
                    self.join_nchans.append(self.nchan)
                    self.join_nchanxs.append(self.nchanx)
                    self.join_params.append(0.0)
                    self.join_fit_flags.append(0)
                    self.join_params.append(data.DM * 0.0)
                    # Change the below to 0 to not fit for first DM
                    # Also see fit_gaussian_portrait(...) to fit for single DM
                    self.join_fit_flags.append(1)
                    self.nbin = data.nbin
                    self.phases = data.phases
                    refprof = data.prof
                    self.source = data.source
                else:
                    self.join_nchans.append(self.nchan)
                    self.join_nchanxs.append(self.nchanx)
                    prof = data.prof
                    phi = -fit_phase_shift(prof, refprof, Ns=self.nbin).phase
                    self.join_params.append(phi)  # Multiply by 0 to fix phase
                    # Change the below to 0 to not fit for phase
                    self.join_fit_flags.append(1)
                    self.join_params.append(data.DM * 0.0)
                    self.join_fit_flags.append(1)
                self.Ps += data.Ps.mean()
                lf = data.freqs.min() - (old_div(abs(data.bw), (2 * data.nchan)))
                if lf < self.lofreq:
                    self.lofreq = lf
                hf = data.freqs.max() + (old_div(abs(data.bw), (2 * data.nchan)))
                if hf > self.hifreq:
                    self.hifreq = hf
                self.freqs.extend(data.freqs[0])
                self.freqsxs.extend(data.freqs[0, data.ok_ichans[0]])
                self.masks.extend(data.masks[0, 0])
                self.port.extend(data.subints[0, 0] * data.masks[0, 0])
                self.portx.extend(data.subints[0, 0, data.ok_ichans[0]])
                self.flux_prof.extend(data.flux_prof)
                self.flux_profx.extend(data.flux_prof[data.ok_ichans[0]])
                self.noise_stds.extend(data.noise_stds[0, 0])
                self.noise_stdsxs.extend(
                    data.noise_stds[0, 0][data.ok_ichans[0]])
                self.SNRs.extend(data.SNRs[0, 0])
                self.SNRsxs.extend(data.SNRs[0, 0][data.ok_ichans[0]])
                self.weights.extend(data.weights[0])
                self.weightsxs.extend(data.weights[0, data.ok_ichans[0]])
            self.Ps /= len(self.datafiles)
            self.Ps = [self.Ps]  # This line is a toy
            self.bw = self.hifreq - self.lofreq
            self.freqs = np.array(self.freqs)
            self.freqsxs = np.array(self.freqsxs)
            self.nu0 = self.freqs.mean()
            self.isort = np.argsort(self.freqs)
            self.isortx = np.argsort(self.freqsxs)
            for ijoin in range(self.njoin):
                join_ichans = np.intersect1d(np.where(self.isort >=
                                                      self.join_nchans[ijoin])[0], np.where(self.isort <
                                                                                            self.join_nchans[
                                                                                                ijoin + 1])[0])
                self.join_ichans.append(join_ichans)
                join_ichanxs = np.intersect1d(np.where(self.isortx >=
                                                       self.join_nchanxs[ijoin])[0], np.where(self.isortx <
                                                                                              self.join_nchanxs[
                                                                                                  ijoin + 1])[0])
                self.join_ichanxs.append(join_ichanxs)
            self.masks = np.array(self.masks)[self.isort]
            self.masks = np.array([[self.masks]])
            self.port = np.array(self.port)[self.isort]
            self.portx = np.array(self.portx)[self.isortx]
            self.flux_prof = np.array(self.flux_prof)[self.isort]
            self.flux_profx = np.array(self.flux_profx)[self.isortx]
            self.noise_stds = np.array(self.noise_stds)[self.isort]
            self.noise_stds = np.array([[self.noise_stds]])  # For consistency
            self.noise_stdsxs = np.array(self.noise_stdsxs)[self.isortx]
            self.SNRs = np.array(self.SNRs)[self.isort]
            self.SNRsxs = np.array(self.SNRsxs)[self.isortx]
            self.weights = np.array([np.array(self.weights)[self.isort]])
            self.weightsxs = np.array([np.array(self.weightsxs)[self.isortx]])
            self.freqs.sort()
            self.freqsxs.sort()
            self.freqs = np.array([self.freqs])
            self.freqsxs = [self.freqsxs]
            self.join_params = np.array(self.join_params)
            self.join_fit_flags = np.array(self.join_fit_flags)
            if self.joinfile:  # Read joinfile
                joinfile_lines = open(self.joinfile, "r").readlines()[-len(
                    self.datafiles):]
                joinfile_lines = [line.split() for line in joinfile_lines]
                try:
                    for ifile in range(len(joinfile_lines)):
                        ijoin = self.datafiles.index(joinfile_lines[ifile][0])
                        phi = np.double(joinfile_lines[ifile][1])
                        if len(joinfile_lines[ifile]) > 3:  # New joinfiles...
                            DM = np.float(joinfile_lines[ifile][3])
                        else:  # Old joinfiles...
                            DM = np.float(joinfile_lines[ifile][2])
                        self.join_params[ijoin * 2] = phi
                        self.join_params[ijoin * 2 + 1] = DM
                except:
                    print("Bad join file.")
            self.all_join_params = [self.join_ichanxs, self.join_params,
                                    self.join_fit_flags]
            if len(self.datafiles) == 1:
                self.data = data
                # Unpack the data dictionary into the local namespace;
                # see load_data for dictionary keys.
                # BWM: since we are updating the object attributes, it's "safe" to 
                # directly update the object __dict__
                self.__dict__.update(**self.data)
                # for key in list(self.data.keys()):
                #     exec("self." + key + " = self.data['" + key + "']")
        else:
            self.njoin = 0
            self.join_params = []
            self.join_ichans = []
            self.all_join_params = []
            self.datafile = datafile
            self.datafiles = [datafile]
            self.data = load_data(datafile, dedisperse=True,
                                  dededisperse=False, tscrunch=True, pscrunch=True,
                                  fscrunch=False, flux_prof=True, refresh_arch=True,
                                  return_arch=True, quiet=quiet, **load_data_kwargs)
            # Unpack the data dictionary into the local namespace;
            # see load_data for dictionary keys.
            self.__dict__.update(**self.data)
            # for key in list(self.data.keys()):
            #     exec("self." + key + " = self.data['" + key + "']")
            if self.source is None: self.source = "noname"
            self.port = (self.masks * self.subints)[0, 0]
            self.portx = self.port[self.ok_ichans[0]]
            self.flux_profx = self.flux_prof[self.ok_ichans[0]]
            self.freqsxs = [self.freqs[0, self.ok_ichans[0]]]
            self.noise_stdsxs = self.noise_stds[0, 0, self.ok_ichans[0]]
            self.SNRsxs = self.SNRs[0, 0, self.ok_ichans[0]]

    def apply_joinfile(self, nu_ref, undo=False):
        """
        Apply parameters in joinfile.

        nu_ref is the reference frequency [MHz], which should be the fitted
            model's reference frequency.
        undo=True rotates the data the other way.
        """
        undo = (-1) ** (int(undo))
        for ii in range(self.njoin):
            jic = self.join_ichans[ii]
            self.port[jic] = rotate_data(self.port[jic],
                                         -self.join_params[0::2][ii] * undo,
                                         -self.join_params[1::2][ii] * undo, self.Ps[0],
                                         self.freqs[0, jic], nu_ref)
            jicx = self.join_ichanxs[ii]
            self.portx[jicx] = rotate_data(self.portx[jicx],
                                           -self.join_params[0::2][ii] * undo,
                                           -self.join_params[1::2][ii] * undo, self.Ps[0],
                                           self.freqsxs[0][jicx], nu_ref)
        #    self.model[jic] = rotate_data(self.model[jic],
        #            -self.join_params[0::2][ii]*undo,
        #            -self.join_params[1::2][ii]*undo, self.Ps[0],
        #            self.freqs[0,jic], nu_ref)
        # self.model_masked = self.model * self.masks[0,0]
        # self.modelx = np.compress(self.masks[0,0].mean(axis=1), self.model,
        #        axis=0)

    def normalize_portrait(self, method="rms"):
        """
        Normalize each channel's profile using normalize_portrait(...).

        NB: currently only works properly when nsub = 1.
        """
        if method not in ("mean", "max", "prof", "rms", "abs"):
            print("Unknown method for normalize_portrait(...), '%s'." % method)
        else:
            if method == "prof":
                weights = self.weights[0]
                weightsx = self.weights[self.weights > 0]
            else:
                weights = weightsx = None
            # Full portrait
            self.unnorm_noise_stds = np.copy(self.noise_stds)
            self.port, self.norm_values = normalize_portrait(self.port, method,
                                                             weights=weights, return_norms=True)
            self.noise_stds[0, 0] = get_noise(self.port, chans=True)
            self.flux_prof = self.port.mean(axis=1)
            # Condensed portrait
            self.unnorm_noise_stdsxs = np.copy(self.noise_stdsxs)
            self.portx = normalize_portrait(self.portx, method,
                                            weights=weightsx, return_norms=False)
            self.noise_stdsxs = get_noise(self.portx, chans=True)
            self.flux_profx = self.portx.mean(axis=1)

    def unnormalize_portrait(self):
        """
        Undo normalize_portrait.
        """
        if hasattr(self, 'unnorm_noise_stds'):
            self.port = (self.norm_values * self.port.transpose()).transpose()
            self.noise_stds = np.copy(self.unnorm_noise_stds)
            del (self.unnorm_noise_stds)
            self.flux_prof = self.port.mean(axis=1)
            self.portx = (self.norm_values[self.ok_ichans[0]] * \
                          self.portx.transpose()).transpose()
            self.noise_stdsxs = np.copy(self.unnorm_noise_stdsxs)
            del (self.unnorm_noise_stdsxs)
            self.flux_profx = self.portx.mean(axis=1)
            self.norm_values = np.ones(len(self.port))

    def smooth_portrait(self, smart=False, **kwargs):
        """
        Smooth portrait data using default settings from wavelet_smooth.

        smart=True uses smart_smooth(...).
        **kwargs get passed to wavelet_smooth and/or smart_smooth.
        """
        # Full portrait
        if smart:
            self.port = smart_smooth(self.port, try_nlevels=min(8,
                                                                int(np.log2(self.nbin))), **kwargs)
        else:
            self.port = wavelet_smooth(self.port, **kwargs)
        for ichan in range(len(self.port)):
            self.noise_stds[0, 0, ichan] = get_noise(self.port[ichan])
        self.flux_prof = self.port.mean(axis=1)
        # Condensed portrait
        if smart:
            self.portx = smart_smooth(self.portx, try_nlevels=min(8,
                                                                  int(np.log2(self.nbin))), **kwargs)
        else:
            self.portx = wavelet_smooth(self.portx, **kwargs)
        for ichanx in range(len(self.portx)):
            self.noise_stdsxs[ichanx] = get_noise(self.portx[ichanx])
        self.flux_profx = self.portx.mean(axis=1)

    def fit_flux_profile(self, channel_errs=None, nu_ref=None, guessA=1.0,
                         guessalpha=0.0, plot=True, savefig=False, quiet=False):
        """
        Fit a power-law to the phase-averaged flux spectrum of the data.

        Fitted parameters and uncertainties are added as class attributes.

        guessA is the initial amplitude parameter.
        guessalpha is the initial spectral index parameter.
        plot=True shows the fit results.
        savefig specifies a string for a saved figure; will not show the plot.
        quiet=True suppresses output.
        """
        if nu_ref is None: nu_ref = self.nu0
        # Noise level below may be off
        if channel_errs is None: channel_errs = np.ones(len(self.freqsxs[0]))
        fp = fit_powlaw(self.flux_profx, np.array([guessA, guessalpha]),
                        channel_errs, self.freqsxs[0], nu_ref)
        if not quiet:
            print("")
            print("Flux-density power-law fit")
            print("----------------------------------")
            print("residual mean = %.2f" % fp.residuals.mean())
            print("residual std. = %.2f" % fp.residuals.std())
            print("reduced chi-squared = %.2f" % (old_div(fp.chi2, fp.dof)))
            print("A = %.3f +/- %.3f (flux at %.2f MHz)" % (fp.amp,
                                                            fp.amp_err, fp.nu_ref))
            print("alpha = %.3f +/- %.3f" % (fp.alpha, fp.alpha_err))
        if plot or savefig:
            ax1 = plt.subplot(211, position=(0.1, 0.1, 0.8, 0.4))
            ax2 = plt.subplot(212, position=(0.1, 0.5, 0.8, 0.4))
            ax1.errorbar(self.freqsxs[0], fp.residuals, channel_errs, fmt='r+')
            plot_freqs = np.linspace(self.freqs[0].min(), self.freqs[0].max(),
                                     1000)
            ax2.plot(plot_freqs, powlaw(plot_freqs, fp.nu_ref, fp.amp,
                                        fp.alpha), 'k-')
            ax2.errorbar(self.freqsxs[0], self.flux_profx, channel_errs,
                         fmt='r+')
            ax1.set_xlim(self.freqs[0].min(), self.freqs[0].max())
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticklabels([])
            ax1.set_yticks(ax1.get_yticks()[1:-1])
            ax2.set_yticks(ax2.get_yticks()[1:-1])
            ax2.text(0.05, 0.1, r"A$_{\nu_0}$ = %.2f $\pm$ %.2f" % (
                fp.amp, fp.amp_err) + "\n" + r"$\alpha$ = %.2f $\pm$ %.2f" % (
                         fp.alpha, fp.alpha_err), ha="left", va="bottom",
                     transform=ax2.transAxes)
            ax1.set_xlabel("Frequency [MHz]")
            ax1.set_ylabel("Residual")
            ax2.set_ylabel("Flux")
            ax2.set_title("Average Flux Profile for %s" % self.source)
            if savefig: plt.savefig(savefig)
            if plot: plt.show()
        self.flux_fit = fp
        self.spect_A = fp.amp
        self.spect_A_err = fp.amp_err
        self.spect_A_ref = fp.nu_ref
        self.spect_index = fp.alpha
        self.spect_index_err = fp.alpha_err

    def write_join_parameters(self):
        """
        Write the JOIN parameters to file.

        This function is a hack until something better is developed for how to
            deal with these alignment parameters.

        NB: The join parameters are "opposite" of how they are used to rotate
            the data with e.g. rotate_data; use a negative!
        ALSO NB: In order to get "proper barycentic DMs" from these delta-DMs,
                 one must do something like DM = (DM0 + delta-DM)*df, where df
                 is the doppler factor and DM0 is the nominal dispersion
                 measure, both obtained from the archive via PSRCHIVE:
                 e.g.  df = archive.get_Integration(0).get_doppler_factor and
                      DM0 = archive.get_dispersion_measure().
        """
        # print "Beware: JOIN Parameters should be negated!"
        # print "Beware: JOIN DMs are offsets and not doppler corrected!"
        if self.joinfile is not None:
            joinfile = self.joinfile
        else:
            joinfile = self.model_name + ".join"
        jf = open(joinfile, "a")
        header = "# archive name" + " " * 32 + "-phase offset & err [rot]" + \
                 " " * 2 + "-delta-DM & err [cm**-3 pc]\n"
        jf.write(header)
        for ifile in range(len(self.datafiles)):
            datafile = self.datafiles[ifile]
            phase = self.join_params[ifile * 2]
            dm = self.join_params[ifile * 2 + 1]
            line = datafile + " " * abs(45 - len(datafile)) + "% .10f" % phase + \
                   " " * 1 + "%.10f" % self.join_param_errs[::2][ifile] + \
                   " " * 2 + "% .6f" % dm + " " * 1 + \
                   "%.6f" % self.join_param_errs[1::2][ifile] + "\n"
            jf.write(line)
        jf.close()

    def rotate_stuff(self, phase=0.0, DM=0.0, ichans=None, ichanxs=None,
                     nu_ref=None, model=False):
        """
        Rotates the data or model portraits according to rotate_portrait(...).

        phase is a value specifying the amount of achromatic rotation [rot].
        DM is a value specifying the amount of rotation based on the
            cold-plasma dispersion law [cm**-3 pc].
        ichans is an array specifying the channel indices to be rotated; in
            this way, disparate bands can be aligned.  ichans=None defaults to
            all channels.
        ichanxs is the same as ichans, but for the condensed portrait.
        nu_ref is the reference frequency [MHz] that has zero dispersive delay.
            If nu_ref=None, defaults to self.nu0.
        model=True applies the rotation to the model.
        """
        P = self.Ps[0]
        if nu_ref is None: nu_ref = self.nu0
        if ichans is None: ichans = np.arange(len(self.freqs[0]))
        if ichanxs is None: ichanxs = np.arange(len(self.freqsxs[0]))
        freqs = self.freqs[0][ichans]
        freqsxs = self.freqsxs[0][ichanxs]

        if not model:
            self.port[ichans] = rotate_portrait(self.port[ichans], phase, DM,
                                                P, freqs, nu_ref)
            self.portx[ichanxs] = rotate_portrait(self.portx[ichanxs], phase,
                                                  DM, P, freqsxs, nu_ref)
            if hasattr(self, 'prof'):
                self.prof = rotate_portrait([self.prof], phase)[0]
            if hasattr(self, 'mean_prof'):
                self.mean_prof = rotate_portrait([self.mean_prof], phase)[0]
            if hasattr(self, 'eigvec'):
                self.eigvec = rotate_portrait(self.eigvec.T, phase).T

        if model and hasattr(self, 'model'):
            self.model[ichans] = rotate_portrait(self.model[ichans], phase, DM,
                                                 P, freqs, nu_ref)
            self.modelx[ichanxs] = rotate_portrait(self.modelx[ichanxs], phase,
                                                   DM, P, freqsxs, nu_ref)
            self.model_masked[ichans] = rotate_portrait(
                self.model_masked[ichans], phase, DM, P, freqs, nu_ref)
            if hasattr(self, 'smooth_mean_prof'):
                self.smooth_mean_prof = rotate_portrait(
                    [self.smooth_mean_prof], phase)[0]
            if hasattr(self, 'smooth_eigvec'):
                self.smooth_eigvec = rotate_portrait(self.smooth_eigvec.T,
                                                     phase).T

    def unload_archive(self, outfile=None, quiet=False):
        """
        Unload a PSRCHIVE archive to outfile containing the same data.

        outfile=None overwrites the input archive.
        quiet=True suppresses output.

        Only works if input datafile is a PSRCHIVE archive.  The written
            archive will have the same weighted channels as the input datafile.
        """
        if hasattr(self, 'arch'):
            if outfile is None: outfile = self.datafile
            shape = self.arch.get_data().shape
            nsub, npol, nchan, nbin = shape
            data = np.zeros(shape)
            for isub in range(nsub):
                for ipol in range(npol):
                    data[isub, ipol] = self.port
            unload_new_archive(data, self.arch, outfile,
                               DM=self.arch.get_dispersion_measure(),
                               # dmc=int(self.arch.get_dedispersed()), weights=self.weights,
                               dmc=self.dmc, weights=self.weights,
                               quiet=quiet)

    def write_model_archive(self, outfile, quiet=False):
        """
        Write a PSRCHIVE archive to outfile containing the model.

        outfile is the name of the written archive.
        quiet=True suppresses output.

        Only works if input datafile is a PSRCHIVE archive.  The written
            archive will have the same weighted channels as the input datafile.
        """
        if hasattr(self, 'model') and hasattr(self, 'arch'):
            shape = self.arch.get_data().shape
            nsub, npol, nchan, nbin = shape
            model_data = np.zeros(shape)
            for isub in range(nsub):
                for ipol in range(npol):
                    model_data[isub, ipol] = self.model
            unload_new_archive(model_data, self.arch, outfile, DM=0.0, dmc=0,
                               weights=self.weights, quiet=quiet)

    def show_data_portrait(self, **kwargs):
        """
        Show the data portrait.

        See show_portrait(...)
        """
        title = "%s Data Portrait" % self.source
        show_portrait(self.port * self.masks[0, 0], self.phases, self.freqs[0],
                      title, True, True, bool(self.bw < 0), **kwargs)

    def show_model_portrait(self, **kwargs):
        """
        Show the masked model portrait.

        See show_portrait(...)
        """
        if not hasattr(self, 'model'): return None
        title = "%s Model Portrait" % self.source
        show_portrait(self.model * self.masks[0, 0], self.phases, self.freqs[0],
                      title, True, True, bool(self.bw < 0), **kwargs)

    def show_model_fit(self, **kwargs):
        """
        Show the model, data, and residuals.

        See show_residual_plot(...)
        """
        if not hasattr(self, 'model'): return None
        resids = self.port - self.model_masked
        titles = ("%s" % self.datafile, "%s" % self.model_name, "Residuals")
        show_residual_plot(self.port, self.model, resids, self.phases,
                           self.freqs[0], self.noise_stds[0, 0], 0, titles,
                           bool(self.bw < 0), **kwargs)


#############
# functions #
#############

def set_colormap(colormap):
    """
    Set the default colormap to colormap and apply to current image if any.

    See help(colormaps) for more information

    Stolen from matplotlib.pyplot: plt.pink().
    """
    plt.rc("image", cmap=colormap)
    im = plt.gci()

    if im is not None:
        cmap = plt.get_cmap(colormap)
        im.set_cmap(cmap)
    plt.draw_if_interactive()


def get_bin_centers(nbin, lo=0.0, hi=1.0):
    """
    Return nbin bin centers with extremities at lo and hi.

    nbin is the number of bins to return the centers of.
    lo is the value of the ``left-edge'' of the first bin.
    hi is the value of the ``right-edge'' of the last bin.
    """
    lo = np.double(lo)
    hi = np.double(hi)
    diff = hi - lo
    bin_centers = np.linspace(lo + old_div(diff, (nbin * 2)), hi - old_div(diff, (nbin * 2)), nbin)
    bin_centers = np.double(bin_centers)
    return bin_centers


def count_crossings(x, x0):
    """
    Return number of crossings in the 1-d array x across threshold x0.

    x is the 1-D array of values.
    x0 is the threshold across which to count crossings.
    """
    ncross = (np.diff(np.sign(x - x0)) != 0).sum() - ((x - x0) == 0).sum()
    return ncross


def weighted_mean(data, errs=1.0):
    """
    Return the weighted mean and its standard error.

    data is a 1-D array of data values.
    errs is a 1-D array of data errors a la 1-sigma uncertainties (errs**-2 are
        the weights in the weighted mean).
    """
    if hasattr(errs, 'is_integer'):
        errs = np.ones(len(data))
    iis = np.where(errs > 0.0)[0]
    mean = old_div((data[iis] * (errs[iis] ** -2.0)).sum(), (errs[iis] ** -2.0).sum())
    mean_std_err = (errs[iis] ** -2.0).sum() ** -0.5
    return mean, mean_std_err


def get_WRMS(data, errs=1.0):
    """
    Return the weighted root-mean-square value.  Mostly untested.

    data is a 1-D array of data values.
    errs is a 1-D array of data errors a la 1-sigma uncertainties (errs**-2 are
        the weights in the weighted mean).
    """
    if hasattr(errs, 'is_integer'):
        errs = np.ones(len(data))
    iis = np.where(errs > 0.0)[0]
    w_mean = weighted_mean(data, errs)[0]
    d_sum = ((data[iis] - w_mean) ** 2.0 * (errs[iis] ** -2.0)).sum()
    w_sum = (errs[iis] ** -2.0).sum()
    return (old_div(d_sum, w_sum)) ** 0.5


def get_red_chi2(data, model, errs=None, dof=None):
    """
    Return reduced chi-squared given input data and model.

    data is a 1- or 2-D array of data values.
    model is a 1- or 2-D array of model values.
    errs is a value or 1-D array of '1-sigma' uncertainties on the data.  If
        None, errs is estimated using get_noise(data).
    dof is an integer number of degrees of freedom.  If None, dof =
            sum(data.shape).
    """
    resids = data - model
    if errs is None:
        if len(data.shape) == 1:
            errs = get_noise(data)
        elif len(data.shape) == 2:
            errs = get_noise(data, chans=True)
        else:
            print("Can only handle 1- or 2-D input.")
    if dof is None: dof = sum(data.shape)
    if len(data.shape) == 1:
        red_chi2 = old_div(np.sum((old_div(resids, errs)) ** 2.0), dof)
    else:
        red_chi2 = old_div(np.array([(old_div(resids[ii], errs[ii])) ** 2.0 for ii in
                                     range(len(resids))]).sum(), dof)
    return red_chi2


def gaussian_function(xs, loc, wid, norm=False):
    """
    Evaluates a Gaussian function with parameters loc and wid at values xs.

    xs is the array of values that are evaluated in the function.
    loc is the pulse phase location (0-1) [rot].
    wid is the Gaussian pulse's full width at half-max (FWHM) [rot].
    norm=True returns the profile such that the integrated density = 1.
    """
    mean = loc
    sigma = old_div(wid, (2 * np.sqrt(2 * np.log(2))))
    scale = 1.0
    zs = old_div((xs - mean), sigma)
    ys = np.exp(-0.5 * zs ** 2)
    if norm:
        scale *= (sigma ** 2.0 * 2.0 * np.pi) ** -0.5
    return scale * ys


def gaussian_profile(nbin, loc, wid, norm=False, abs_wid=False, zeroout=True):
    """
    Return a Gaussian pulse profile with nbin bins and peak amplitude of 1.

    nbin is the number of bins in the profile.
    loc is the pulse phase location (0-1) [rot].
    wid is the Gaussian pulse's full width at half-max (FWHM) [rot].
    norm=True returns the profile such that the integrated density = 1.
    abs_wid=True, will use abs(wid).
    zeroout=True and wid <= 0, return a zero array.

    Note: The FWHM of a Gaussian is approx 2.35482 "sigma", or exactly
          2*sqrt(2*ln(2)).

    Taken and tweaked from SMR's pygaussfit.py
    """
    # Maybe should move these checks to gen_gaussian_portrait?
    if abs_wid:
        wid = abs(wid)
    if wid > 0.0:
        pass
    elif wid == 0.0:
        return np.zeros(nbin, 'd')
    elif wid < 0.0 and zeroout:
        return np.zeros(nbin, 'd')
    elif wid < 0.0 and not zeroout:
        pass
    else:
        return 0
    sigma = old_div(wid, (2 * np.sqrt(2 * np.log(2))))
    mean = loc % 1.0
    locval = get_bin_centers(nbin, lo=0.0, hi=1.0)
    if (mean < 0.5):
        locval = np.where(np.greater(locval, mean + 0.5), locval - 1.0, locval)
    else:
        locval = np.where(np.less(locval, mean - 0.5), locval + 1.0, locval)
    try:
        zs = old_div((locval - mean), sigma)
        okzinds = np.compress(np.fabs(zs) < 20.0, np.arange(nbin))  # Why 20?
        okzs = np.take(zs, okzinds)
        retval = np.zeros(nbin, 'd')
        np.put(retval, okzinds, old_div(np.exp(-0.5 * (okzs) ** 2.0), (sigma * np.sqrt(2 *
                                                                                       np.pi))))
        if norm:
            return retval
        else:
            if np.max(abs(retval)) == 0.0:
                return retval
            else:
                z = old_div((locval[retval.argmax()] - loc), sigma)
                fact = old_div(np.exp(-0.5 * z ** 2.0), retval[retval.argmax()])
                return fact * retval
    except OverflowError:
        print("Problem in gaussian_profile:  mean = %f  sigma = %f" % (mean,
                                                                       sigma))
        return np.zeros(nbin, 'd')


def gen_gaussian_profile(params, nbin):
    """
    Return a model profile with ngauss Gaussian components.

    params is a sequence of 2 + (ngauss*3) values where the first value is
        the DC component, the second value is the scattering timescale [bin]
        and each remaining group of three represents the Gaussians' loc (0-1),
        wid (i.e. FWHM) (0-1), and amplitude (>0.0).
    nbin is the number of bins in the model.

    Taken and tweaked from SMR's pygaussfit.py
    """
    ngauss = old_div((len(params) - 2), 3)
    model = np.zeros(nbin, dtype='d') + params[0]
    for igauss in range(ngauss):
        loc, wid, amp = params[(2 + igauss * 3):(5 + igauss * 3)]
        model += amp * gaussian_profile(nbin, loc, wid)
    if params[1] != 0.0:
        bins = np.arange(nbin)
        # sk = scattering_kernel(params[1], 1.0, np.array([1.0]), bins, P=1.0,
        #        alpha=scattering_alpha)[0]  #alpha here does not matter
        # model = add_scattering(model, sk, repeat=3)
        sp_FT = scattering_profile_FT(float(params[1]) / nbin, nbin)
        model = np.fft.irfft(sp_FT * np.fft.rfft(model))
    return model


def gen_gaussian_portrait(model_code, params, scattering_index, phases, freqs,
                          nu_ref, join_ichans=[], P=None):
    """
    Return a Gaussian-component model portrait based on input parameters.

    model_code is a three digit string specifying the evolutionary functions
        to be used for the three Gaussian parameters (loc,wid,amp); see
        pplib.py header for details.
    params is an array of 2 + (ngauss*6) + 2*len(join_ichans) values.
        The first value is the DC component, and the second value is the
        scattering timescale [bin].  The next ngauss*6 values represent the
        Gaussians' loc (0-1), evolution parameter in loc, wid (i.e. FWHM)
        (0-1), evolution parameter in wid, amplitude (>0,0), and spectral
        index alpha (no implicit negative).  The remaining 2*len(join_ichans)
        parameters are pairs of phase and DM.  The iith list of channels in
        join_ichans gets rotated in the generated model by the iith pair of
        phase and DM.
    scattering_index is the power-law index of the scattering law; the default
        is set in the header lines of pplib.py.
    phases is the array of phase values (will pass nbin to
        gen_gaussian_profile).
    freqs in the array of frequencies at which to calculate the model.
    nu_ref is the frequency to which the locs, wids, and amps reference.
    join_ichans is used only in ppgauss, in which case the period P [sec] needs
        to be provided.

    The units of the evolution parameters and the frequencies need to match
        appropriately.
    """
    njoin = len(join_ichans)
    if njoin:
        join_params = params[-njoin * 2:]
        params = params[:-njoin * 2]
    # Below, params[1] is multiplied by 0 so that scattering is taken care of
    # outside of gen_gaussian_profile
    refparams = np.array([params[0]] + [params[1] * 0.0] + list(params[2::2]))
    tau = params[1]
    locparams = params[3::6]
    widparams = params[5::6]
    ampparams = params[7::6]
    ngauss = len(refparams[2::3])
    nbin = len(phases)
    nchan = len(freqs)
    gport = np.empty([nchan, nbin])
    gparams = np.empty([nchan, len(refparams)])
    # DC term
    gparams[:, 0] = refparams[0]
    # Scattering term - first make unscattered portrait
    gparams[:, 1] = refparams[1]
    # Locs
    gparams[:, 2::3] = evolve_parameter(freqs, nu_ref, refparams[2::3],
                                        locparams, model_code[0])
    # Wids
    gparams[:, 3::3] = evolve_parameter(freqs, nu_ref, refparams[3::3],
                                        widparams, model_code[1])
    # Amps
    gparams[:, 4::3] = evolve_parameter(freqs, nu_ref, refparams[4::3],
                                        ampparams, model_code[2])
    for ichan in range(nchan):
        # Need to contrain so values don't go negative, etc., which is currently
        # taken care of in gaussian_profile
        gport[ichan] = gen_gaussian_profile(gparams[ichan], nbin)
    if tau != 0.0:
        # sk = scattering_kernel(tau, nu_ref, freqs, np.arange(nbin), 1.0,
        #        alpha=scattering_index)
        # gport = add_scattering(gport, sk, repeat=3)
        taus = scattering_times(float(tau) / nbin, scattering_index, freqs,
                                nu_ref)
        sp_FT = scattering_portrait_FT(taus, nbin)
        gport = np.fft.irfft(sp_FT * np.fft.rfft(gport, axis=-1), axis=-1)
    if njoin:
        for ij in range(njoin):
            join_ichan = join_ichans[ij]
            phi = join_params[0::2][ij]
            DM = join_params[1::2][ij]
            gport[join_ichan] = rotate_data(gport[join_ichan], phi,
                                            DM, P, freqs[join_ichan], nu_ref)
    return gport


def gen_spline_portrait(mean_prof, freqs, eigvec, tck, nbin=None):
    """
    Generate a model portrait from make_spline_model(...) output.

    mean_prof is the mean profile.
    freqs are the frequencies at which to build the model.
    eigvec are the eigenvectors providing the basis for the B-spline curve.
    tck is a tuple containing knot locations, B-spline coefficients, and spline
        degree (output of si.splprep(...)).
    nbin is the number of phase bins to use in the model; if different from
        len(mean_prof), a resampling function is used.
    """
    if not eigvec.shape[1]:
        port = np.tile(mean_prof, len(freqs)).reshape(len(freqs),
                                                      len(mean_prof))
    else:
        proj_port = np.array(si.splev(freqs, tck, der=0, ext=0)).T
        delta_port = np.dot(proj_port, eigvec.T)
        port = delta_port + mean_prof
    if nbin is not None:
        if len(mean_prof) != nbin:
            shift = 0.5 * (nbin ** -1 - len(mean_prof) ** -1)
            port = ss.resample(port, nbin, axis=1)
            port = rotate_portrait(port, shift)  # ss.resample introduces shift!
    return port


def make_constant_portrait(archive, outfile, profile=None, DM=0.0, dmc=False,
                           weights=None, quiet=False):
    """
    Fill an archive with one profile.

    archive is the name of the dummy PSRCHIVE archive that will be filled with
        profile.
    outfile is the name of the unloaded, output archive, which will have the
        same nsub, npol, nchan, nbin, frequencies, etc., as archive.
    profile is the array containing the profile to be used; it must have the
        same number of phase bins nbin as archive.  If None, archive will be
        T-, P-, and F-scrunched and the resulting average profile will be used.
    DM is the header DM to store in the unloaded archive.
    dmc=False stores the unloaded archive in the dispersed state.
    weights is an nsub x nchan array of channel weights; if None, ones are used
        as weights.
    quiet=True suppresses output.
    """
    arch = pr.Archive_load(archive)
    nsub, npol, nchan, nbin = arch.get_data().shape
    if profile is None:
        arch.tscrunch()
        arch.pscrunch()
        arch.fscrunch()
        profile = arch.get_data()[0, 0, 0]
        arch.refresh()
    nbin_check_output = "len(profile) != number of bins in dummy archive"
    assert (len(profile) == nbin), nbin_check_output
    if weights is None:
        weights = np.ones([nsub, nchan])
    data = np.zeros([nsub, npol, nchan, nbin])
    for isub in range(nsub):
        for ipol in range(npol):
            for ichan in range(nchan):
                data[isub, ipol, ichan] = profile
    unload_new_archive(data, arch, outfile, DM=DM, dmc=dmc, weights=weights,
                       quiet=quiet)


def power_law_evolution(freqs, nu_ref, parameter, index):
    """
    Evolve the parameter over freqs by a power-law with index = index.

    F(freq) = parameter * (freq/nu_ref)**index

    freqs is an array of frequencies [MHz] to evolve parameter over.
    nu_ref is the reference frequency [MHz] for parameter (= F(nu_ref)).
    parameter is an array of length ngauss containing the values of the
        parameter at nu_ref.
    index is an array of length ngauss containing the values of the power-law
        index.
    """
    nchan = len(freqs)
    return np.exp(np.outer(np.log(freqs) - np.log(nu_ref), index) + \
                  np.outer(np.ones(nchan), np.log(parameter)))


def linear_evolution(freqs, nu_ref, parameter, slope):
    """
    Evolve the parameter over freqs by a power-law with slope = slope.

    F(freq) = parameter + slope*(freqs - nu_ref)

    freqs is an array of frequencies [MHz] to evolve parameter over.
    nu_ref is the reference frequency [MHz] for parameter (= F(nu_ref)).
    parameter is an array of length ngauss containing the values of the
        parameter at nu_ref.
    slope is an array of length ngauss containing the values of the linear
        slope.
    """
    nchan = len(freqs)
    return np.outer(freqs - nu_ref, slope) + \
           np.outer(np.ones(nchan), parameter)


def evolve_parameter(freqs, nu_ref, parameter, evol_parameter, code):
    """
    Evolve parameter over freqs using a function based on code.

    freqs is an array of frequencies [MHz] to evolve parameter over.
    nu_ref is the reference frequency [MHz] for parameter (= F(nu_ref)).
    parameter is an array of length ngauss containing the values of the
        parameter at nu_ref.
    evol_parameter is an array of length ngauss containing the values of the
       single parameter used by the evolution function.
    code is a single digit string that specifies the evolution function to be
       used; the dictionary of such functions is defined in _this_ function.
    """
    # Dictionary containing the labels for model_code and the function names.
    evolution_functions = {'0': power_law_evolution,
                           '1': linear_evolution}
    return evolution_functions[code](freqs, nu_ref, parameter, evol_parameter)


def powlaw(nu, nu_ref, A, alpha):
    """
    Return a power-law 'spectrum' given by F(nu) = A*(nu/nu_ref)**alpha
    """
    return A * (old_div(nu, nu_ref)) ** alpha


def powlaw_integral(nu2, nu1, nu_ref, A, alpha):
    """
    Return the definite integral of a powerlaw from nu1 to nu2.

    The powerlaw is of the form A*(nu/nu_ref)**alpha.
    """
    alpha = np.float64(alpha)
    if alpha == -1.0:
        return A * nu_ref * np.log(old_div(nu2, nu1))
    else:
        C = old_div(A * (nu_ref ** -alpha), (1 + alpha))
        diff = ((nu2 ** (1 + alpha)) - (nu1 ** (1 + alpha)))
        return C * diff


def powlaw_freqs(lo, hi, N, alpha, mid=False):
    """
    Return frequencies spaced such that each channel has equal flux.

    Given a bandwidth from lo to hi frequencies, split into N channels, and a
        power-law index alpha, this function finds the frequencies such that
        each channel contains the same amount of flux.

    mid=True, returns N frequencies, corresponding to the center frequency in
        each channel. Default behavior returns N+1 frequencies (includes both
        lo and hi freqs).
    """
    alpha = np.float64(alpha)
    nus = np.zeros(N + 1)
    if alpha == -1.0:
        nus = np.exp(np.linspace(np.log(lo), np.log(hi), N + 1))
    else:
        nus = np.power(np.linspace(lo ** (1 + alpha), hi ** (1 + alpha), N + 1),
                       (1 + alpha) ** -1)
        # Equivalently:
        # for ii in range(N+1):
        #    nus[ii] = ((ii / np.float64(N)) * (hi**(1+alpha)) + (1 - (ii /
        #        np.float64(N))) * (lo**(1+alpha)))**(1 / (1+alpha))
    if mid:
        midnus = np.zeros(N)
        for ii in range(N):
            midnus[ii] = 0.5 * (nus[ii] + nus[ii + 1])
        nus = midnus
    return nus


def scattering_kernel(tau, nu_ref, freqs, phases, P, alpha):
    """
    Return a scattering kernel based on input parameters.

    tau is the scattering timescale in [sec] or [bin].
    nu_ref is the reference frequency for tau.
    freqs is the array of center frequencies in the nchan x nbin kernel.
    phases [rot] gives the phase-bin centers of the nchan x nbin kernel; phases
        is in [bin] if tau is also in [bin].
    P is the period [sec]; use P = 1.0 if tau is in units of [bin].
    alpha is the power-law index for the scattering evolution.
    """
    nchan = len(freqs)
    nbin = len(phases)
    if tau == 0.0:
        ts = np.zeros([nchan, nbin])
        ts[:, 0] = 1.0
    else:
        ts = np.array([phases * P for ichan in range(nchan)])
        taus = tau * (old_div(freqs, nu_ref)) ** alpha
        sk = np.exp(-np.transpose(np.transpose(ts) * taus ** -1.0))
    return sk


def add_scattering(port, kernel, repeat=3):
    """
    Add scattering into a portrait.

    port is the nchan x nbin portrait.
    kernel is the scattering kernel to be used in the convolution with port.
    repeat attempts to rid the convolution of edge effects by repeating the
        port and kernel repeat times before convolution; the center portion is
        returned.
    """
    mid = old_div(repeat, 2)
    d = np.array(list(port.transpose()) * repeat).transpose()
    k = np.array(list(kernel.transpose()) * repeat).transpose()
    if len(port.shape) == 1:
        nbin = port.shape[0]
        norm_kernel = old_div(kernel, kernel.sum())
        scattered_port = ss.convolve(norm_kernel, d)[mid * nbin: (mid + 1) *
                                                                 nbin]
    else:
        nbin = port.shape[1]
        norm_kernel = np.transpose(np.transpose(k) * k.sum(axis=1) ** -1)
        scattered_port = np.fft.irfft(np.fft.rfft(norm_kernel) *
                                      np.fft.rfft(d))[:, mid * nbin: (mid + 1) * nbin]
    return scattered_port


def add_scintillation(port, params=None, random=True, nsin=2, amax=1.0,
                      wmax=3.0):
    """
    Add totally fake scintillation to a portrait based on sinusoids.

    port is the nchan x nbin array of data values.
    params are triplets of sinusoidal parameters: "amps", "freqs" [cycles],
        and "phases" [cycles].
    if params is None and random is True, random params for nsin sinusoids are
        chosen from a uniform distribution on [0,amax], a chi-squared
        distribution with wmax dof, and a uniform distribution on [0,1].
    """
    nchan = len(port)
    pattern = np.zeros(nchan)
    if params is None and random is False:
        return port
    elif params is not None:
        nsin = old_div(len(params), 3)
        for isin in range(nsin):
            a, w, p = params[isin * 3:isin * 3 + 3]
            pattern += a * np.sin(np.linspace(0, w * np.pi, nchan) +
                                  p * np.pi) ** 2
    else:
        for isin in range(nsin):
            (a, w, p) = (np.random.uniform(0, amax),
                         np.random.chisquare(wmax), np.random.uniform(0, 1))
            pattern += a * np.sin(np.linspace(0, w * np.pi, nchan) +
                                  p * np.pi) ** 2
    return np.transpose(np.transpose(port) * pattern)


def mean_C2N(nu, D, bw_scint):
    """
    Return mean_C2N [m**(-20/3)]

    For use with scattering measure.
    nu is the frequency [MHz]
    D is the distance [kpc]
    bw_scint is the scintillation bandwidth [MHz]

    Reference: Foster, Fairhead, and Backer (1991)
    """
    return 2e-14 * nu ** (11 / 3.0) * D ** (-11 / 6.0) * bw_scint ** (-5 / 6.0)


def dDM(D, D_screen, nu, bw_scint):
    """
    Return the delta-DM [cm**-3 pc] predicted for a frequency dependent DM.

    D is the distance to the pulsar [kpc]
    D_screen is the distance from the Earth to the scattering screen [kpc]
    nu is the frequency [MHz]
    bw_scint is the scintillation bandwidth at nu [MHz]

    References: Cordes & Shannon (2010); Foster, Fairhead, and Backer (1991)
    """
    # SM is the scattering measure [m**(-20/3) kpc]
    SM = mean_C2N(nu, D, bw_scint) * D
    return 10 ** 4.45 * SM * D_screen ** (5 / 6.0) * nu ** (-11 / 6.0)


def fit_powlaw_function(params, freqs, nu_ref, data=None, errs=None):
    """
    Return the weighted residuals from a power-law model and data.

    params is an array = [amplitude at reference frequency, spectral index].
    freqs is an nchan array of frequencies.
    nu_ref is the frequency at which the amplitude is referenced.
    data is the array of the data values.
    errs is the array of uncertainties on the data values.
    """
    prms = np.array([param.value for param in list(params.values())])
    A = prms[0]
    alpha = prms[1]
    return old_div((data - powlaw(freqs, nu_ref, A, alpha)), errs)


def fit_gaussian_profile_function(params, data=None, errs=None):
    """
    Return the weighted residuals from a Gaussian profile model and data.

    See gen_gaussian_profile for form of input params.
    data is the array of data values.
    errs is the array of uncertainties on the data values.
    """
    prms = np.array([param.value for param in list(params.values())])
    return old_div((data - gen_gaussian_profile(prms, len(data))), errs)


def fit_gaussian_portrait_function(params, model_code, phases, freqs, nu_ref,
                                   data=None, errs=None, join_ichans=None, P=None, ):
    """
    Return the weighted residuals from a Gaussian-component model and data.

    See gen_gaussian_portrait for form of input.
    data is the 2D array of data values.
    errs is the 2D array of the uncertainties on the data values.
    """
    prms = np.array([param.value for param in list(params.values())])
    deviates = np.ravel(old_div((data - gen_gaussian_portrait(model_code, prms[:-1],
                                                              prms[-1], phases, freqs, nu_ref, join_ichans, P)), errs))
    return deviates


def fit_phase_shift_function(phase, model=None, data=None, err=None):
    """
    Return the negative, weighted inverse DFT at phase of model and data.

    phase is the input phase [rot]
    model is the array of model values (Fourier domain)
    data is the array of data values (Fourier domain)
    err is the noise level (Fourier domain)
    """
    harmind = np.arange(len(model))
    phasor = np.exp(harmind * 2.0j * np.pi * phase)
    C = old_div(-np.real((data * np.conj(model) * phasor).sum()), err ** 2.0)
    return C


def fit_phase_shift_function_deriv(phase, model=None, data=None, err=None):
    """
    Return the first derivative of fit_phase_shift_function at phase.

    See fit_phase_shift_function for form of input.
    """
    harmind = np.arange(len(model))
    phasor = np.exp(harmind * 2.0j * np.pi * phase)
    dC = old_div(-np.real((2.0j * np.pi * harmind * data * np.conj(model) *
                           phasor).sum()), err ** 2.0)
    return dC


def fit_phase_shift_function_2deriv(phase, model=None, data=None, err=None):
    """
    Return the second derivative of fit_phase_shift_function at phase.

    See fit_phase_shift_function for form of input.
    """
    harmind = np.arange(len(model))
    phasor = np.exp(harmind * 2.0j * np.pi * phase)
    d2C = old_div(-np.real((-4.0 * (np.pi ** 2.0) * (harmind ** 2.0) * data *
                            np.conj(model) * phasor).sum()), err ** 2.0)
    return d2C


def fit_portrait_function(params, model=None, p_n=None, data=None, errs=None,
                          P=None, freqs=None, nu_ref=np.inf):
    """
    Return the function to be minimized by fit_portrait, evaluated at params.

    The returned value is equivalent to the chi-squared value of the model and
    data, given the input parameters, differing only by a constant depending
    on a weighted, quadratic sum of the data (see 'd' in fit_portrait).
    The quantity is related to the inverse DFT of model and data (i.e. the
    cross-correlation of the time-domain quantities).

    NB: both model and data must already be in the Fourier domain.

    params is an array = [phase, DM], with phase in [rot] and DM in
        [cm**-3 pc].
    model is the nchan x nbin phase-frequency model portrait that has been
        DFT'd along the phase axis.
    p_n is an nchan array containing a quadratic sum of the model (see 'p_n' in
        fit_portrait).
    data is the nchan x nbin phase-frequency data portrait that has been DFT'd
        along the phase axis.
    err is the nchan array of noise level estimates (in the Fourier domain).
    P is the period [s] of the pulsar at the data epoch.
    freqs is an nchan array of frequencies [MHz].
    nu_ref is the frequency [MHz] that is defined to have zero delay from a
        non-zero dispersion measure.
    """
    phase = params[0]
    m = 0.0
    if P is None or freqs is None:
        D = 0.0
        freqs = np.inf * np.ones(len(model))
    else:
        D = old_div(Dconst * params[1], P)
    for nn in range(len(freqs)):
        freq = freqs[nn]
        p = p_n[nn]
        err = errs[nn]
        harmind = np.arange(len(model[nn]))
        phasor = np.exp(harmind * 2.0j * np.pi * (phase + (D * (freq ** -2.0 -
                                                                nu_ref ** -2.0))))
        # Cdp is related to the inverse DFT of the cross-correlation
        Cdp = np.real(data[nn, :] * np.conj(model[nn, :]) * phasor).sum()
        m += old_div((Cdp ** 2.0), (err ** 2.0 * p))
    return -m


def fit_portrait_function_deriv(params, model=None, p_n=None, data=None,
                                errs=None, P=None, freqs=None, nu_ref=np.inf):
    """
    Return the two first-derivatives of fit_portrait_function.

    See fit_portrait_function for form of input.
    """
    phase = params[0]
    D = old_div(Dconst * params[1], P)
    d_phi, d_DM = 0.0, 0.0
    for nn in range(len(freqs)):
        freq = freqs[nn]
        p = p_n[nn]
        err = errs[nn]
        harmind = np.arange(len(model[nn]))
        phasor = np.exp(harmind * 2.0j * np.pi * (phase + (D * (freq ** -2.0 -
                                                                nu_ref ** -2.0))))
        Cdp = np.real(data[nn, :] * np.conj(model[nn, :]) * phasor).sum()
        dCdp1 = np.real(2.0j * np.pi * harmind * data[nn, :] *
                        np.conj(model[nn, :]) * phasor).sum()
        dDM = (freq ** -2.0 - nu_ref ** -2.0) * (old_div(Dconst, P))
        d_phi += old_div(-2 * Cdp * dCdp1, (err ** 2.0 * p))
        d_DM += old_div(-2 * Cdp * dCdp1 * dDM, (err ** 2.0 * p))
    return np.array([d_phi, d_DM])


def fit_portrait_function_2deriv(params, model=None, p_n=None, data=None,
                                 errs=None, P=None, freqs=None, nu_ref=np.inf):
    """
    Return the three second-derivatives of fit_portrait_function.

    The three unique values in the Hessian, which is a 2x2 symmetric matrix of
    the second-derivatives of fit_portrait_function, are returned, evaluated at
    params, as well as the estimate of the zero-covariance reference
    frequency.

    NB: The curvature matrix is one-half the second-derivative of the
    chi-squared function (this function).  The covariance matrix is the
    inverse of the curvature matrix.

    See fit_portrait_function for form of input.
    """
    phase = params[0]
    D = old_div(Dconst * params[1], P)
    d2_phi, d2_DM, d2_cross = 0.0, 0.0, 0.0
    W_n = np.zeros(len(freqs))
    for nn in range(len(freqs)):
        freq = freqs[nn]
        p = p_n[nn]
        err = errs[nn]
        harmind = np.arange(len(model[nn]))
        phasor = np.exp(harmind * 2.0j * np.pi * (phase + (D *
                                                           (freq ** -2.0 - nu_ref ** -2.0))))
        Cdp = np.real(data[nn, :] * np.conj(model[nn, :]) * phasor).sum()
        dCdp1 = np.real(2.0j * np.pi * harmind * data[nn, :] *
                        np.conj(model[nn, :]) * phasor).sum()
        dCdp2 = np.real(pow(2.0j * np.pi * harmind, 2.0) * data[nn, :] *
                        np.conj(model[nn, :]) * phasor).sum()
        dDM = (freq ** -2.0 - nu_ref ** -2.0) * (old_div(Dconst, P))
        W = (pow(dCdp1, 2.0) + (Cdp * dCdp2))
        W_n[nn] = old_div(W, (err ** 2.0 * p))
        d2_phi += old_div(-2.0 * W, (err ** 2.0 * p))
        d2_DM += old_div(-2.0 * W * dDM ** 2.0, (err ** 2.0 * p))
        d2_cross += old_div(-2.0 * W * dDM, (err ** 2.0 * p))
    nu_zero = (old_div(W_n.sum(), np.sum(W_n * freqs ** -2))) ** 0.5
    return (np.array([d2_phi, d2_DM, d2_cross]), nu_zero)


def wiener_filter(prof, noise):
    # FIX does not work
    """
    Return the 'optimal' Wiener filter given a noisy pulse profile.

    <UNDER CONSTRUCTION>

    prof is a noisy pulse profile.
    noise is standard error of the profile.

    Reference: PBD's PhDT
    """
    FFT = fft.rfft(prof)
    pows = old_div(np.real(FFT * np.conj(FFT)), len(prof))
    return old_div(pows, (pows + (noise ** 2)))
    # return (pows - (noise**2)) / pows


def brickwall_filter(N, kc):
    """
    Return a 'brickwall' filter with N points.

    The brickwall filter has the first kc as ones and the remainder as zeros.
    """
    fk = np.zeros(N)
    fk[:kc] = 1.0
    return fk


def fit_brickwall(prof, noise):
    # FIX this is obviously wrong
    """
    Return the index kc for the best-fit brickwall.

    See brickwall_filter and wiener_filter.

    <UNDER CONSTRUCTION>
    """
    wf = wiener_filter(prof, noise)
    N = len(wf)
    X2 = np.zeros(N)
    for ii in range(N):
        X2[ii] = np.sum((wf - brickwall_filter(N, ii)) ** 2)
    return X2.argmin()


def half_triangle_function(a, b, dc, N):
    """
    Return a half-triangle function with base a and height b.

    dc is an overall baseline level.
    N is the length of the returned function.
    """
    fn = np.zeros(N) + dc
    a = int(np.floor(a))
    fn[:a] += -(old_div(np.float64(b), a)) * np.arange(a) + b
    return fn


def find_kc_function(params, data, errs=1.0, fn='exp_dc'):
    """
    Return the (weighted) chi-squared statistic for find_kc.

    params are the input parameters for either a decaying exponential or
        half-triangle function, (a, b, dc), with a as the width parameter, b as
        the height parameter, and dc as the offset parameter.
    data is the array of data values.
    errs is the array of uncertainties on the data.
    fn is either 'half_tri' (half-triangle) or 'exp_dc' (decaying exponential).
    """
    a, b, dc = params[0], params[1], params[2]
    if fn == 'exp_dc':
        model = b * np.exp(-a * np.arange(len(data))) + dc
    elif fn == 'half_tri':
        model = half_triangle_function(a, b, dc, len(data))
    else:
        return 0.0
    return np.sum((old_div((data - model), errs)) ** 2.0)


def find_kc(pows, errs=1.0, fn='exp_dc'):
    """
    Return the critical cutoff index kc based on a half-triangle function fit.

    The function attempts to find where the noise-floor in a power-spectrum
    begins.

    pows is a 1-D array of input power-spectrum amplitudes.
    """
    data = np.log10(pows)
    other_args = [data, errs, fn]
    if fn == 'exp_dc':
        ranges = [tuple((len(data) ** -1, 1.0)),
                  tuple((0, data.max() - data.min())),
                  tuple((data.min(), data.max()))]
    elif fn == 'half_tri':
        ranges = [tuple((1, len(data))), tuple((0, data.max() - data.min())),
                  tuple((data.min(), data.max()))]
    else:
        return 0
    results = opt.brute(find_kc_function, ranges, args=other_args, Ns=20,
                        full_output=False, finish=None)
    a, b, dc = results[0], results[1], results[2]
    if fn == 'exp_dc':
        try:
            return np.where(np.exp(-a * np.arange(len(data))) < 0.005)[0].min()
        except ValueError:
            return len(data) - 1
    elif fn == 'half_tri':
        return int(np.floor(a))
    else:
        return len(data) - 1


def pca(port, mean_prof=None, weights=None, quiet=False):
    """
    Compute the pricinpal components of port.

    Returns the eigvalues and eigenvectors sorted by the eigenvalues.  Note
        eigenvectors are column vectors.

    port is an nchan x nbin array of data values; in the PCA, these dimensions
        are interpreted as nmeasurements of nvariables, respectively.
    mean_prof is an nbin array of the mean profile to be subtracted; if None,
        a weighted average is calculated using weights.
    weights are the nchan weights used in the mean profile calculation and are
        also passed to np.cov as 'aweights' for the construction of the
        covariance matrix.  Default is equal weights.
    quiet=True suppresses output.

    Written mostly by EF.
    """

    nmes, ndim = port.shape  # nchan x nbin

    if not quiet: print(
        "Performing principal component analysis on data with %d dimensions and %d measurements..." % (ndim, nmes))

    if weights is None: weights = np.ones(len(port))

    # Subtract weighted average from each set of measurements
    if mean_prof is None:
        mean_prof = old_div((port.T * weights).T.sum(axis=0), weights.sum())
    delta_port = port - mean_prof

    # Compute unbiased weighted covariance matrix
    cov = np.cov(delta_port.T, aweights=weights, ddof=1)

    # Compute eigenvalues/vectors of cov, and order them
    eigval, eigvec = np.linalg.eigh(cov)
    isort = (np.argsort(eigval))[::-1]
    eigval, eigvec = eigval[isort], eigvec[:, isort]
    return eigval, eigvec


def reconstruct_portrait(port, mean_prof, eigvec):
    """
    Reconstruct a portrait from a mean profile and set of basis eigenvectors.

    Returns the reconstructed portrait.

    port is an nchan x nbin array of data values.
    mean_prof is an nbin array of the mean profile.
    eigvec is the nbin x ncomp array of basis column eigenvectors that will be
        projected onto.

    See pca(...) for details.

    """
    # Reconstruct port projected into space of eigvec
    delta_port = port - mean_prof
    reconst_port = np.dot(np.dot(delta_port, eigvec), eigvec.T) + mean_prof
    return reconst_port


def find_significant_eigvec(eigvec, check_max=10, return_max=10,
                            snr_cutoff=150.0, check_crossings=True, check_acorr=True,
                            return_smooth=True, **kwargs):
    """
    Determine which eigenvectors are "significant" based on smoothing and S/N.

    Returns the indices of significant eigenvectors (and the smoothed eigvec,
        if return_smooth is True).

    eigvec is the nbin x ncomp array of eigenvectors to be examined.
    check_max is the maximum number of consecutive eigenvectors to check for
        significance.
    return_max is the maximum number of eigenvectors to return as significant;
        note that these may not be the return_max most significant.
    snr_cutoff is the S/N ratio value above or equal to which an eigenvector is
        deemed "significant".
    check_crossings=True adds another check to weed out noisy/RFI eigenvectors
        that still pass based on snr_cutoff.
    check_acorr=True adds another check...
    return_smooth=True will return the array of smoothed eigenvectors so that
        smart_smooth(...) need not be run separately.
    **kwargs get passed to smart_smooth(...).

    NB: an alternate scheme could involve looking at the projection of the
        profiles onto the eigenvectors, and its e.g. autocorrelation.
    """
    if return_smooth: smooth_eigvec = np.zeros(eigvec.shape)
    ieig = []
    neig = 0
    for ivec in range(max(check_max, return_max)):
        add_eigvec = False
        ev = smart_smooth(eigvec.T[ivec], **kwargs)
        # Get Fourier-domain noise
        ev_noise = get_noise(eigvec.T[ivec]) * np.sqrt(len(ev) / 2.0)
        # Get Fourier-domain signal, and calculate S/N
        ev_snr = old_div(np.sum(np.abs(np.fft.rfft(ev)[1:]) ** 2), ev_noise)
        if ev_snr >= snr_cutoff:
            # Only check if close
            if check_crossings and ev_snr < 3 * snr_cutoff:
                ncross = count_crossings(abs(ev), 0.1 * abs(ev).max())
                if ncross < int(0.02 * len(ev)):
                    add_eigvec = True
            # Only check if close
            elif check_acorr and ev_snr < 3 * snr_cutoff and add_eigvec:
                acorr = np.correlate(ev, ev, 'same')
                fwhm = acorr.argmax() - \
                       np.where(acorr > acorr.max() / 2.0)[0].min()
                if fwhm > 5:
                    add_eigvec = True
                else:
                    add_eigvec = False
                    print("Borderline case eigenvector %d failed test." % ivec)
            else:
                add_eigvec = True
        if add_eigvec:
            ieig.append(ivec)
            neig += 1
            if return_smooth: smooth_eigvec[:, ivec] = ev
        if ivec + 1 == check_max: break
        if neig == return_max: break
    ieig = np.array(ieig)
    if return_smooth:
        return ieig, smooth_eigvec
    else:
        return ieig


def wavelet_smooth(port, wavelet='db8', nlevel=5, threshtype='hard', fact=1.0):
    """
    Compute the wavelet-denoised version of a portrait or profile.

    Returns the smoothed portrait or profile.

    port is a nchan x nbin array, or a single profile array of length nbin.
    wavelet is the name of the mother wavelet or pywt.Wavelet object; see
        PyWavelets for more [default=Daubechies 8].
    nlevel is the integer number of decomposition levels (5-6 typical).
    threshtype is the type of wavelet thresholding ('hard' or 'soft').
    fact is a fudge factor that scales the threshold value.

    Written mostly by EF.
    """
    if 'pywt' not in sys.modules:
        raise ImportError("You failed to import pywt and need PyWavelets to use wavelet_smooth!")
    try:
        nchan, nbin = port.shape
        one_prof = False
    except:
        port = np.array([port])
        nchan, nbin = port.shape
        one_prof = True

    smooth_port = np.zeros(port.shape)

    # Smooth each channel
    for ichan in range(nchan):
        prof = port[ichan]
        # Translation-invariant (stationary) wavelet transform/denoising
        coeffs = np.array(pw.swt(prof, wavelet, level=nlevel, start_level=0,
                                 axis=-1))
        # Get threshold value
        lopt = fact * (np.median(np.abs(coeffs[0])) / 0.6745) * np.sqrt(2 * \
                                                                        np.log(nbin))
        # Do wavelet thresholding
        coeffs = pw.threshold(coeffs, lopt, mode=threshtype, substitute=0.0)
        # Reconstruct data
        smooth_port[ichan] = pw.iswt(list(map(tuple, coeffs)), wavelet)

    # Return smoothed portrait
    if one_prof:
        return smooth_port[0]
    else:
        return smooth_port


def smart_smooth(port, try_nlevels=None, rchi2_tol=0.1, **kwargs):
    """
    Attempts to use wavelet_smooth(...) in a smart/iterative but automated way.

    For each profile in port, a "best-fit" smooth profile is found by
        maximizing the signal-to-noise ratio while keeping the reduced
        chi-squared value between the profile and the smoothed profile to
        within rchi2_tol of 1.0.  The optimization takes place over nlevel and
        the coefficient thresholding factor fact within wavelet_smooth(...).

    port is a nchan x nbin array, or a single profile array of length nbin.
    try_nlevels is the number of levels to minimize over in
        wavelet_smooth(...).  A value of 0 returns the port as is.  nlevel
        cannot be higher than log2(nbin), which is the default value for
        try_nlevels.  If nbin is odd, try_nlevels = 0, if nbin is not a power
        of two, try_nlevels = 1.
    rchi2_tol is the tolerance parameter that will allow greater deviations in
        the smooth profile from the input profile shape.
    **kwargs are passed to wavelet_smooth(...)
    """
    if try_nlevels == 0: return port
    try:
        nchan, nbin = port.shape
        one_prof = False
    except:
        port = np.array([port])
        nchan, nbin = port.shape
        one_prof = True
    if nbin % 2 != 0:
        return port
    elif np.modf(np.log2(nbin))[0] != 0.0:
        try_nlevels = 1
    elif try_nlevels is None:
        try_nlevels = int(np.log2(port.shape[-1]))
    smooth_port = np.zeros(port.shape)
    if 'wavelet' in kwargs:
        wavelet = kwargs['wave']
    else:
        wavelet = 'db8'
    if 'nlevel' in kwargs: kwargs.pop('nlevel')
    if 'threshtype' in kwargs:
        threshtype = kwargs['threshtype']
    else:
        threshtype = 'hard'
    if 'fact' in kwargs: kwargs.pop('fact')
    for iprof, prof in enumerate(port):
        if not np.any(prof): continue
        fun_vals = np.zeros([try_nlevels])
        fact_mins = np.zeros([try_nlevels])
        for ilevel in range(try_nlevels):
            options = {'maxiter': 1000, 'disp': False}  # , xatol:1e-8}
            other_args = (prof, wavelet, ilevel + 1, threshtype, rchi2_tol)
            # results = opt.minimize_scalar(fit_wavelet_smooth_function,
            #        bounds=[0.0,3.0], args=other_args, method='bounded',
            #        options=options)
            # fact_mins[ilevel] = results.x
            # fun_vals[ilevel] = results.fun
            results = opt.brute(fit_wavelet_smooth_function,
                                ranges=[tuple((0.0, 3.0))], args=other_args, Ns=30,
                                full_output=True)
            fact_mins[ilevel] = results[0][0]
            fun_vals[ilevel] = results[1]
        ilevel_min = fun_vals.argmin()
        fact_min = fact_mins[ilevel_min]
        smooth_port[iprof] = wavelet_smooth(prof, wavelet=wavelet,
                                            nlevel=ilevel_min + 1, threshtype=threshtype, fact=fact_min)
        red_chi2 = get_red_chi2(prof, smooth_port[iprof])
        if abs(red_chi2 - 1.0) > rchi2_tol: smooth_port[iprof] *= 0.0
    if one_prof:
        return smooth_port[0]
    else:
        return smooth_port


def fit_wavelet_smooth_function(fact, prof, wavelet, nlevel, threshtype,
                                rchi2_tol):
    """
    Calculate a S/N value for smart_smooth(...).

    Returns the a pseudo-S/N estimate of a smoothed profile.

    See smart_smooth(...) and wavelet_smooth(...) for arguments.
    """
    smooth_prof = wavelet_smooth(prof, wavelet=wavelet, nlevel=nlevel,
                                 threshtype=threshtype, fact=fact)
    smooth_prof_signal = np.sum(np.abs(np.fft.rfft(smooth_prof)[1:]) ** 2)
    if smooth_prof_signal:
        smooth_prof_noise = get_noise(smooth_prof) * \
                            np.sqrt(len(smooth_prof) / 2.0)
        if smooth_prof_noise:
            smooth_prof_snr = old_div(smooth_prof_signal, \
                                      smooth_prof_noise)
        else:
            smooth_prof_snr = np.inf
    else:
        smooth_prof_snr = 0.0
    red_chi2 = get_red_chi2(prof, smooth_prof)
    if abs(red_chi2 - 1.0) > rchi2_tol: smooth_prof_snr = 0.0
    return -smooth_prof_snr


def fit_powlaw(data, init_params, errs, freqs, nu_ref):
    """
    Fit a power-law function to data.

    lmfit is used for the minimization.
    Returns an object containing the fitted parameter values, the parameter
    errors, an array of the residuals, the chi-squared value, and the number of
    degrees of freedom.

    data is the input array of data values used in the fit.
    init_params is a list of initial parameter guesses = [amplitude at nu_ref,
        spectral index].
    errs is the array of uncertainties on the data values.
    freqs is an nchan array of frequencies.
    nu_ref is the frequency at which the amplitude is referenced.
    """
    if 'lmfit' not in sys.modules:
        raise ImportError("You failed to import lmfit and need it to use fit_powlaw!")
    # Generate the parameter structure
    params = lm.Parameters()
    params.add('amp', init_params[0], vary=True, min=None, max=None)
    params.add('alpha', init_params[1], vary=True, min=None, max=None)
    other_args = {'freqs': freqs, 'nu_ref': nu_ref, 'data': data, 'errs': errs}
    # Now fit it
    results = lm.minimize(fit_powlaw_function, params, kws=other_args)
    # fitted_params = np.array([param.value for param in
    #    results.params.itervalues()])
    dof = results.nfree
    chi2 = results.chisqr
    red_chi2 = results.redchi
    # The lmfit residuals are scaled by errs (in my fit function).
    residuals = results.residual * errs
    # fit_errs = np.array([param.stderr for param in
    #    results.params.itervalues()])
    results = DataBunch(alpha=results.params['alpha'].value,
                        alpha_err=results.params['alpha'].stderr,
                        amp=results.params['amp'].value,
                        amp_err=results.params['amp'].stderr, residuals=residuals,
                        nu_ref=nu_ref, chi2=chi2, dof=dof)
    return results


def fit_DM_to_freq_resids(freqs, frequency_residuals, errs):
    """
    Fit for a DM and reference frequency from frequency residuals.

    freqs is the nchan arrray of frequencies [MHz]
    frequency_residuals is the nchan array of residuals [s].
    errs is the array of uncertainties on the frequency residuals [s].

    Returned parameters are DM, offset, nu_ref for:
       res = Dconst*DM*(freqs**-2) + offset
       res = Dconst*DM*(freqs**-2 - nu_ref**-2)
    Returned covariance is of the linear coefficients a,b: ax + b.
    """
    x = freqs ** -2
    y = frequency_residuals
    w = errs ** -2
    p, V = np.polyfit(x=x, y=y, deg=1, w=w, cov=True)
    a, b = p[0], p[1]
    DM = old_div(a, Dconst)
    offset = b
    nu_ref = (old_div(-b, a)) ** -0.5
    a_err = (np.diag(V)[0]) ** 0.5
    b_err = (np.diag(V)[1]) ** 0.5
    cov = V.ravel()[1]
    DM_err = old_div(a_err, Dconst)
    offset_err = b_err
    nu_ref_err = (((nu_ref ** 2) / 4.0) * \
                  (((old_div(a_err, a)) ** 2) + ((old_div(b_err, b)) ** 2) - (old_div(2 * cov, (a * b))))) ** 0.5
    residuals = frequency_residuals - (a * (freqs ** -2) + b)
    chi2 = ((old_div(residuals, errs)) ** 2).sum()
    dof = len(frequency_residuals) - 2
    red_chi2 = old_div(chi2, dof)
    results = DataBunch(DM=DM, DM_err=DM_err, offset=offset,
                        offset_err=offset_err, nu_ref=nu_ref, nu_ref_err=nu_ref_err,
                        ab_cov=cov, residuals=residuals, chi2=chi2, dof=dof,
                        red_chi2=red_chi2)
    return results


def fit_gaussian_profile(data, init_params, errs, fit_flags=None,
                         fit_scattering=False, quiet=True):
    """
    Fit Gaussian functions to a profile.

    lmfit is used for the minimization.
    Returns an object containing an array of fitted parameter values, an array
        of parameter errors, an array of the residuals, the chi-squared value,
        and the number of degrees of freedom.

    data is the pulse profile array of length nbin used in the fit.
    init_params is a list of initial guesses for the 2 + (ngauss*3) values;
        the first value is the DC component, the second value is the
        scattering timescale [bin] and each remaining group of three represents
        the Gaussians' loc (0-1), wid (i.e. FWHM) (0-1), and amplitude (>0.0).
    errs is the array of uncertainties on the data values.
    fit_flags is an array specifying which of the non-scattering parameters to
        fit; defaults to fitting all.
    fit_scattering=True fits a scattering timescale parameter via convolution
        with a one-sided exponential function.
    quiet=True suppresses output.
    """
    if 'lmfit' not in sys.modules:
        raise ImportError("You failed to import lmfit and need it to use fit_gaussian_profile!")
    nparam = len(init_params)
    ngauss = old_div((len(init_params) - 2), 3)
    if fit_flags is None:
        fit_flags = [True for t in range(nparam)]
        fit_flags[1] = fit_scattering
    else:
        fit_flags = [np.bool(fit_flags[0]), fit_scattering] + \
                    [np.bool(fit_flags[iflag]) for iflag in range(1, nparam - 1)]
    # Generate the parameter structure
    params = lm.Parameters()
    for ii in range(nparam):
        if ii == 0:
            params.add('dc', init_params[ii], vary=fit_flags[ii], min=None,
                       max=None, expr=None)
        elif ii == 1:
            params.add('tau', init_params[ii], vary=fit_flags[ii], min=0.0,
                       max=None, expr=None)
        elif ii in range(nparam)[2::3]:
            params.add('loc%s' % str(old_div((ii - 2), 3) + 1), init_params[ii],
                       vary=fit_flags[ii], min=None, max=None, expr=None)
        elif ii in range(nparam)[3::3]:
            params.add('wid%s' % str(old_div((ii - 3), 3) + 1), init_params[ii],
                       vary=fit_flags[ii], min=0.0, max=wid_max, expr=None)
        elif ii in range(nparam)[4::3]:
            params.add('amp%s' % str(old_div((ii - 4), 3) + 1), init_params[ii],
                       vary=fit_flags[ii], min=0.0, max=None, expr=None)
        else:
            print("Undefined index %d." % ii)
            return DataBunch()
    other_args = {'data': data, 'errs': errs}
    # Now fit it
    results = lm.minimize(fit_gaussian_profile_function, params,
                          kws=other_args)
    fitted_params = np.array([param.value for param in
                              list(results.params.values())])
    fit_errs = np.array([param.stderr for param in
                         list(results.params.values())])
    dof = results.nfree
    chi2 = results.chisqr
    red_chi2 = results.redchi
    # The lmfit residuals are scaled by errs (in my fit function).
    residuals = results.residual * errs
    # residuals = data - gen_gaussian_profile(fitted_params, len(data))
    if not quiet:
        print("---------------------------------------------------------------")
        print("Multi-Gaussian Profile Fit Results")
        print("---------------------------------------------------------------")
        print("lmfit status:", results.message)
        print("Gaussians:", ngauss)
        print("DoF:", dof)
        print("reduced chi-sq: %.2f" % red_chi2)
        print("residuals mean: %.3g" % np.mean(residuals))
        print("residuals std.: %.3g" % np.std(residuals))
        print("---------------------------------------------------------------")
    results = DataBunch(fitted_params=fitted_params, fit_errs=fit_errs,
                        residuals=residuals, chi2=chi2, dof=dof)
    return results


def fit_gaussian_portrait(model_code, data, init_params, scattering_index,
                          errs, fit_flags, fit_scattering_index, phases, freqs, nu_ref,
                          join_params=[], P=None, quiet=True):
    """
    Fit evolving Gaussian components to a portrait.

    lmfit is used for the minimization.
    Returns an object containing an array of fitted parameter values, an array
        of parameter errors, the chi-squared value, and the number of degrees
        of freedom.

    model_code is a three digit string specifying the evolutionary functions
        to be used for the three Gaussian parameters (loc,wid,amp); see
        pplib.py header for details.
    data is the nchan x nbin phase-frequency data portrait used in the fit.
    init_params is a list of initial guesses for the 1 + (ngauss*6)
        parameters in the model; the first value is the DC component.  Each
        remaining group of six represent the Gaussians loc (0-1), linear slope
        in loc, wid (i.e. FWHM) (0-1), linear slope in wid, amplitude (>0,0),
        and spectral index alpha (no implicit negative).
    scattering_index is the scattering index for the model.
    errs is the array of uncertainties on the data values.
    fit_flags is an array of 1 + (ngauss*6) values, where non-zero entries
        signify that the parameter should be fit.
    fit_scattering_index will also fit for the power-law index of the
        scattering law, with the initial guess as scattering_index.
    phases is the array of phase values.
    freqs in the array of frequencies at which to calculate the model.
    nu_ref [MHz] is the frequency to which the locs, wids, and amps reference.
    join_params specifies how to simultaneously fit several portraits; see
        ppgauss.
    P is the pulse period [sec].
    quiet=True suppresses output.
    """
    if 'lmfit' not in sys.modules:
        raise ImportError("You failed to import lmfit and need it to use fit_gaussian_portrait!")
    nparam = len(init_params)
    ngauss = old_div((len(init_params) - 2), 6)
    # Generate the parameter structure
    params = lm.Parameters()
    for ii in range(nparam):
        if ii == 0:  # DC, not limited
            params.add('dc', init_params[ii], vary=bool(fit_flags[ii]),
                       min=None, max=None, expr=None)
        elif ii == 1:  # tau, limited by 0
            params.add('tau', init_params[ii], vary=bool(fit_flags[ii]),
                       min=0.0, max=None, expr=None)
        elif ii % 6 == 2:  # loc limits
            params.add('loc%s' % str(old_div((ii - 2), 6) + 1), init_params[ii],
                       vary=bool(fit_flags[ii]), min=None, max=None, expr=None)
        elif ii % 6 == 3:  # loc slope limits
            params.add('m_loc%s' % str(old_div((ii - 3), 6) + 1), init_params[ii],
                       vary=bool(fit_flags[ii]), min=None, max=None, expr=None)
        elif ii % 6 == 4:  # wid limits, limited by 0
            params.add('wid%s' % str(old_div((ii - 4), 6) + 1), init_params[ii],
                       vary=bool(fit_flags[ii]), min=0.0, max=wid_max, expr=None)
        elif ii % 6 == 5:  # wid slope limits
            params.add('m_wid%s' % str(old_div((ii - 5), 6) + 1), init_params[ii],
                       vary=bool(fit_flags[ii]), min=None, max=None, expr=None)
        elif ii % 6 == 0:  # amp limits, limited by 0
            params.add('amp%s' % str(old_div((ii - 6), 6) + 1), init_params[ii],
                       vary=bool(fit_flags[ii]), min=0.0, max=None, expr=None)
        elif ii % 6 == 1:  # amp index limits
            params.add('alpha%s' % str(old_div((ii - 7), 6) + 1), init_params[ii],
                       vary=bool(fit_flags[ii]), min=None, max=None, expr=None)
        else:
            print("Undefined index %d." % ii)
            return DataBunch()
    if len(join_params):
        join_ichans = join_params[0]
        njoin = len(join_ichans)
        for ii in range(njoin):
            params.add('phase%s' % str(ii + 1), join_params[1][0::2][ii],
                       vary=bool(join_params[2][0::2][ii]), min=None, max=None,
                       expr=None)
            params.add('DM%s' % str(ii + 1), join_params[1][1::2][ii],
                       vary=bool(join_params[2][1::2][ii]), min=None, max=None,
                       expr=None)
            # Comment out above DM param line and uncomment below to fix
            # join DM params to be the same!
            # if ii == 0:
            #    params.add('DM%s'%str(ii+1), join_params[1][1::2][ii],
            #            vary=bool(join_params[2][1::2][ii]), min=None,
            #            max=None, expr=None)
            # else:
            #    params.add('DM%s'%str(ii+1), join_params[1][1::2][ii],
            #            vary=bool(join_params[2][1::2][ii]), min=None,
            #            max=None, expr='DM1')

    else:
        join_ichans = []
    other_args = {'model_code': model_code, 'data': data, 'errs': errs,
                  'phases': phases, 'freqs': freqs, 'nu_ref': nu_ref,
                  'join_ichans': join_ichans, 'P': P}
    # Fit scattering index?  Not recommended.
    params.add('scattering_index', scattering_index, vary=fit_scattering_index,
               min=None, max=None, expr=None)
    # Now fit it
    results = lm.minimize(fit_gaussian_portrait_function, params,
                          kws=other_args)
    fitted_params = np.array([param.value for param in
                              list(results.params.values())])
    scattering_index = fitted_params[-1]
    fitted_params = fitted_params[:-1]
    fit_errs = np.array([param.stderr for param in
                         list(results.params.values())])
    scattering_index_err = fit_errs[-1]
    fit_errs = fit_errs[:-1]
    dof = results.nfree
    chi2 = results.chisqr
    red_chi2 = results.redchi
    # The lmfit residuals are scaled by errs (in my fit function).
    residuals = results.residual.reshape(errs.shape) * errs
    if not quiet:
        print("---------------------------------------------------------------")
        print("Gaussian Portrait Fit")
        print("---------------------------------------------------------------")
        print("lmfit status:", results.message)
        print("Gaussians:", ngauss)
        print("DoF:", dof)
        print("reduced chi-sq: %.2g" % red_chi2)
        print("residuals mean: %.3g" % np.mean(residuals))
        print("residuals std.: %.3g" % np.std(residuals))
        print("data std.: %.3g" % get_noise(data))
        print("---------------------------------------------------------------")
    results = DataBunch(lm_results=results, fitted_params=fitted_params,
                        fit_errs=fit_errs, scattering_index=scattering_index,
                        scattering_index_err=scattering_index_err, chi2=chi2, dof=dof)
    return results


def fit_phase_shift(data, model, noise=None, bounds=[-0.5, 0.5], Ns=100):
    """
    Fit a phase shift between data and model.

    This is a simple implementation of FFTFIT using a brute-force algorithm
    Returns an object containing the fitted parameter values (phase and scale),
        the parameter errors, and the reduced chi-squared value.

    The returned phase is the phase of the data with respect to the model.
    NB: the provided rotation functions rotate to earlier phases, given a
        positive phase.

    data is the array of data profile values.
    model is the array of model profile values.
    noise is time-domain noise-level; it is measured if None.
    bounds is the list containing the bounds on the phase.
    Ns is the number of grid points passed to opt.brute; *linear* slow-down!
    """
    dFFT = fft.rfft(data)
    dFFT[0] *= F0_fact
    mFFT = fft.rfft(model)
    mFFT[0] *= F0_fact
    if noise is None:
        # err = np.real(dFFT[-len(dFFT)/4:]).std()
        err = get_noise(data) * np.sqrt(len(data) / 2.0)
    else:
        err = noise * np.sqrt(len(data) / 2.0)
    d = old_div(np.real(np.sum(dFFT * np.conj(dFFT))), err ** 2.0)
    p = old_div(np.real(np.sum(mFFT * np.conj(mFFT))), err ** 2.0)
    other_args = (mFFT, dFFT, err)
    start = time.time()
    results = opt.brute(fit_phase_shift_function, [tuple(bounds)],
                        args=other_args, Ns=Ns, full_output=True)
    duration = time.time() - start
    phase = results[0][0]
    fmin = results[1]
    scale = old_div(-fmin, p)
    # In the next two error equations, consult fit_portrait for factors of 2
    phase_error = (scale * fit_phase_shift_function_2deriv(phase, mFFT, dFFT,
                                                           err)) ** -0.5
    scale_error = p ** -0.5
    red_chi2 = old_div((d - (old_div((fmin ** 2), p))), (len(data) - 2))
    # SNR of the fit, based on PDB's notes
    snr = pow(scale ** 2 * p, 0.5)
    return DataBunch(phase=phase, phase_err=phase_error, scale=scale,
            scale_err=scale_error, snr=snr, red_chi2=red_chi2,
            duration=duration)


def fit_portrait(data, model, init_params, P, freqs, nu_fit=None, nu_out=None,
                 errs=None, bounds=[(None, None), (None, None)], id=None, quiet=True):
    """
    Fit a phase offset and DM between a data portrait and model portrait.

    A truncated Newtonian algorithm is used.
    Returns an object containing the fitted parameter values, the parameter
        errors, and other attributes.

    data is the nchan x nbin phase-frequency data portrait.
    model is the nchan x nbin phase-frequency model portrait.
    init_params is a list of initial parameter guesses = [phase, DM], with
        phase in [rot] and DM in [cm**-3 pc].
    P is the period [s] of the pulsar at the data epoch.
    freqs is an nchan array of frequencies [MHz].
    nu_fit is the frequency [MHz] used as nu_ref in the fit.  Defaults to the
        mean value of freqs.
    nu_out is the desired output reference frequency [MHz].  Defaults to the
        zero-covariance frequency calculated in the fit.
    errs is the array of uncertainties on the data values (time-domain); they
        are measured if None.
    bounds is the list of 2 tuples containing the bounds on the phase and DM.
    id provides a label for the TOA.
    quiet = False produces more diagnostic output.
    """
    dFFT = fft.rfft(data, axis=1)
    dFFT[:, 0] *= F0_fact
    mFFT = fft.rfft(model, axis=1)
    mFFT[:, 0] *= F0_fact
    if errs is None:
        # errs = np.real(dFFT[:, -len(dFFT[0])/4:]).std(axis=1)
        errs = get_noise(data, chans=True) * np.sqrt(len(data[0]) / 2.0)
    else:
        errs = np.copy(errs) * np.sqrt(len(data[0]) / 2.0)
    d = np.real(np.sum(np.transpose(errs ** -2.0 * np.transpose(dFFT *
                                                                np.conj(dFFT)))))
    p_n = np.real(np.sum(mFFT * np.conj(mFFT), axis=1))
    if nu_fit is None: nu_fit = freqs.mean()
    # BEWARE BELOW! Order matters!
    other_args = (mFFT, p_n, dFFT, errs, P, freqs, nu_fit)
    minimize = opt.minimize
    # fmin_tnc seems to work best, fastest
    method = 'TNC'
    start = time.time()
    results = minimize(fit_portrait_function, init_params, args=other_args,
                       method=method, jac=fit_portrait_function_deriv, bounds=bounds,
                       options={'maxiter': 1000, 'disp': False, 'xtol': 1e-10})
    duration = time.time() - start
    phi = results.x[0]
    DM = results.x[1]
    nfeval = results.nfev
    return_code = results.status
    rcstring = RCSTRINGS["%s" % str(return_code)]
    # return code 4, LSFAIL, has not proved to give bad results
    # Someimes a failure code is returned because xtol is too stringent, or
    # the initial phase_guess is 'bad'
    # if not quiet and results.success is not True and results.status != 4:
    if not quiet and results.success is not True and \
            results.status not in [1, 2, 4]:
        if id is not None:
            ii = id[::-1].index("_")
            isub = id[-ii:]
            filename = id[:-ii - 1]
            sys.stderr.write(
                "Fit failed with return code %d: %s -- %s subint %s\n" % (
                    results.status, rcstring, filename, isub))
        else:
            sys.stderr.write(
                "Fit failed with return code %d -- %s" % (results.status,
                                                          rcstring))
    if not quiet and results.success is True and 0:  # For debugging
        sys.stderr.write("Fit succeeded with return code %d -- %s\n"
                         % (results.status, rcstring))
    # Curvature matrix = 1/2 2deriv of chi2 (cf. Gregory sect 11.5)
    # Parameter errors are related to curvature matrix by **-0.5
    # Calculate nu_zero
    nu_zero = fit_portrait_function_2deriv(np.array([phi, DM]), mFFT,
                                           p_n, dFFT, errs, P, freqs, nu_fit)[1]
    if nu_out is None:
        nu_out = nu_zero
    phi_out = phase_transform(phi, DM, nu_fit, nu_out, P, mod=True)
    # Calculate Hessian
    hessian = fit_portrait_function_2deriv(np.array([phi_out, DM]),
                                           mFFT, p_n, dFFT, errs, P, freqs, nu_out)[0]
    hessian = np.array([[hessian[0], hessian[2]], [hessian[2], hessian[1]]])
    covariance_matrix = np.linalg.inv(0.5 * hessian)
    covariance = covariance_matrix[0, 1]
    # These are true 1-sigma errors iff covariance == 0
    param_errs = list(covariance_matrix.diagonal() ** 0.5)
    dof = len(data.ravel()) - (len(freqs) + 2)
    chi2 = (d + results.fun)
    red_chi2 = old_div(chi2, dof)
    # Calculate scales
    scales = get_scales(data, model, phi, DM, P, freqs, nu_fit)
    # Errors on scales, if ever needed (these may be wrong b/c of covariances)
    scale_errs = pow(old_div(p_n, errs ** 2.0), -0.5)
    # SNR of the fit, based on PDB's notes
    snr = pow(np.sum(old_div(scales ** 2.0 * p_n, errs ** 2.0)), 0.5)
    results = DataBunch(phase=phi_out, phase_err=param_errs[0], DM=DM,
                        DM_err=param_errs[1], scales=scales, scale_errs=scale_errs,
                        nu_ref=nu_out, covariance=covariance, chi2=chi2, red_chi2=red_chi2,
                        snr=snr, duration=duration, nfeval=nfeval, return_code=return_code)
    return results


def get_noise(data, method=default_noise_method, **kwargs):
    """
    Estimate the off-pulse noise.

    data is a 1- or 2-D array of input values.
    method is either "PS" or "fit" where:
        "PS" uses the mean of the last quarter if the power spectrum, and
        "fit" attempts to find the noise floor by fitting a half-triangle
        function to the power spectrum.
    **kwargs are passed to the selected noise-measuring function.

    Other noise-measuring methods to come.
    """
    if method == "PS":
        return get_noise_PS(data, **kwargs)
    elif method == "fit":
        return get_noise_fit(data, **kwargs)
    else:
        print("Unknown get_noise method.")
        return 0


def get_noise_PS(data, frac=4, chans=False):
    """
    Estimate the off-pulse noise.

    This function will estimate the noise based on the mean of the highest
    1/frac harmonics in the power spectrum of data.

    data is a 1- or 2-D array of values.
    frac is a value specifying the inverse of the fraction of the
        power-spectrum to be examined.
    chans=True will return an estimate of the noise in each input channel.
    """
    if chans:
        noise = np.zeros(len(data))
        for ichan in range(len(noise)):
            prof = data[ichan]
            FFT = fft.rfft(prof)
            pows = old_div(np.real(FFT * np.conj(FFT)), len(prof))
            kc = int((1 - frac ** -1) * len(pows))
            noise[ichan] = np.sqrt(np.mean(pows[kc:]))
        return noise
    else:
        raveld = data.ravel()
        FFT = fft.rfft(raveld)
        pows = old_div(np.real(FFT * np.conj(FFT)), len(raveld))
        kc = int((1 - frac ** -1) * len(pows))
        return np.sqrt(np.mean(pows[kc:]))


def get_noise_fit(data, fact=1.1, chans=False):
    """
    Estimate the off-pulse noise.

    This function will estimate the noise based on the mean of the highest
        harmonics, where critical cutoff harmonic is found by a fit of a
        half-triangle function to the power-spectrum of the data.

    data is a 1- or 2-D array of values.
    fact is a value to scale the fitted cutoff harmonic.
    chans=True will return an estimate of the noise in each input channel.
    """
    if chans:
        noise = np.zeros(len(data))
        for ichan in range(len(noise)):
            prof = data[ichan]
            FFT = fft.rfft(prof)
            pows = old_div(np.real(FFT * np.conj(FFT)), len(prof))
            k_crit = fact * find_kc(pows)
            if k_crit >= len(pows):
                # Will only matter in unresolved or super narrow, high SNR cases
                k_crit = min(int(0.99 * len(pows)), k_crit)
            noise[ichan] = np.sqrt(np.mean(pows[int(k_crit):]))
        return noise
    else:
        raveld = data.ravel()
        FFT = fft.rfft(raveld)
        pows = old_div(np.real(FFT * np.conj(FFT)), len(raveld))
        k_crit = fact * find_kc(pows)
        if k_crit >= len(pows):
            # Will only matter in unresolved or super narrow, high SNR cases
            k_crit = min(int(0.99 * len(pows)), k_crit)
        return np.sqrt(np.mean(pows[int(k_crit):]))


def get_SNR(prof, fudge=3.25):
    """
    Return an estimate of the signal-to-noise ratio (SNR) of the data.

    Assumes that the baseline is removed!

    prof is an input array of data.
    fudge is a scale factor that attempts to match (poorly) PSRCHIVE's SNRs.

    Reference: Lorimer & Kramer (2005).
    """
    noise = get_noise(prof)
    nbin = len(prof)
    # dc = np.real(np.fft.rfft(data))[0]
    dc = 0
    Weq = old_div((prof - dc).sum(), (prof - dc).max())
    mask = np.where(Weq <= 0.0, 0.0, 1.0)
    Weq = np.where(Weq <= 0.0, 1.0, Weq)
    SNR = old_div((prof - dc).sum(), (noise * Weq ** 0.5))
    return old_div((SNR * mask), fudge)


def get_scales(data, model, phase, DM, P, freqs, nu_ref=np.inf):
    """
    Return the best-fit, per-channel scaling amplitudes.

    data is the nchan x nbin phase-frequency data portrait.
    model is the nchan x nbin phase-frequency model portrait.
    phase is the best-fit phase [rot]
    DM is the best-fit dispersion measure in [cm**-3 pc].
    P is the period [s] of the pulsar at the data epoch.
    freqs is an nchan array of frequencies [MHz].
    nu_ref is the reference frequency of the input phase [MHz].

    Reference: Equation 11 of Pennucci, Demorest, & Ransom (2014).
    """
    scales = np.zeros(len(freqs))
    dFFT = fft.rfft(data, axis=1)
    dFFT[:, 0] *= F0_fact
    mFFT = fft.rfft(model, axis=1)
    mFFT[:, 0] *= F0_fact
    p_n = np.real(np.sum(mFFT * np.conj(mFFT), axis=1))
    D = old_div(Dconst * DM, P)
    harmind = np.arange(len(mFFT[0]))
    phasor = np.exp(2.0j * np.pi * np.outer((phase + (D * (freqs ** -2.0 -
                                                           nu_ref ** -2.0))), harmind))
    scales = np.real(np.sum(dFFT * np.conj(mFFT) * phasor, axis=1))
    scales /= p_n
    return scales


def rotate_data(data, phase=0.0, DM=0.0, Ps=None, freqs=None, nu_ref=np.inf):
    """
    Rotate and/or dedisperse data.

    Positive values of phase and DM rotate the data to earlier phases (i.e., it
        "dedisperses") for freqs < nu_ref.

    Simpler functions are rotate_portrait or rotate_profile.

    data is a 1-, 2-, or 4-D array of data -- i.e. either an array of nbin (a
        profile), nchan x nbin (a portrait), or nsub x npol x nchan x nbin
        (a subint) values.
    phase is a value specifying the amount of achromatic rotation [rot].
    DM is a value specifying the amount of rotation based on the cold-plasma
        dispersion law [cm**-3 pc].
    Ps is a single float or an array of nsub periods [sec] needed if DM != 0.0.
    freqs is a single float or an array of either nchan or nsub x nchan
        frequencies [MHz].
    nu_ref is the reference frequency [MHz] that has zero dispersive delay.
    """
    shape = data.shape
    ndim = data.ndim
    if DM == 0.0:
        idim = 'ijkl'
        idim = idim[:ndim]
        iaxis = list(range(ndim))
        baxis = iaxis[-1]
        bdim = list(idim)[baxis]
        dFFT = fft.rfft(data, axis=baxis)
        nharm = dFFT.shape[baxis]
        harmind = np.arange(nharm)
        baxis = iaxis.pop(baxis)
        othershape = np.take(shape, iaxis)
        ones = np.ones(othershape)
        order = np.take(list(idim), iaxis)
        order = ''.join([order[iorder] for iorder in range(len(order))])
        phasor = np.exp(harmind * 2.0j * np.pi * phase)
        phasor = np.einsum(order + ',' + bdim, ones, phasor)
        dFFT *= phasor
        return fft.irfft(dFFT, axis=baxis)
    else:
        datacopy = np.copy(data)
        while (datacopy.ndim != 4):
            datacopy = np.array([datacopy])
        baxis = 3
        nsub = datacopy.shape[0]
        npol = datacopy.shape[1]
        nchan = datacopy.shape[2]
        dFFT = fft.rfft(datacopy, axis=baxis)
        nharm = dFFT.shape[baxis]
        harmind = np.arange(nharm)
        D = old_div(Dconst * DM, (np.ones(nsub) * Ps))
        if len(D) != nsub:
            print("Wrong shape for array of periods.")
            return 0
        try:
            test = float(nu_ref)
        except TypeError:
            print("Only one nu_ref permitted.")
            return 0
        if not hasattr(freqs, 'ndim'):
            freqs = np.ones(nchan) * freqs
        if freqs.ndim == 0:
            freqs = np.ones(nchan) * float(freqs)
        if freqs.ndim == 1:
            if nchan != len(freqs):
                print("Wrong number of frequencies.")
                return 0
            fterm = np.tile(freqs, nsub).reshape(nsub, nchan) ** -2.0 - \
                    nu_ref ** -2.0
        else:
            fterm = freqs ** -2.0 - nu_ref ** -2.0
        if fterm.shape[1] != nchan or fterm.shape[0] != nsub:
            print("Wrong shape for frequency array.")
            return 0
        phase += np.array([D[isub] * fterm[isub] for isub in range(nsub)])
        phase = np.einsum('ij,k', phase, harmind)
        phasor = np.exp(2.0j * np.pi * phase)
        dFFT = np.array([dFFT[:, ipol, :, :] * phasor for ipol in range(npol)])
        dFFT = np.einsum('jikl', dFFT)
        if ndim == 1:
            return fft.irfft(dFFT, axis=baxis)[0, 0, 0]
        elif ndim == 2:
            return fft.irfft(dFFT, axis=baxis)[0, 0]
        elif ndim == 4:
            return fft.irfft(dFFT, axis=baxis)
        else:
            print("Wrong number of dimensions.")
            return 0


def rotate_portrait(port, phase=0.0, DM=None, P=None, freqs=None,
                    nu_ref=np.inf):
    """
    Rotate and/or dedisperse a portrait.

    Positive values of phase and DM rotate the data to earlier phases
        (i.e. it "dedisperses") for freqs < nu_ref.

    When used to dediserpse, rotate_portrait is virtually identical to
        arch.dedisperse() in PSRCHIVE.

    port is a nchan x nbin array of data values.
    phase is a value specifying the amount of achromatic rotation [rot].
    DM is a value specifying the amount of rotation based on the cold-plasma
        dispersion law [cm**-3 pc].
    P is the pulsar period [sec] at the epoch of the data, needed if a DM is
        provided.
    freqs is an array of frequencies [MHz], needed if a DM is provided.
    nu_ref is the reference frequency [MHz] that has zero delay for any value
        of DM.
    """
    pFFT = fft.rfft(port, axis=1)
    for nn in range(len(pFFT)):
        if DM is None and freqs is None:
            pFFT[nn, :] *= np.exp(np.arange(len(pFFT[nn])) * 2.0j * np.pi *
                                  phase)
        else:
            D = old_div(Dconst * DM, P)
            freq = freqs[nn]
            phasor = np.exp(np.arange(len(pFFT[nn])) * 2.0j * np.pi * (phase +
                                                                       (D * (freq ** -2.0 - nu_ref ** -2.0))))
            pFFT[nn, :] *= phasor
    return fft.irfft(pFFT)


def normalize_portrait(port, method='rms', weights=None, return_norms=False):
    """
    Normalize each profile in a portrait.

    method is either "mean", "max", "prof", "rms", or "abs".
        if "mean", then normalize by the profile mean (flux).
        if "max", then normalize by the profile maximum.
        if "prof", then normalize by the mean profile.
        if "rms", then normalize by the noise level, such that
            get_noise(profile) = 1.
        if "abs", then normalize such that each profile would have the same
            'length' in an nbin vector space.
    weights=None assumes equal weights for all profiles in port when using
        method "prof"; otherwise specifies an array of nchan weights.
    return_norms=True returns an array of the normalization values.
    """
    if method not in ("mean", "max", "prof", "rms", "abs"):
        print("Unknown method for normalize_portrait(...), '%s'." % method)
    else:
        norm_port = np.zeros(port.shape)
        norm_vals = np.ones(len(port))
        if method == "prof":
            good_ichans = np.where(port.sum(axis=1) != 0.0)[0]
            if weights is None:
                weights = np.ones(len(good_ichans))
            else:
                weights = weights[good_ichans]
            mean_prof = np.average(port[good_ichans], axis=0, weights=weights)
        for ichan in range(len(port)):
            if port[ichan].any():
                if method == "mean":
                    norm = port[ichan].mean()
                elif method == "max":
                    norm = port[ichan].max()
                elif method == "prof":
                    norm = fit_phase_shift(port[ichan], mean_prof).scale
                elif method == "rms":
                    norm = get_noise(port[ichan])
                else:
                    norm = (pow(port[ichan], 2.0).sum()) ** 0.5
                norm_port[ichan] = old_div(port[ichan], norm)
                norm_vals[ichan] = norm
        if return_norms:
            return norm_port, norm_vals
        else:
            return norm_port


def add_DM_nu(port, phase=0.0, DM=None, P=None, freqs=None, xs=[-2.0],
              Cs=[1.0], nu_ref=np.inf):
    """
    Rotate a portrait to simulate a frequency-dependent DM.

    This function is identical to rotate_portrait, but allows for an arbitrary
        power-law frequency dependence in the DM rotation.  Note that the
        default behavior is (should be) identical to rotate_portrait and that
        "DM" is the overall constant in front of the frequency term.  Also note
        that nu_ref doubles as nu_DM.  The additional arguments for this
        function are:

    xs is an array of powers that determine the observed frequency dependence
        of the dispersion law.  The terms are added to the frequency term
        of the phasor in the form + C*(nu**x - nu_ref**x), where the
        coefficients are given by Cs.
    Cs coefficients for the above.  They will always be assumed to be 1.0,
        unless specified.
    """
    pFFT = fft.rfft(port, axis=1)
    for nn in range(len(pFFT)):
        if DM is None and freqs is None:
            pFFT[nn, :] *= np.exp(np.arange(len(pFFT[nn])) * 2.0j * np.pi *
                                  phase)
        else:
            D = old_div(Dconst * DM, P)
            freq = freqs[nn]
            freq_term = 0.0
            if not hasattr(Cs, "__iter__"):
                Cs = np.ones(len(xs))
            if len(Cs) < len(xs):
                Cs = list(Cs) + list(np.ones(len(xs) - len(Cs)))
            for C, x in zip(Cs, xs):
                freq_term += C * (freq ** x - nu_ref ** x)
            phasor = np.exp(np.arange(len(pFFT[nn])) * 2.0j * np.pi * (phase +
                                                                       (D * (freq_term))))
            pFFT[nn, :] *= phasor
    return fft.irfft(pFFT)


def rotate_profile(profile, phase=0.0):
    """
    Rotate a profile by phase.

    Positive values of phase rotate to earlier phase.

    profile is an input array of data.
    phase is a value specifying the amount of rotation [rot].
    """
    pFFT = fft.rfft(profile)
    pFFT *= np.exp(np.arange(len(pFFT)) * 2.0j * np.pi * phase)
    return fft.irfft(pFFT)


def fft_rotate(arr, bins):
    """
    Return array 'arr' rotated by 'bins' places to the left.

    The rotation is done in the Fourier domain using the Shift Theorem.
    'bins' can be fractional.
    The resulting vector will have the same length as the original.

    Taken and tweaked from SMR's PRESTO.  Used for testing.
    """
    arr = np.asarray(arr)
    freqs = np.arange(old_div(arr.size, 2) + 1, dtype=np.float64)
    phasor = np.exp(old_div(complex(0.0, 2 * np.pi) * freqs * bins,
                            np.float64(arr.size)))
    return np.fft.irfft(phasor * np.fft.rfft(arr), arr.size)


def DM_delay(DM, freq, freq_ref=np.inf, P=None):
    """
    Return the amount of dispersive delay [sec] between two frequencies.

    DM is the dispersion measure [cm**-3 pc].
    freq is the delayed frequency [MHz].
    freq_ref is the frequency [MHz] against which the delay is measured.
    P is a period [sec]; if provided, the return is in [rot].
    """
    delay = Dconst * DM * ((freq ** -2.0) - (freq_ref ** -2.0))
    if P:
        return old_div(delay, P)
    else:
        return delay


def phase_transform(phi, DM, nu_ref1=np.inf, nu_ref2=np.inf, P=None,
                    mod=False):
    """
    Transform an input delay at nu_ref1 to a delay at nu_ref2.

    phi is an input delay [rot] or [sec].
    DM is the dispersion measure [cm**-3 pc].
    nu_ref1 is the reference frequency [MHz] for phi.
    nu_ref2 is the reference frequency [MHz] of the ouput delay.
    P is the pulsar period; if not provided, assumes phi is in [sec].
    mod=True ensures the output delay in [rot] is on the interval [-0.5, 0.5).

    Default behavior is for P=1.0 [sec], i.e. transform delays [sec]
    """
    if P is None:
        P = 1.0
        mod = False
    phi_prime = phi + (Dconst * DM * P ** -1 * (nu_ref2 ** -2.0 - nu_ref1 ** -2.0))
    if mod:
        # phi_prime %= 1
        phi_prime = np.where(abs(phi_prime) >= 0.5, phi_prime % 1, phi_prime)
        phi_prime = np.where(phi_prime >= 0.5, phi_prime - 1.0, phi_prime)
        if not phi_prime.shape:
            phi_prime = np.float64(phi_prime)
    return phi_prime


def guess_fit_freq(freqs, SNRs=None):
    """
    Estimate a zero-covariance frequency.

    Returns a "center of mass" frequency, where the weights are given by
        SNR*(freq**-2).

    freqs is an array of frequencies.
    SNRs is an array of signal-to-noise ratios (defaults to 1s).
    """
    nu0 = (freqs.min() + freqs.max()) * 0.5
    if SNRs is None:
        SNRs = np.ones(len(freqs))
    diff = old_div(np.sum((freqs - nu0) * SNRs * freqs ** -2), np.sum(SNRs * freqs ** -2))
    return nu0 + diff


def calculate_TOA(epoch, P, phi, DM=0.0, nu_ref1=np.inf, nu_ref2=np.inf):
    """
    Calculate a TOA [PSRCHIVE MJD] for given input.

    epoch is a PSRCHIVE MJD.
    P is the pulsar period [sec].
    phi is the phase offset [rot].
    DM is the dispersion measure [cm**-3 pc], for transforming to nu_ref2.
    nu_ref1 is the reference frequency [MHz] of phi.
    nu_ref2 is the reference frequency [MHz] of the output TOA.
    """
    # The pre-Doppler corrected DM must be used
    phi_prime = phase_transform(phi, DM, nu_ref1, nu_ref2, P, mod=False)
    TOA = epoch + pr.MJD(old_div((phi_prime * P), (3600 * 24.)))
    return TOA


def load_data(filename, state=None, dedisperse=False, dededisperse=False,
              tscrunch=False, pscrunch=False, fscrunch=False, rm_baseline=True,
              flux_prof=False, refresh_arch=True, return_arch=True, quiet=False):
    """
    Load data from a PSRCHIVE archive.

    Returns an object containing a large number of useful archive attributes.

    filename is the input PSRCHIVE archive.
    Most of the options should be self-evident; archives are manipulated by
        PSRCHIVE only.
    Setting state='Intensity' or pscrunch=True overrides any conflicting
        keyword that would result in npol=4.
    flux_prof=True will include an array with the phase-averaged flux profile.
    refresh_arch=True refreshes the returned archive to its original state.
    return_arch=False will not return the archive, which may be smart at times.
    quiet=True suppresses output.
    """
    # Load archive
    arch = pr.Archive_load(filename)
    source = arch.get_source()
    if not quiet:
        print("\nReading data from %s on source %s..." % (filename, source))
    # Basic info used in TOA output
    telescope = arch.get_telescope()
    try:
        telescope_code = telescope_code_dict[telescope.upper()][0]
    except KeyError:
        telescope_code = telescope
    frontend = arch.get_receiver_name()
    backend = arch.get_backend_name()
    backend_delay = arch.get_backend_delay()
    # Set state?
    if state is not None:
        if state != arch.get_state():
            arch.convert_state(state)
    # De/dedisperse?
    if dedisperse: arch.dedisperse()
    if dededisperse: arch.dededisperse()
    DM = arch.get_dispersion_measure()
    dmc = arch.get_dedispersed()
    # Maybe use better baseline subtraction??
    if rm_baseline: arch.remove_baseline()
    # tscrunch?
    if tscrunch: arch.tscrunch()
    nsub = arch.get_nsubint()
    # Integration length
    integration_length = arch.integration_length()
    # doppler_factors are the Doppler factors:
    #    doppler_factor = nu_source / nu_observed = sqrt( (1+beta) / (1-beta)),
    #    for beta = v/c, and v > 0 for /increasing/ distance (redshift).
    #    NB: It might be that PSRFITS/PSRCHIVE define v as positive for
    #        /decreasing/ distance (blueshift), but then the signs for beta
    #        above would be switched such that doppler_factor is still > 1 for
    #        redshift.
    doppler_factors = np.array([arch.get_Integration( \
        int(isub)).get_doppler_factor() for isub in range(nsub)])
    arch.execute('fix pointing')  # PSRCHIVE hack
    parallactic_angles = np.array([arch.get_Integration( \
        int(isub)).get_parallactic_angle() for isub in range(nsub)])
    # pscrunch?
    if pscrunch: arch.pscrunch()
    state = arch.get_state()
    npol = arch.get_npol()
    # fscrunch?
    if fscrunch: arch.fscrunch()
    # Nominal "center" of the band, but not necessarily
    nu0 = arch.get_centre_frequency()
    # For the negative BW cases.  Good fix
    # bw = abs(arch.get_bandwidth())
    bw = arch.get_bandwidth()
    nchan = arch.get_nchan()
    # Centers of frequency channels
    freqs = np.array([[sub.get_centre_frequency(ichan) for ichan in \
                       range(nchan)] for sub in arch])
    nbin = arch.get_nbin()
    # Centers of phase bins
    phases = get_bin_centers(nbin, lo=0.0, hi=1.0)
    # phases = np.linspace(0.0 + (nbin*2)**-1, 1.0 - (nbin*2)**-1, nbin)
    # These are NOT the bin centers...
    # phases = np.arange(nbin, dtype='d') / nbin
    # Get data
    # PSRCHIVE indices are [isub:ipol:ichan:ibin]
    subints = arch.get_data()
    Ps = np.array([sub.get_folding_period() for sub in arch], dtype=np.double)
    epochs = [sub.get_epoch() for sub in arch]
    subtimes = [sub.get_duration() for sub in arch]
    # Get weights
    weights = arch.get_weights()
    weights_norm = np.where(weights == 0.0, np.zeros(weights.shape),
                            np.ones(weights.shape))
    # Get off-pulse noise
    if not use_get_noise:
        noise_stds = np.array([sub.baseline_stats()[1] ** 0.5 for sub in arch])
    else:
        noise_stds = np.zeros([nsub, npol, nchan])
        for isub in range(nsub):
            for ipol in range(npol):
                noise_stds[isub, ipol] = get_noise(subints[isub, ipol],
                                                   chans=True)
    # Temporary hack -- needed for some data with non-zero near-constant chans
    # for isub in range(nsub):
    #    for ipol in range(npol):
    #        #for ibad_chan in np.where(noise_stds[isub,ipol] == 0.0)[0]:
    #        for ibad_chan in np.where(noise_stds[isub,ipol] < 1e-8)[0]:
    #            weights_norm[isub,ibad_chan] *= 0.0
    #            weights[isub,ibad_chan] *= 0.0
    ok_isubs = np.compress(weights_norm.mean(axis=1), list(range(nsub)))
    ok_ichans = [np.compress(weights_norm[isub], list(range(nchan))) \
                 for isub in range(nsub)]
    # np.einsum is AWESOME
    masks = np.einsum('ij,k', weights_norm, np.ones(nbin))
    masks = np.einsum('j,ikl', np.ones(npol), masks)
    SNRs = np.zeros([nsub, npol, nchan])
    for isub in range(nsub):
        for ipol in range(npol):
            for ichan in range(nchan):
                SNRs[isub, ipol, ichan] = \
                    arch.get_Integration(
                        isub).get_Profile(ipol, ichan).snr()
    # The rest is now ignoring npol...
    arch.pscrunch()
    if flux_prof:
        # Flux profile
        # The below is about equal to bscrunch to ~6 places
        arch.dedisperse()
        arch.tscrunch()
        flux_prof = arch.get_data().mean(axis=3)[0][0]
    else:
        flux_prof = np.array([])
    # Get pulse profile
    arch.tscrunch()
    arch.fscrunch()
    prof = arch.get_data()[0, 0, 0]
    prof_noise = arch.get_Integration(0).baseline_stats()[1][0, 0] ** 0.5
    prof_SNR = arch.get_Integration(0).get_Profile(0, 0).snr()
    # Number unzapped channels (mean), subints
    nchanx = np.array(list(map(len, ok_ichans))).mean()
    nsubx = len(ok_isubs)
    if not quiet:
        P = arch.get_Integration(0).get_folding_period() * 1000.0
        print("\tP [ms]             = %.3f\n\
        DM [cm**-3 pc]     = %.6f\n\
        center freq. [MHz] = %.4f\n\
        bandwidth [MHz]    = %.1f\n\
        # bins in prof     = %d\n\
        # channels         = %d\n\
        # chan (mean)      = %d\n\
        # subints          = %d\n\
        # unzapped subint  = %d\n\
        pol'n state        = %s\n" % (P, DM, nu0, bw, nbin, nchan, nchanx, nsub,
                                      nsubx, state))
    if refresh_arch: arch.refresh()
    if not return_arch: arch = None
    # Return getitem/attribute-accessible class!
    data = DataBunch(arch=arch, backend=backend, backend_delay=backend_delay,
                     bw=bw, doppler_factors=doppler_factors, DM=DM, dmc=dmc,
                     epochs=epochs, filename=filename, flux_prof=flux_prof, freqs=freqs,
                     frontend=frontend, integration_length=integration_length,
                     masks=masks, nbin=nbin, nchan=nchan, noise_stds=noise_stds,
                     npol=npol, nsub=nsub, nu0=nu0, ok_ichans=ok_ichans,
                     ok_isubs=ok_isubs, parallactic_angles=parallactic_angles,
                     phases=phases, prof=prof, prof_noise=prof_noise, prof_SNR=prof_SNR,
                     Ps=Ps, SNRs=SNRs, source=source, state=state, subints=subints,
                     subtimes=subtimes, telescope=telescope,
                     telescope_code=telescope_code, weights=weights)
    return data


def unpack_dict(data):
    """
    unpack a DataBunch/dictionary

    <UNDER CONSTRUCTION>

    This does not work yet; just for reference...
    Dictionary has to be named 'data'...
    """
    for key in list(data.keys()):
        exec(key + " = data['" + key + "']")


def write_model(filename, name, model_code, nu_ref, model_params, fit_flags,
                alpha, fit_alpha, append=False, quiet=False):
    """
    Write a Gaussian-component model.

    filename is the output file name.
    name is the name of the model.
    model_code is a three digit string specifying the evolutionary functions
        to be used for the three Gaussian parameters (loc,wid,amp); see
        pplib.py header for details.
    nu_ref is the reference frequency [MHz] of the model.
    model_params is the list of 2 + 6*ngauss model parameters, where index 1 is
        the scattering timescale [sec].
    fit_flags is the list of 2 + 6*ngauss flags (1 or 0) designating a fit.
    alpha is the scattering index
    fit_alpha is the fit flag for fitting the scattering index; not yet
        fully-implemented.
    append=True will append to a file named filename.
    quiet=True suppresses output.
    """
    if append:
        outfile = open(filename, "a")
    else:
        outfile = open(filename, "w")
    outfile.write("MODEL   %s\n" % name)
    outfile.write("CODE    %s\n" % model_code)
    outfile.write("FREQ    %.5f\n" % nu_ref)
    outfile.write("DC     % .8f %d\n" % (model_params[0], fit_flags[0]))
    outfile.write("TAU    % .8f %d\n" % (model_params[1], fit_flags[1]))
    outfile.write("ALPHA  % .3f      %d\n" % (alpha, fit_alpha))
    ngauss = old_div((len(model_params) - 2), 6)
    for igauss in range(ngauss):
        comp = model_params[(2 + igauss * 6):(8 + igauss * 6)]
        fit_comp = fit_flags[(2 + igauss * 6):(8 + igauss * 6)]
        line = (igauss + 1,) + tuple(np.array(list(zip(comp, fit_comp))).ravel())
        outfile.write("COMP%02d % .8f %d  % .8f %d  % .8f %d  % .8f %d  % .8f %d  % .8f %d\n" % line)
    outfile.close()
    if not quiet: print("%s written." % filename)


def read_model(modelfile, phases=None, freqs=None, P=None, quiet=False):
    """
    Read-in a Gaussian-component model.

    If only modelfile is specified, returns the contents of the modelfile:
        (model name, model reference frequency, number of Gaussian components,
        list of parameters, list of fit flags).
    Otherwise, builds a model based on the input phases, frequencies, and
        period (if scattering timescale != 0.0).

    modelfile is the name of the write_model(...)-type of model file.
    phases is an array of phase-bin centers [rot].
    freqs in an array of center-frequencies [MHz].
    P is the pulsar period [sec]; needed if the scattering timescale != 0.0.
    quiet=True suppresses output.
    """
    if phases is None and freqs is None:
        read_only = True
    else:
        read_only = False
    ngauss = 0
    comps = []
    if not quiet:
        print("Reading model from %s..." % modelfile)
    modeldata = open(modelfile, "r").readlines()
    for line in modeldata:
        info = line.split()
        try:
            if info[0] == "MODEL":
                modelname = info[1]
            elif info[0] == "CODE":
                model_code = info[1]
            elif info[0] == "FREQ":
                nu_ref = np.float64(info[1])
            elif info[0] == "DC":
                dc = np.float64(info[1])
                fit_dc = int(info[2])
            elif info[0] == "TAU":
                tau = np.float64(info[1])
                fit_tau = int(info[2])
            elif info[0] == "ALPHA":
                alpha = np.float64(info[1])
                fit_alpha = int(info[2])
            elif info[0][:4] == "COMP":
                comps.append(line)
                ngauss += 1
            else:
                pass
        except IndexError:
            pass
    params = np.zeros(ngauss * 6 + 2)
    fit_flags = np.zeros(len(params))
    params[0] = dc
    params[1] = tau
    fit_flags[0] = fit_dc
    fit_flags[1] = fit_tau
    for igauss in range(ngauss):
        comp = list(map(np.float64, comps[igauss].split()[1::2]))
        fit_comp = list(map(int, comps[igauss].split()[2::2]))
        params[2 + igauss * 6: 8 + (igauss * 6)] = comp
        fit_flags[2 + igauss * 6: 8 + (igauss * 6)] = fit_comp
    if not read_only:
        nbin = len(phases)
        nchan = len(freqs)
        if params[1] != 0:
            if P is None:
                print("Need period P for non-zero scattering value TAU.")
                return 0
            else:
                params[1] *= old_div(nbin, P)
        model = gen_gaussian_portrait(model_code, params, alpha, phases, freqs,
                                      nu_ref)
    if not quiet and not read_only:
        print("Model Name: %s" % modelname)
        print("Made %d component model with %d profile bins," % (ngauss, nbin))
        if len(freqs) != 1:
            bw = (freqs[-1] - freqs[0]) + ((freqs[-1] - freqs[-2]))
        else:
            bw = 0.0
        print("%d frequency channels, ~%.0f MHz bandwidth, centered near ~%.0f MHz," % (nchan, abs(bw), freqs.mean()))
        print("with model parameters referenced at %.3f MHz." % nu_ref)
    # This could be changed to a DataBunch
    if read_only:
        return (modelname, model_code, nu_ref, ngauss, params, fit_flags,
                alpha, fit_alpha)
    else:
        return (modelname, ngauss, model)


def read_spline_model(modelfile, freqs=None, nbin=None, quiet=False):
    """
    Read-in a model created by make_spline_model(...).

    If only modelfile is specified, returns the contents of the pickled model:
        (model name, source name, datafile name from which the model was
        created, mean profile vector used in the PCA, the eigenvectors, and the
        'tck' tuple containing knot locations, B-spline coefficients, and
        spline degree)
    Otherwise, builds a model based on the input frequencies using the function
        gen_spline_portrait(...).

    modelfile is the name of the make_spline_model(...)-type of model file.
    freqs in an array of frequencies at which to build the model; these should
        be in the same units as the datafile frequencies, and they should be
        within the same bandwidth range (cf. the knot vector).
    nbin is the number of phase bins to use in the model; by default it uses
        the number in modelfile.
    quiet=True suppresses output.
    """
    if freqs is None:
        read_only = True
    else:
        read_only = False
    if not quiet:
        print("Reading model from %s..." % modelfile)
    try:
        modelname, source, datafile, mean_prof, eigvec, tck = \
                pickle.load(open(modelfile, 'rb'))
    except UnicodeDecodeError:  # python2 to python3 pickling issues
        modelname, source, datafile, mean_prof, eigvec, tck = \
                pickle.load(open(modelfile, 'rb'), encoding='bytes')
    if read_only:
        return (modelname, source, datafile, mean_prof, eigvec, tck)
    else:
        return (modelname,
                gen_spline_portrait(mean_prof, freqs, eigvec, tck, nbin))


def get_spline_model_coords(modelfile, nfreq=1000, lo_freq=None, hi_freq=None,
                            write_pick=False):
    """
    Returns the spline model coordinates and the frequencies evaluated.

    modelfile is the name of the make_spline_model(...)-type of model file.
    nfreq is the number of frequencies at which to evaluate the spline.
    lo_freq=None uses the lowest frequency knot endpoint; otherwise in MHz.
    hi_freq=None uses the highest frequency knot endpoint; otherwise in MHz.
    write_pick=True writes a pickle file containing the frequencies and spline
        model coordinates.
    """
    modelname, source, datafile, mean_prof, eigvec, tck = \
        read_spline_model(modelfile, quiet=True)
    if lo_freq is None: lo_freq = tck[0].min()
    if hi_freq is None: hi_freq = tck[0].max()
    model_freqs = np.linspace(lo_freq, hi_freq, nfreq)
    proj_port = np.array(si.splev(model_freqs, tck, der=0, ext=0)).T
    if write_pick:
        outfile = modelfile + "_coords.pick"
        print("Unloading %s..." % outfile)
        of = open(outfile, 'wb')
        pickle.dump([model_freqs, proj_port], of, protocol=2)
        of.close()
    return model_freqs, proj_port


def file_is_type(filename, filetype="ASCII"):
    """
    Checks if a file is a certain type.

    filename is the name of file to be checked by parsing the output from a
        call to the command 'file -L <filename>'.
    filetype is the string that is searched for in the output.
    """
    cmd = "file -L %s" % filename
    o = subprocess.Popen(cmd, shell=isinstance(cmd, str),
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, close_fds=True)
    line = o.stdout.readline().decode().split()
    try:
        line.index(filetype)
        return True
    except ValueError:
        return False


def unload_new_archive(data, arch, outfile, DM=None, dmc=0, weights=None,
                       quiet=False):
    """
    Unload a PSRFITS archive containing new data values.

    PSRCHIVE unloads frequencies as floats with three digits of precision.

    data is the nsub x npol x nchan x nbin array of amplitudes to be stored,
        which has the same shape as arch.get_data().shape.
    arch is the PSRCHIVE archive instance to be otherwise copied.
    outfile is the name of the new written archive.
    DM is the DM value [cm**-3 pc] to be stored in the archive; if None,
        nothing is changed.
    dmc=0 means the archive is stored dededispersed (not "DM corrected"); the
        data provided should be in the same state as dmc implies.
    weights is an nsub x nchan array of channel weights; if None, nothing is
        changed.
    quiet=True suppresses output.
    """
    if dmc:
        if arch.get_dedispersed():
            pass
        else:
            arch.dedisperse()
    else:
        if arch.get_dedispersed():
            arch.dededisperse()
        else:
            pass
    if DM is not None: arch.set_dispersion_measure(DM)
    nsub, npol, nchan, nbin = arch.get_data().shape
    for isub in range(nsub):
        sub = arch.get_Integration(isub)
        for ipol in range(npol):
            for ichan in range(nchan):
                prof = sub.get_Profile(ipol, ichan)
                prof.get_amps()[:] = data[isub, ipol, ichan]
                if weights is not None:
                    sub.set_weight(ichan, float(weights[isub, ichan]))
    arch.unload(outfile)
    if not quiet: print("\nUnloaded %s.\n" % outfile)


def write_archive(data, ephemeris, freqs, nu0=None, bw=None,
                  outfile="pparchive.fits", tsub=1.0, start_MJD=None, weights=None,
                  dedispersed=False, state="Stokes", telescope="GBT", quiet=False):
    """
    Write data to a PSRCHIVE psrfits file (using a hack).

    Not guaranteed to work perfectly.  See also unload_new_archive(...).

    Takes dedispersed data, please.

    data is a nsub x npol x nchan x nbin array of values.
    ephemeris is the timing ephemeris to be installed.
    freqs is an array of the channel center-frequencies [MHz].
    nu0 is the center frequency [MHz]; if None, defaults to the mean of freqs.
    bw is the bandwidth; if None, is calculated from freqs.
    outfile is the desired output file name.
    tsub is the duration of each subintegration [sec].
    start_MJD is the starting epoch of the data in PSRCHIVE MJD format,
        e.g. pr.MJD(56700.123456789012345).
    weights is a nsub x nchan array of weights.
    dedispersed=True, will save the archive as dedispered.
    state is the polarization state of the data ("Coherence" or "Stokes" for
        npol == 4, or "Intensity" for npol == 1).
    telescope is the telescope name.
    quiet=True suppresses output.

    Mostly written by PBD.
    """
    nsub, npol, nchan, nbin = data.shape
    if nu0 is None:
        # This is off by a tiny bit...
        nu0 = freqs.mean()
    if bw is None:
        # This is off by a tiny bit...
        bw = (freqs.max() - freqs.min()) + abs(freqs[1] - freqs[0])
    # Phase bin centers
    phases = get_bin_centers(nbin, lo=0.0, hi=1.0)
    # phases = np.linspace(0.0 + (nbin*2)**-1, 1.0 - (nbin*2)**-1, nbin)
    # Create the Archive instance.
    # This is kind of a weird hack, if we create a PSRFITS
    # Archive directly, some header info does not get filled
    # in.  If we instead create an ASP archive, it will automatically
    # be (correctly) converted to PSRFITS when it is unloaded...
    arch = pr.Archive_new_Archive("ASP")
    arch.resize(nsub, npol, nchan, nbin)
    try:
        import parfile
        par = parfile.psr_par(ephemeris)
        try:
            PSR = par.PSR
        except AttributeError:
            PSR = par.PSRJ
        DECJ = par.DECJ
        RAJ = par.RAJ
        DM = par.DM
    except ImportError:
        parfile = open(ephemeris, "r").readlines()
        for iline in range(len(parfile)):
            param = parfile[iline].split()
            if len(param) == 0:
                pass
            elif param[0] == ("PSR" or "PSRJ"):
                PSR = param[1]
            elif param[0] == "RAJ":
                RAJ = param[1]
            elif param[0] == "DECJ":
                DECJ = param[1]
            elif param[0] == "DM":
                DM = np.float64(param[1])
            else:
                pass
    # Dec needs to have a sign for the following sky_coord call
    if (DECJ[0] != '+' and DECJ[0] != '-'):
        DECJ = "+" + DECJ
    arch.set_dispersion_measure(DM)
    arch.set_source(PSR)
    arch.set_coordinates(pr.sky_coord(RAJ + DECJ))
    # Set some other stuff
    arch.set_centre_frequency(nu0)
    arch.set_bandwidth(bw)
    arch.set_telescope(telescope)
    if npol == 4:
        arch.set_state(state)
    # Fill in some subintegration attributes
    if start_MJD is None:
        start_MJD = pr.MJD(50000, 0, 0.0)
    epoch = start_MJD
    epoch += tsub / 2.0  # *Yes* add seconds to days, this is how it works...
    for subint in arch:
        subint.set_epoch(epoch)
        subint.set_duration(tsub)
        epoch += tsub
        for ichan in range(nchan):
            subint.set_centre_frequency(ichan, freqs[ichan])
    # Fill in polycos
    arch.set_ephemeris(ephemeris)
    # Now finally, fill in the data!
    arch.set_dedispersed(True)
    arch.dedisperse()
    if weights is None: weights = np.ones([nsub, nchan])
    isub = 0
    for subint in arch:
        for ipol in range(npol):
            for ichan in range(nchan):
                subint.set_weight(ichan, weights[isub, ichan])
                prof = subint.get_Profile(ipol, ichan)
                prof.get_amps()[:] = data[isub, ipol, ichan]
        isub += 1
    if not dedispersed: arch.dededisperse()
    arch.unload(outfile)
    if not quiet: print("\nUnloaded %s.\n" % outfile)


def make_fake_pulsar(modelfile, ephemeris, outfile="fake_pulsar.fits", nsub=1,
                     npol=1, nchan=512, nbin=2048, nu0=1500.0, bw=800.0, tsub=300.0,
                     phase=0.0, dDM=0.0, start_MJD=None, weights=None, noise_stds=1.0,
                     scales=1.0, dedispersed=False, t_scat=0.0, alpha=scattering_alpha,
                     scint=False, xs=None, Cs=None, nu_DM=np.inf, state="Stokes",
                     telescope="GBT", quiet=False):
    """
    Generate fake pulsar data written to a PSRCHIVE psrfits archive.

    Not guaranteed to work perfectly.

    modelfile is the write_model(...)-type of file specifying the
        Gaussian-component model to use.
    ephemeris is the timing ephemeris to be installed.
    outfile is the desired output file name.
    nsub is the number of subintegrations in the data.
    npol is the number of polarizations in the data.
    nchan is the number of frequency channels in the data.
    nbin is the number of phase bins in the data.
    nu0 is the center frequency [MHz] of the data.
    bw is the bandwidth of the data.
    tsub is the duration of each subintegration [sec].
    phase is an arbitrary rotation [rot] to all subints, with respect to nu0.
    dDM is a dispersion measure [cm**-3 pc] added to what is given by the
        ephemeris. Dispersion occurs at infinite frequency. (??)
    start_MJD is the starting epoch of the data in PSRCHIVE MJD format,
        e.g. pr.MJD(57000.123456789012345).
    weights is a nsub x nchan array of weights.
    noise_stds is the level of the RMS additive noise injected into the data;
        can be either a single float or an array of length nchan.
    scales is an arbitrary scaling to the amplitude parameters in modelfile;
        can be either a single float or an array of length nchan.
    dedispersed=True, will save the archive as dedispered.
    t_scat != 0.0 convolves the data with a scattering timescale t_scat [sec],
        referenced at nu0.
        NB: Should only be used if not provided in modelfile!
    alpha is the scattering index.
        NB: Should only be used if not provided in modelfile!
    scint=True adds random scintillation, based on default parameters. scint
        can also be a list of parameters taken by add_scintillation.
    xs is an array of powers to simulate a DM(nu) effect; see add_DM_nu for
        details.
    Cs is an array of coefficients to simulate a DM(nu) effect; see add_DM_nu
        for details.
    nu_DM is the frequency [MHz] to which the DM refers to, if simulating
        DM(nu) with xs and Cs.
    state is the polarization state of the data ("Coherence" or "Stokes" for
        npol == 4, or "Intensity" for npol == 1).
    telescope is the telescope name.
    quiet=True suppresses output.

    Mostly written by PBD.
    """
    chanwidth = old_div(bw, nchan)
    lofreq = nu0 - (old_div(bw, 2))
    # Channel frequency centers
    freqs = np.linspace(lofreq + (chanwidth / 2.0), lofreq + bw -
                        (chanwidth / 2.0), nchan)
    # Phase bin centers
    phases = get_bin_centers(nbin, lo=0.0, hi=1.0)
    # phases = np.linspace(0.0 + (nbin*2)**-1, 1.0 - (nbin*2)**-1, nbin)
    # Channel noise
    try:
        if len(noise_stds) != nchan:
            print("\nlen(noise_stds) != nchan\n")
            return 0
    except TypeError:
        noise_stds = noise_stds * np.ones(nchan)
    # Channel amplitudes
    try:
        if len(scales) != nchan:
            print("\nlen(scales) != nchan\n")
            return 0
    except TypeError:
        scales = scales * np.ones(nchan)
    # Create the Archive instance.
    # This is kind of a weird hack, if we create a PSRFITS
    # Archive directly, some header info does not get filled
    # in.  If we instead create an ASP archive, it will automatically
    # be (correctly) converted to PSRFITS when it is unloaded...
    arch = pr.Archive_new_Archive("ASP")
    arch.resize(nsub, npol, nchan, nbin)
    try:
        import parfile
        par = parfile.psr_par(ephemeris)
        try:
            PSR = par.PSR
        except AttributeError:
            PSR = par.PSRJ
        DECJ = par.DECJ
        RAJ = par.RAJ
        P0 = par.P0
        PEPOCH = par.PEPOCH
        DM = par.DM
    except ImportError:
        parfile = open(ephemeris, "r").readlines()
        for iline in range(len(parfile)):
            param = parfile[iline].split()
            if param[0] == ("PSR" or "PSRJ"):
                PSR = param[1]
            elif param[0] == "RAJ":
                RAJ = param[1]
            elif param[0] == "DECJ":
                DECJ = param[1]
            elif param[0] == "F0":
                P0 = np.float64(param[1]) ** -1
            elif param[0] == "P0":
                P0 = np.float64(param[1])
            elif param[0] == "PEPOCH":
                PEPOCH = np.float64(param[1])
            elif param[0] == "DM":
                DM = np.float64(param[1])
            else:
                pass
    # Dec needs to have a sign for the following sky_coord call
    if (DECJ[0] != '+' and DECJ[0] != '-'):
        DECJ = "+" + DECJ
    arch.set_dispersion_measure(DM)
    arch.set_source(PSR)
    arch.set_coordinates(pr.sky_coord(RAJ + DECJ))
    # Set some other stuff
    arch.set_centre_frequency(nu0)
    arch.set_bandwidth(bw)
    arch.set_telescope(telescope)
    if npol == 4:
        arch.set_state(state)
    # Fill in some subintegration attributes
    if start_MJD is None:
        start_MJD = pr.MJD(PEPOCH)
    else:
        # start_MJD = pr.MJD(start_MJD)
        start_MJD = start_MJD
    epoch = start_MJD
    epoch += tsub / 2.0  # *Yes* add seconds to days, this is how it works...
    for subint in arch:
        subint.set_epoch(epoch)
        subint.set_duration(tsub)
        epoch += tsub
        for ichan in range(nchan):
            subint.set_centre_frequency(ichan, freqs[ichan])
    # Fill in polycos
    arch.set_ephemeris(ephemeris)
    # Now finally, fill in the data!
    # NB the different pols are not realistic: same model, same noise_stds
    # rotmodel is now set to the unrotated model, and all rotation done by
    # PSRCHIVE since whether or not the PSRCHIVE configuration is set to use the
    # barycentric correction will matter
    arch.set_dedispersed(True)
    arch.dedisperse()
    if weights is None: weights = np.ones([nsub, nchan])
    isub = 0
    (name, model_code, nu_ref, ngauss, params, fit_flags, scattering_index,
     fit_scattering_index) = read_model(modelfile, quiet=True)
    for subint in arch:
        P = subint.get_folding_period()
        for ipol in range(npol):
            name, ngauss, model = read_model(modelfile, phases, freqs, P,
                                             quiet=True)
            if xs is None:
                rotmodel = model
                # rotmodel=rotate_data(model, -phase, -(DM+dDM), P, freqs, nu0)
            else:
                phase = phase_transform(phase, DM + dDM, nu0, nu_DM, P)
                # rotmodel = add_DM_nu(model, -phase, -(DM+dDM), P, freqs, xs,
                #        Cs, nu_DM)
                rotmodel = add_DM_nu(model, -phase, -dDM, P, freqs, xs, Cs,
                                     nu_DM)
            # rotmodel = model
            if t_scat and not params[1]:  # modelfile overrides
                # sk = scattering_kernel(t_scat, nu0, freqs, phases, P,
                #        alpha=alpha)
                # rotmodel = add_scattering(rotmodel, sk, repeat=3)
                taus = scattering_times(old_div(t_scat, P), alpha, freqs, nu0)
                sp_FT = scattering_portrait_FT(taus, nbin)
                rotmodel = np.fft.irfft(sp_FT * \
                                        np.fft.rfft(rotmodel, axis=-1), axis=-1)
            if scint is not False:
                if scint is True:
                    rotmodel = add_scintillation(rotmodel, random=True, nsin=3,
                                                 amax=1.0, wmax=5.0)
                else:
                    rotmodel = add_scintillation(rotmodel, scint)
            for ichan in range(nchan):
                subint.set_weight(ichan, float(weights[isub, ichan]))
                prof = subint.get_Profile(ipol, ichan)
                noise = noise_stds[ichan]
                if noise:
                    prof.get_amps()[:] = scales[ichan] * rotmodel[ichan] + \
                                         np.random.normal(0.0, noise, nbin)
                else:
                    prof.get_amps()[:] = scales[ichan] * rotmodel[ichan]
        isub += 1
    if dedispersed:
        arch.dedisperse()
    else:
        arch.dededisperse()
    arch.unload(outfile)
    if not quiet: print("\nUnloaded %s.\n" % outfile)


def filter_TOAs(TOAs, flag, cutoff, criterion=">=", pass_unflagged=False,
                return_culled=False):
    """
    Filter TOAs based on a flag and cutoff value.

    TOAs is a TOA list from pptoas.
    flag is a string specifying what attribute of the toa is filtered.
    cutoff is the cutoff value for the flag.
    criterion is a string specifying the condition e.g. '>', '<=', etc.
    pass_unflagged=True will pass TOAs if they do not have the flag.
    return_culled=True will return a second list of the filtered out TOAs.
    """
    new_toas = []
    culled_toas = []

    if criterion == ">":
        op = operator.gt
    elif criterion == ">=":
        op = operator.ge
    elif criterion == "<":
        op = operator.lt
    elif criterion == "<=":
        op = operator.le
    elif criterion == "==":
        op = operator.eq
    elif criterio == "!=":
        op = operator.ne
    else:
        print("Undefined criterion {0}".format(criterion))
        print("Defaulting to '=='")
        op = operator.eq

    for toa in TOAs:
        if flag in toa.flags:
            if op(toa.flags[flag], cutoff):
                new_toas.append(toa)
            else:
                culled_toas.appens(toa)
        else:
            if pass_unflagged:
                new_toas.append(toa)
            else:
                culled_toas.append(toa)
    if return_culled:
        return new_toas, return_culled
    else:
        return new_toas


def write_princeton_TOA(TOA_MJDi, TOA_MJDf, TOA_err, nu_ref, dDM, obs='@',
                        name=' ' * 13):
    """
    Write Princeton-style TOAs.

    Additional formats coming soon...

    Princeton Format

    columns     item
    1-1     Observatory (one-character code) '@' is barycenter
    2-2     must be blank
    16-24   Reference (not necessarily observing) frequency [MHz]
    25-44   TOA [MJD] (decimal point must be in column 30 or column 31)
    45-53   TOA uncertainty [us]
    69-78   DM correction [cm**-3 pc]

    TOA_MJDi is the integer part of the TOA's MJD epoch.
    TOA_MJDf is the fractional part of the TOA's MJD epoch.
    TOA_err is the TOA's "1-sigma" uncertainty [us].
    nu_ref is the TOA's reference frequency.
    dDM is this TOA's DM "correction" (cf. tempo's NDDM).
    obs is the observatory code.
    name is whitespace.

    Taken and tweaked from SMR's PRESTO.
    """
    if nu_ref == np.inf: nu_ref = 0.0
    # Splice together the fractional and integer MJDs
    TOA = "%5d" % int(TOA_MJDi) + ("%.13f" % TOA_MJDf)[1:]
    # if dDM != 0.0:
    print(obs + " %13s %8.3f %s %8.3f              %9.5f" % (name, nu_ref, TOA,
                                                             TOA_err, dDM))
    # else:
    #    print obs + " %13s %8.3f %s %8.3f"%(name, nu_ref, TOA, TOA_err)


def write_TOAs(TOAs, inf_is_zero=True, SNR_cutoff=0.0, outfile=None,
               append=True):
    """
    Write loosely-IPTA formatted TOAs to file.

    inf_is_zero=True follows the TEMPO/2 convention of writing 0.0 MHz as the
        frequency for infinite-frequency TOAs.
    TOAs is a single TOA of the TOA class from pptoas, or a list of them.
    SNR_cutoff is a value specifying which TOAs are written based on the snr
        flag
    outfile is the output file name; if None, will print to standard output.
    append=False will overwrite a file with the same name as outfile.
    """
    if not hasattr(TOAs, "__len__"):
        toas = [TOAs]
    else:
        toas = TOAs
    toas = filter_TOAs(toas, "snr", SNR_cutoff, ">=", pass_unflagged=False)
    if outfile is not None:
        if append:
            mode = 'a'
        else:
            mode = 'w'
        of = open(outfile, mode)
    for toa in toas:
        if toa.frequency == np.inf and inf_is_zero:
            toa_string = "%s %.8f %d" % (toa.archive, 0.0,
                                         toa.MJD.intday()) + ("%.15f   %.3f  %s" % (toa.MJD.fracday(),
                                                                                    toa.TOA_error, toa.telescope_code))[
                                                             1:]
        else:
            toa_string = "%s %.8f %d" % (toa.archive, toa.frequency,
                                         toa.MJD.intday()) + ("%.15f   %.3f  %s" % (toa.MJD.fracday(),
                                                                                    toa.TOA_error, toa.telescope_code))[
                                                             1:]
        if toa.DM is not None:
            # toa_string += " -dm %.7f"%toa.DM
            toa_string += " -pp_dm %.7f" % toa.DM
        if toa.DM_error is not None:
            # toa_string += " -dm_err %.7f"%toa.DM_error
            toa_string += " -pp_dme %.7f" % toa.DM_error
        for flag, value in list(toa.flags.items()):
            if value is not None:
                if hasattr(value, "lower"):
                    toa_string += ' -%s %s' % (flag, value)
                elif 'int' in str(type(value)):
                    toa_string += ' -%s %d' % (flag, value)
                elif flag.find("_cov") >= 0:
                    toa_string += ' -%s %.1e' % (flag, toa.flags[flag])
                elif flag.find("phs") >= 0:
                    toa_string += ' -%s %.8f' % (flag, toa.flags[flag])
                elif flag.find("flux") >= 0:
                    toa_string += ' -%s %.5f' % (flag, toa.flags[flag])
                else:
                    toa_string += ' -%s %.3f' % (flag, toa.flags[flag])

        if outfile is not None:
            toa_string += "\n"
            of.write(toa_string)
        else:
            print(toa_string)
    if outfile is not None: of.close()


def show_portrait(port, phases=None, freqs=None, title=None, prof=True,
                  fluxprof=True, rvrsd=False, colorbar=True, savefig=False, show=True,
                  aspect="auto", interpolation="none", origin="lower", extent=None,
                  **kwargs):
    """
    Show a pulsar portrait.

    To be improved.

    port is the nchan x nbin pulsar portrait array.
    phases is the nbin array with phase-bin centers [rot].  Defaults to
        phase-bin indices.
    freqs is the nchan array with frequency-channel centers [MHz]. Defaults to
        frequency-channel indices.
    title is a string to be displayed.
    prof=True adds a panel showing the average pulse profile.
    fluxprof=True adds a panel showing the phase-averaged spectrum.
    rvrsd=True flips the frequency axis.
    colorbar=True adds the color bar.
    savefig specifies a string for a saved figure; will not show the plot.
    aspect sets the aspect ratio
    interpolation sets the interpolation scheme
    origin tells pyplot where to put the (0,0) point (?).
    extent is a 4-element tuple setting the (lo_phase, hi_phase, lo_freq,
        hi_freq) limits on the plot.
    **kwargs get passed to imshow.  e.g. vmin, vmax...
    """
    nn = 2 * 3 * 5  # Need something divisible by 2,3,5...
    grid = gs.GridSpec(nn, nn)
    pi = 0
    fi = 0
    ci = 0
    if freqs is None:
        freqs = np.arange(len(port))
        ylabel = "Channel Number"
    else:
        ylabel = "Frequency [MHz]"
    if phases is None:
        phases = np.arange(len(port[0]))
        xlabel = "Bin Number"
    else:
        xlabel = "Phase [rot]"
    if rvrsd:
        freqs = freqs[::-1]
        port = port[::-1]
    if extent is None:
        extent = (phases[0], phases[-1], freqs[0], freqs[-1])
    if prof or fluxprof:
        weights = port.mean(axis=1)
    if prof:
        portx = np.compress(weights, port, axis=0)
        prof = portx.mean(axis=0)  # NB: this is unweighted mean
        pi = 1
    if fluxprof:
        fluxprof = port.mean(axis=1)
        fluxprofx = np.compress(weights, fluxprof)
        freqsx = np.compress(weights, freqs)
        fi = 1
    if colorbar: ci = 1
    ax1 = plt.subplot(grid[(old_div(pi * nn, 6)):, (old_div(fi * nn, 6)):])
    im = ax1.imshow(port, aspect=aspect, origin=origin, extent=extent,
                    interpolation=interpolation, **kwargs)
    if colorbar: plt.colorbar(im, ax=ax1, use_gridspec=False)
    ax1.set_xlabel(xlabel)
    if not fi:
        ax1.set_ylabel(ylabel)
    else:
        # ytklbs = ax1.get_yticklabels()
        ax1.set_yticklabels(())
    if not pi:
        if title: plt.title(title)
    if pi:
        if ci:
            ax2 = plt.subplot(grid[:(old_div(pi * nn, 6)), (old_div(fi * nn, 6)):old_div(((3 + fi + ci) * nn),
                                                                                         (4 + fi + ci))])
        else:
            ax2 = plt.subplot(grid[:(old_div(pi * nn, 6)), (old_div(fi * nn, 6)):])
        ax2.plot(phases, prof, 'k-')
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels(())
        ax2.set_yticks([0, round(old_div(prof.max(), 2), 1), round(prof.max(), 1)])
        ax2.set_xlim(phases.min(), phases.max())
        rng = prof.max() - prof.min()
        ax2.set_ylim(prof.min() - 0.03 * rng, prof.max() + 0.05 * rng)
        ax2.set_ylabel("Flux Units")
        if title: plt.title(title)
    if fi:
        ax3 = plt.subplot(grid[(old_div(pi * nn, 6)):, :(old_div(fi * nn, 6))])
        ax3.plot(fluxprofx, freqsx, 'kx')
        ax3.set_xticks([0, round(old_div(fluxprofx.max(), 2), 2),
                        round(fluxprofx.max(), 2)])
        rng = fluxprofx.max() - fluxprofx.min()
        ax3.set_xlim(fluxprofx.max() + 0.03 * rng, fluxprofx.min() - 0.01 * rng)
        ax3.set_xlabel("Flux Units")
        ax3.set_yticks(ax1.get_yticks())
        # ax3.set_yticklabels(ytklbs)
        # ax3.set_ylim(freqs[0], freqs[-1])
        ax3.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1])
        ax3.set_ylabel(ylabel)
    # if title: plt.suptitle(title)
    # plt.tight_layout(pad = 1.0, h_pad=0.0, w_pad=0.0)
    if savefig:
        plt.savefig(savefig, format='png')
        plt.close()
    if show:
        plt.show()


def show_stacked_profiles(data_profiles, model_profiles=None, phases=None,
                          freqs=None, rvrsd=False, fit=False, title=None, fact=0.25,
                          savefig=False):
    """
    Show stacked, offset data profiles, with optional overlaid model profiles.

    data_profiles is the nprof(nchan) x nbin array of data profiles.
    model_profiles is the nprof(nchan) x nbin array of model profiles.
    phases is the nbin array with phase-bin centers [rot].  Defaults to
        phase-bin indices.
    freqs is the nchan array with frequency-channel centers [MHz]. Defaults to
        frequency-channel indices.
    rvrsd=True flips the frequency axis.
    fit=True will fit the model profile to the data profile on a per-profile
        basis using fit_phase_shift(...).
    title is a string to be displayed.
    fact is a scaling factor to increase the separation between profiles.
    savefig specifies a string for a saved figure; will not show the plot.
    """
    if model_profiles is None:
        model_profiles = np.copy(data_profiles)
    else:
        pass
    if phases is None:
        phases = np.arange(len(data_profiles[0]))
        xlabel = "Bin Number"
    else:
        xlabel = "Phase [rot]"
    if freqs is None:
        freqs = np.arange(len(data_profiles))
        ylabel = "Approx. Channel Number"
    else:
        ylabel = "Approx. Frequency [MHz]"
    if rvrsd:
        freqs = freqs[::-1]
        data_profiles = data_profiles[::-1]
        model_profiles = model_profiles[::-1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    off = (data_profiles.max() - data_profiles.min()) * fact
    for iprof, dprof in enumerate(data_profiles):
        freq = freqs[iprof]
        mprof = model_profiles[iprof]
        if fit and np.any(dprof - mprof):
            r = fit_phase_shift(dprof, mprof, Ns=100)
            mprof = r.scale * rotate_profile(mprof, -r.phase)
        m, = ax.plot(phases, mprof + iprof * off, lw=2, ls='dashed')
        d, = ax.plot(phases, dprof + iprof * off, lw=2, ls='solid', color=m.get_color())
    ax.set_xlabel(xlabel)
    ax.set_yticks(np.arange(len(data_profiles))[::10] * off)
    ytick_labels = ax.get_yticklabels()
    ytick_labels = freqs[::10]
    # ytick_labels = np.linspace(freqs[0], freqs[-1], len(ytick_labels))
    ytick_labels = list(map(round, ytick_labels))
    ytick_labels = list(map(int, ytick_labels))
    ytick_labels = list(map(str, ytick_labels))
    ax.set_yticklabels(ytick_labels)
    ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)
    if savefig:
        plt.savefig(savefig, format='png')
        plt.close()
    else:
        plt.show()


def show_profiles(model, phases=None, cmap=plt.cm.Spectral, s=1, offset=None,
                  **kwargs):
    """
    Show stacked profiles colored by amplitude; good for displaying models.

    model is an nchan x nbin array of profiles to display.
    phases=None assumes the end phases are 0.0 and 1.0; otherwise are an nbin
        array of phase values to plot against.
    cmap is a matplotlib.colormap instance.
    s is the marker size in squared points.
    offset=None calculates the offset between profiles.
    **kwargs are passed to plt.scatter(...).
    """
    model_min = model.min()
    model_max = model.max()
    model_range = model_max - model_min
    if phases is None:
        phases = get_bin_centers(len(model[0]))
    if offset is None: offset = model_range / float(len(model))
    for iprof, prof in enumerate(model):
        norm_prof = old_div((prof - model_min), model_range)
        c = cmap(norm_prof)
        plt.scatter(phases, prof + (offset * iprof), c=c, edgecolor='none', s=s,
                    **kwargs)


def show_residual_plot(port, model, resids=None, phases=None, freqs=None,
                       noise_stds=None, nfit=0, titles=(None, None, None), rvrsd=False,
                       colorbar=True, savefig=False, aspect="auto", interpolation="none",
                       origin="lower", extent=None, **kwargs):
    """
    Show a portrait, model, and residuals in a single plot.

    To be improved.

    port is the nchan x nbin pulsar portrait array.
    model is the nchan x nbin model portrait array.
    resids=None tells the function to calculate the residuals in situ.
    phases is the nbin array with phase-bin centers [rot].  Defaults to
        phase-bin indices.
    freqs is the nchan array with frequency-channel centers [MHz]. Defaults to
        frequency-channel indices.
    noise_stds is the noise level for calculation of channel red. chi2 values;
        if None, will use default from get_noise(...).
    nfit is the number of dof to subtract from nbin in the calculation of the
        reduced chi2 (i.e., on a per-channel basis).
    titles is a three-element tuple for the titles on each plot.
    rvrsd=True flips the frequency axis.
    colorbar=True adds the color bar.
    savefig specifies a string for a saved figure; will not show the plot.
    aspect sets the aspect ratio
    interpolation sets the interpolation scheme
    origin tells pyplot where to put the (0,0) point (?).
    extent is a 4-element tuple setting the (lo_phase, hi_phase, lo_freq,
        hi_freq) limits on the plot.
    **kwargs get passed to imshow.  e.g. vmin, vmax...
    """
    mm = 6
    nn = (2 * mm) + (old_div(mm, 3))
    grid = gs.GridSpec(nn, nn)
    if freqs is None:
        freqs = np.arange(len(port))
        ylabel = "Channel Number"
    else:
        ylabel = "Frequency [MHz]"
    if phases is None:
        phases = np.arange(len(port[0]))
        xlabel = "Bin Number"
    else:
        xlabel = "Phase [rot]"
    if rvrsd:
        freqs = freqs[::-1]
        port = port[::-1]
        model = model[::-1]
        noise_stds = noise_stds[::-1]
    if extent is None:
        extent = (phases[0], phases[-1], freqs[0], freqs[-1])
    fig = plt.figure(figsize=(8.5, 6.67))
    ax1 = plt.subplot(grid[:mm, :mm])
    im = ax1.imshow(port, aspect=aspect, origin=origin, extent=extent,
                    interpolation=interpolation, **kwargs)
    if colorbar: plt.colorbar(im, ax=ax1, use_gridspec=False)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if titles[0] is None:
        ax1.set_title("")
    else:
        ax1.set_title(titles[0])
    ax2 = plt.subplot(grid[:mm, -mm:])
    im = ax2.imshow(model, aspect=aspect, origin=origin, extent=extent,
                    interpolation=interpolation, vmin=im.properties()['clim'],
                    **kwargs)
    if colorbar: plt.colorbar(im, ax=ax2, use_gridspec=False)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    if titles[1] is None:
        ax2.set_title("")
    else:
        ax2.set_title(titles[1])
    ax3 = plt.subplot(grid[-mm:, :mm])
    if resids is None:
        resids = port - model
    else:
        if rvrsd: resids = resids[::-1]
    im = ax3.imshow(resids, aspect=aspect, origin=origin, extent=extent,
                    interpolation=interpolation, **kwargs)
    if colorbar: plt.colorbar(im, ax=ax3, use_gridspec=False)
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)
    if titles[2] is None:
        ax3.set_title("")
    else:
        ax3.set_title(titles[2])
    ax4 = plt.subplot(grid[-mm:, -mm:])
    weights = port.mean(axis=1)
    portx = np.compress(weights, port, axis=0)
    modelx = np.compress(weights, model, axis=0)
    residsx = np.compress(weights, resids, axis=0)
    if noise_stds is None:
        noise_stdsxs = get_noise(portx, chans=True)
    else:
        noise_stdsxs = np.compress(weights, noise_stds, axis=0)
    channel_red_chi2s = np.empty(len(portx))
    for ichnx in range(len(portx)):
        channel_red_chi2s[ichnx] = get_red_chi2(portx[ichnx], modelx[ichnx],
                                                errs=noise_stdsxs[ichnx], dof=len(portx[ichnx]) - nfit)
    bins = list(np.linspace(0.0, 2.0, 21)) + list(np.linspace(3.0, 10.0, 8)) + \
           list(np.linspace(20.0, 100.0, 9)) + list(np.linspace(200.0, 1000.0,
                                                                9)) + [np.inf]
    ax4.hist(channel_red_chi2s, bins=bins, histtype='step', color='k')
    rchi2_min = min(channel_red_chi2s)
    rchi2_max = max(channel_red_chi2s)
    if np.log10(rchi2_max) - np.log10(rchi2_min) > 2:
        log = True
    else:
        log = False
    ax4.set_xlabel(r"Red. $\chi^2$")
    ax4.set_ylabel("# chans. (total = %d)" % len(portx))
    ax4.set_title(r"Channel Reduced $\chi^2$")
    if log: ax4.semilogx()
    xlim = ax4.get_xlim()
    ax4.set_xlim(0.9 * rchi2_min, 1.1 * rchi2_max)
    ax4.set_xticklabels((), minor=True)
    if savefig:
        plt.savefig(savefig, format='png')
        plt.close()
    else:
        plt.show()


def show_spline_curve_projections(projected_port, tck, freqs, weights=None,
                                  ncoord=None, icoord=None, title=None, savefig=False):
    """
    Show projections of the fitted B-spline curve for profile evolution.

    projected_port is the projected portrait of data into the subspace of basis
        vectors; this is an attribute of a DataPortrait instance once
        make_spline_model is called.
    tck is output from si.splprep that parameterizes the B-spline curve; this
        is an attribute of a DataPortrait instance once make_spline_model is
        called.
    freqs is the array of frequencies corresponding to the profile vectors in
        projected_port.
    weights is the nchan weights used in the spline fit; defaults to ones.
    ncoord is the number of coordinates to examine in one plot.  Defaults to
        all coordinates! 1 <= ncoord <= projected_port.shape[1]
    icoord is a specific coordinate index to plot as a function of profile
        index; overrides ncoord. 0 <= icoord <= projected_port.shape[1]
    title is a string to be displayed.
    savefig specifies a substring for the saved figures; will not show the
        plots.
    """
    nprof, nbin = projected_port.shape
    if icoord is not None:
        ncoord = 1
        if icoord < 0 or icoord > nbin - 1:
            print("0 <= icoord <= projected_port.shape[1] - 1 = %d" % (nbin - 1))
            return 0
        else:
            plot_this_coord = icoord
    else:
        if ncoord is None:
            ncoord = nbin
        elif ncoord < 1 or ncoord > nbin:
            print("1 <= ncoord <= projected_port.shape[1] = %d" % nbin)
            return 0
        else:
            pass
        if ncoord == 1:
            plot_this_coord = 0
        else:
            plot_this_coord = None
    if freqs[0] > freqs[-1]:
        flip = -1  # has negative bandwidth
    else:
        flip = 1
    interp_freqs = np.linspace(freqs.min(), freqs.max(), nprof * 10)
    proj_port_interp = np.array(si.splev(interp_freqs, tck, der=0, ext=0)).T
    knots = np.array(si.splev(tck[0], tck, der=0, ext=0)).T

    size = 3  # inches per plot
    buff = 2.0  # inches
    if ncoord - 1 and plot_this_coord is None:
        fig1 = plt.figure(1, figsize=((ncoord - 1) * size + buff * 1.5,
                                      (ncoord - 1) * size + buff * 1.5))
    if plot_this_coord is None:
        fig2 = plt.figure(2, figsize=(2 * size + buff, ncoord * size + buff))
        axes2 = fig2.subplots(nrows=ncoord, ncols=1, sharex=True, sharey=False,
                              squeeze=True)
    marker = 'o'
    color = 'purple'
    if weights is None:
        ms = np.ones(len(projected_port)) + 3.0
    else:
        ms = (weights - weights.min())
        ms /= (ms.max() / 10.0)
        ms += 1.0 + 4.0  # this should map all weights to ms on [5, 15]
    alpha = np.linspace(0.25, 1.0, nprof)
    for icoord in range(ncoord):
        nplot = (ncoord - icoord - 1)  # number of plots in the column for icoord
        if nplot:
            for iplot in range(ncoord - 1)[-nplot:]:
                ocoord = iplot + 1
                plot_number = ((ncoord - 1) * iplot) + (icoord + 1)
                ax = fig1.add_subplot(ncoord - 1, ncoord - 1, plot_number)
                for iprof, prof in enumerate(projected_port):
                    ax.plot(prof[icoord], prof[ocoord], marker=marker,
                            color=color, ms=ms[iprof], alpha=alpha[iprof],
                            mew=0.0)
                ax.plot(projected_port[:, icoord], projected_port[:, ocoord],
                        color='k', ls='solid', lw=1)
                ax.plot(proj_port_interp[:, icoord], proj_port_interp[:, ocoord],
                        color='green', ls='solid', lw=2)
                ax.plot(knots[:, icoord], knots[:, ocoord], 'k*', ms=10)
                if ocoord == ncoord - 1:
                    ax.set_xlabel(icoord + 1)
                else:
                    ax.tick_params(labelbottom=False)
                if icoord == 0:
                    ax.set_ylabel(ocoord + 1)
                else:
                    ax.tick_params(labelleft=False)

        if plot_this_coord is None:
            nuax = axes2[icoord]
            for iprof, prof in enumerate(projected_port):
                nuax.plot(freqs[iprof], prof[icoord], marker=marker,
                          color=color, ms=ms[iprof], alpha=alpha[iprof], mew=0.0)
            nuax.plot(freqs, projected_port[:, icoord], color='k', ls='solid',
                      lw=1)
            nuax.plot(interp_freqs[::flip], proj_port_interp[:, icoord][::flip],
                      color='green', ls='solid', lw=2)
            nuax.plot(tck[0][::flip], knots[:, icoord][::flip], 'k*', ms=10)
            nuax.set_ylabel("Coordinate %d" % (icoord + 1))
            nuax.get_yaxis().set_label_coords(-0.1, 0.5)
            if icoord == ncoord - 1: nuax.set_xlabel("Frequency [MHz]")

        elif plot_this_coord is not None:
            icoord = plot_this_coord
            fig2 = plt.figure(2, figsize=(size + 2 * buff, size + 2 * buff))
            nuax = fig2.add_subplot(111)
            for iprof, prof in enumerate(projected_port):
                nuax.plot(freqs[iprof], prof[icoord], marker=marker,
                          color=color, ms=ms[iprof], alpha=alpha[iprof], mew=0.0)
            nuax.plot(freqs, projected_port[:, icoord], color='k', ls='solid',
                      lw=1)
            nuax.plot(interp_freqs[::flip], proj_port_interp[:, icoord][::flip],
                      color='green', ls='solid', lw=2)
            nuax.plot(tck[0][::flip], knots[:, icoord][::flip], 'k*', ms=10)
            nuax.set_ylabel("Coordinate %d" % (plot_this_coord + 1))
            nuax.get_yaxis().set_label_coords(-0.1, 0.5)
            nuax.set_xlabel("Frequency [MHz]")
    if ncoord > 2:
        fig1.text(0.025, 0.5, "Coordinate", rotation='vertical',
                  ha='center', va='center')
        fig1.text(0.5, 0.025, "Coordinate", ha='center', va='center')
    elif ncoord == 2:
        ax.set_xlabel("Coordinate 1")
        ax.set_ylabel("Coordinate 2")
    if title is not None:
        if ncoord > 1:
            plt.figure(1)
            plt.suptitle(title + '\n')
            plt.figure(2)
            plt.suptitle(title + '\n')
        else:
            plt.figure(2)
            plt.suptitle(title + '\n')
    if savefig:
        if ncoord > 1:
            fig1.savefig(savefig + ".proj.png", format='png')
            fig2.savefig(savefig + ".freq.png", format='png')
        else:
            fig2.savefig(savefig + ".freq.png", format='png')
        plt.close('all')
    else:
        plt.show()


def show_eigenprofiles(eigprofs=None, smooth_eigprofs=None, mean_prof=None,
                       smooth_mean_prof=None, title=None, xlim=(0.0, 1.0), show_snrs=False,
                       savefig=False):
    """
    Show eigenprofiles and mean profile, with smoothed versions, optionally.

    eigprofs is the ncomp x nbin array of eigenprofiles to be plotted, if
        provided.
    smooth_eigprofs is the ncomp x nbin array of smoothed eigenprofiles to be
        plotted, if provided.
    mean_prof is the nbin mean profile to be plotted, if provided.
    smooth_mean_prof is the nbin smoothed mean profile to be plotted, if
        provided.
    title is the string that will be displayed at the top of the plot.
    xlim is the range of phases to be plotted.
    show_snrs=True will show an estimate of the S/N ratio for the
        eigenprofiles, if both the and the smoothed version are provided; see
        find_significant_eigvec(...) for details.
    savefig specifies a substring for the saved figure; will not show the
        plot.
    """
    plot_eigprofs = plot_seigprofs = plot_mean = plot_smean = False
    if eigprofs is not None: plot_eigprofs = True
    if smooth_eigprofs is not None: plot_seigprofs = True
    if mean_prof is not None: plot_mean = True
    if smooth_mean_prof is not None: plot_smean = True
    neig = 0
    if plot_eigprofs:
        neig = eigprofs.shape[0]
    if plot_seigprofs:
        neig = smooth_eigprofs.shape[0]
    npanel = neig + int(bool(plot_mean + plot_smean))

    size = 3  # inches per plot
    buff = 1  # inches
    fig = plt.figure(figsize=(4 * size + buff, npanel * size + buff))
    axes = fig.subplots(nrows=npanel, ncols=1, sharex=True, sharey=False,
                        squeeze=True)
    if npanel == 1: axes = [axes]
    ieig = 0
    iseig = 0
    if plot_eigprofs and plot_seigprofs and show_snrs:
        ev_snrs = np.zeros(len(eigprofs))
        for ie in range(len(eigprofs)):
            ev = eigprofs[ie]
            se = smooth_eigprofs[ie]
            ev_noise = get_noise(ev) * np.sqrt(len(ev) / 2.0)
            ev_snrs[ie] = old_div(np.sum(np.abs(np.fft.rfft(se)[1:]) ** 2), ev_noise)
            # print "ev_snr", ev_snrs[ie]
    for iax, ax in enumerate(axes):
        if plot_mean and iax == 0:
            phases = get_bin_centers(mean_prof.shape[0])
            ax.plot(phases, mean_prof, 'k:', ms=1, alpha=0.5)
            ax.set_ylabel("Mean profile")
            if title is not None: ax.set_title(title)
            ax.set_xlim(xlim)
        elif plot_eigprofs:
            phases = get_bin_centers(eigprofs.shape[1])
            ax.plot(phases, eigprofs[ieig], 'k:', ms=1, alpha=0.5)
            ax.set_ylabel("Eigenprofile %d" % (ieig + 1))
            ax.set_xlim(xlim)
            ieig += 1
        if plot_smean and iax == 0:
            phases = get_bin_centers(smooth_mean_prof.shape[0])
            ax.plot(phases, smooth_mean_prof, 'k-', lw=2)
            ax.set_ylabel("Mean profile")
            if title is not None: ax.set_title(title)
            ax.set_xlim(xlim)
        elif plot_seigprofs:
            phases = get_bin_centers(smooth_eigprofs.shape[1])
            ax.plot(phases, smooth_eigprofs[iseig], 'k-', lw=2)
            ax.set_ylabel("Eigenprofile %d" % (iseig + 1))
            ax.set_xlim(xlim)
            if show_snrs:
                ax.text(0.9, 0.9, "S/N = %d" % ev_snrs[iseig], ha='center',
                        va='center', transform=ax.transAxes)
            iseig += 1
        ax.get_yaxis().set_label_coords(-0.1, 0.5)
        if iax == len(axes) - 1: ax.set_xlabel("Phase [rot]")
    if savefig:
        plt.savefig(savefig + ".eigvec.png", format='png')
        plt.close('all')


# the below scattering functions were originally defined in pptoaslib.py

def scattering_times(tau, alpha, freqs, nu_tau):
    """
    """
    taus = tau * (old_div(freqs, nu_tau)) ** alpha
    return taus


def scattering_profile_FT(tau, nbin, binshift=binshift):
    """
    Return the Fourier transform of the scattering_profile() function.

    tau is the scattering timescale [rot].
    nbin is the number of phase bins in the profile.
    binshift is a fudge-factor; currently has no effect.

    Makes use of the analytic formulation of the FT of a one-sided exponential
        function.  There is no windowing, since the whole scattering kernel is
        convolved with the pulse profile.

    Note that the function returns the analytic FT sampled nbin/2 + 1 times.
    """
    nharm = old_div(nbin, 2) + 1
    if tau == 0.0:
        scat_prof_FT = np.ones(nharm)
    else:
        harmind = np.arange(nharm)
        # harmind = np.arange(-(nharm-1), (nharm-1))
        # scat_prof_FT = tau**-1 * (tau**-1 + 2*np.pi*1.0j*harmind)**-1
        scat_prof_FT = (1.0 + 2 * np.pi * 1.0j * harmind * tau) ** -1
        # scat_prof_FT *= np.exp(-harmind * 2.0j * np.pi * binshift / nbin)
    return scat_prof_FT


def scattering_portrait_FT(taus, nbin, binshift=binshift):
    """
    """
    nchan = len(taus)
    nharm = old_div(nbin, 2) + 1
    if not np.any(taus):
        scat_port_FT = np.ones([nchan, nharm])
    else:
        scat_port_FT = np.zeros([nchan, nharm], dtype='complex_')
        for ichan in range(nchan):
            scat_port_FT[ichan] = scattering_profile_FT(taus[ichan], nbin,
                                                        binshift)
    # Not sure this is needed;
    # probably has no effect since it is multiplied with other ports w/ 0 mean
    # scat_port_FT[:, 0] *= F0_fact
    return scat_port_FT
