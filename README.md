Pulse Portraiture
=================


## What?

A set of libraries and modules to measure "wideband" pulse times-of-arrival (TOAs), written in python. It uses an extension of Joe Taylor's **FFTFIT** algorithm (**Taylor 1992**) to simultaneously measure a phase (TOA) and dispersion measure (DM).  It has subsequently been improved to also incoporate fitting for scattering parameters (timescale tau and index alpha) and frequency**-4 phase delays ("GM"). It is to be used with [PSRCHIVE][psrchive]-compatible folded archives ([PSRFITS][psrfits] format).

## Why?

The motivation behind writing this software was to develop a wideband measurement routine for high-precision pulsar timing in the era of very broadband receivers and high-cadence timing observations for PTA experiments, and it makes up a chunk of Tim Pennucci's Ph.D. thesis. Algorithm development and coding help was provided by Paul Demorest and Scott Ransom.

## How?

The technical description of this work and its related papers are:

* [Pennucci, Demorest, & Ransom (2014), "_Elementary Wideband Timing of Radio Pulsars_", ApJ, 790, 93][2014].

  > **NB**: Equation **13** of the paper was written incorrectly: It should be the reverse:
  >
  > DM<sub>bary</sub> = DM<sub>topo</sub> / doppler_factor.

* [Pennucci (2015), "_Wideband Observations of Radio Pulsars_", PhDT, UVa][2015].
* [Pennucci (2019), "_Frequency-dependent Template Profiles for High-precision Pulsar Timing_", ApJ, 871, 1][2019].
* Pennucci et al. (in prep).

## Requirements

* [**PSRCHIVE**][psrchive], compiled with the python-interface enabled,
* [**NumPy**][numpy] & [**SciPy**][scipy] recent versions will do,
* [**PyWavelets**][pywt] is required for wavelet smoothing / ppspline.py, and
* [**LMFIT**][lmfit] is required for Gaussian portrait modelling / ppgauss.py and a few other functions.

## TL;DR

* [`pplib`][pplib] contains functions and classes needed for the fitting scripts.
* [`ppspline`][ppspline] is a command-line utility to build smoothly varying model portraits based on PCA decomposition, wavelet smoothing, and B-splin einterpolation between the components.
* [`ppgauss`][ppgauss] is a command-line utility to build Gaussian-component model portraits.
* [`pptoaslib`][pptoaslib] contains functions needed for pptoas.
* [`pptoas`][pptoas] is a command-line utility to measure TOAs, DMs, nu**-4 delays, and scattering parameters.
* [`ppalign`][ppalign] is a command-line utility to average homogeneous data by measuring phases and DMs.
* [`ppzap`][ppzap] is a command-line utility which uses pptoas to identify potentially overlooked bad channels to zap.
* The command-line programs can be imported into ipython for additional flexibility of use.
* See the [**examples**][examples] directory for simple command-line use.
* Run and examine [**examples/**`example.py`][examplepy] for a more in-depth demonstration.
* Try the notebook [`example_make_model_and_TOAs.ipynb`][examplenb] for a walk-through.

## License

Released under **GPLv2**, sans "or later" clause.

## Other

Code improvements are underway, as is a broad application to IPTA pulsars of interest. [Suggestions and additional development are welcome](https://github.com/pennucci/PulsePortraiture).

[psrfits]: https://www.atnf.csiro.au/research/pulsar/psrfits_definition/Psrfits.html

[2014]: https://doi.org/10.1088/0004-637X/790/2/93
[2015]: https://doi.org/10.18130/V3W56C
[2019]: https://doi.org/10.3847/1538-4357/aaf6ef

[psrchive]: http://psrchive.sourceforge.net/
[numpy]: https://numpy.org/
[scipy]: https://www.scipy.org/
[pywt]: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html
[lmfit]: https://lmfit.github.io/lmfit-py/index.html

[pplib]: https://github.com/pennucci/PulsePortraiture/blob/master/pplib.py
[ppspline]: https://github.com/pennucci/PulsePortraiture/blob/master/ppspline.py
[ppgauss]: https://github.com/pennucci/PulsePortraiture/blob/master/ppgauss.py
[pptoaslib]: https://github.com/pennucci/PulsePortraiture/blob/master/pptoaslib.py
[pptoas]: https://github.com/pennucci/PulsePortraiture/blob/master/pptoas.py
[ppalign]: https://github.com/pennucci/PulsePortraiture/blob/master/ppalign.py
[ppzap]: https://github.com/pennucci/PulsePortraiture/blob/master/ppzap.py
[examples]: https://github.com/pennucci/PulsePortraiture/tree/master/examples
[examplepy]: https://github.com/pennucci/PulsePortraiture/blob/master/examples/example.py
[examplenb]: https://github.com/pennucci/PulsePortraiture/blob/master/examples/example_make_model_and_TOAs.ipynb
