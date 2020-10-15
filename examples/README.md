Basic Examples
==============


* To get more help:

  - ppalign.py -h
  - ppspline.py -h
  - ppgauss.py -h
  - pptoas.py -h


For a more involved example, run/examine [`example.py`][examplepy].

For a walk-through of basic ppalign, ppspline, and pptoas usage, try the notebook, [`example_make_model_and_TOAs.ipynb`][examplenb].

* Simplest command-line use with minimum arguments:

  - Average some homogeneous data together to make an average portrait:
    ```python
    ppalign.py -M metafile_with_archive_names
    ```

  - Make a smooth model of the data, which can be reconstructed at any frequency within the range of the data:
    ```python
    ppspline.py -d average_psrfits_file.fits
    ```

  - Make a gaussian model (based on averaged data e.g. having used `psradd`):
    ```python
    ppgauss.py -d average_psrfits_file.fits
    ```

  - Measure TOAs & DMs from an archive using a `ppspline` model:
    ```python
    pptoas.py -d psrfits_archive.fits -m model_from_ppspline.spl
    ```

  - Measure TOAs & DMs from an archive using a `ppgauss` model:
    ```python
    pptoas.py -d psrfits_archive.fits -m model_from_ppgauss.gmodel
    ```

  - Measure TOAs, DMs, & nu**-4 delays from an archive using a `ppspline` model:
    ```python
    pptoas.py -d psrfits_archive.fits -m model_from_ppspline.spl --fit_dt4
    ```

  - Measure TOAs, DMs, taus, and alphas from an archive using a `ppgauss` model:
    ```python
    pptoas.py -d psrfits_archive.fits -m model_from_ppgauss.gmodel -fit_scat
    ```

  - Measure TOAs & DMs from an archive using another archive:
    ```python
    pptoas.py -d psrfits_archive.fits -m similar_psrfits_archive.fits
    ```

  - Find overlooked bad channels to zap in data:
    ```python
    ppzap.py -d psrfits_archive.fits -m model_from_ppspline_or_ppgauss
    ```

* More involved command-line use with some additional options (combinable):

  - Make a smooth, parameterized model of the data, but first normalize the data per-channel and output a PSRCHIVE archive showing the reconstruction:
    ```python
    ppspline.py -d psrfits_archive.fits -N prof -a reconstruction.fits
    ```

  - Make a gaussian model based on averaged data archives from different receivers:
    ```python
    ppgauss.py -M metafile_with_archive_names --nu_ref 1500.0 --bw 200.0
    ```

  - Make a gaussian model, but first normalize the data per-channel and fix the first fitted gaussian component's location (fiducial component):
    ```python
    ppgauss.py -d average_psrfits_file.fits --fgauss --norm
    ```

  - Make a gaussian model that includes a fit for a constant scattering timescale (will take longer) and iterate three additional times:
    ```python
    ppgauss.py -d average_psrfits_file.fits --fitscat --niter 3
    ```

  - Make a gaussian model specifying some of the output and show more output:
    ```python
    ppgauss.py -d average_psrfits_file.fits -o ppgauss_model.gmodel -m 2_component_820_MHz_model --verbose
    ```

  - Measure TOAs & DMs from a metafile of archives, but output only a single (average) DM for each archive, and suppress all other output:
    ```python
    pptoas.py -d metafile_with_archive_names -m model_from_ppspline_or_ppgauss --one_DM --quiet
    ```

  - Measure only TOAs in an archive and specify an output file:
    > **NB**: DMs will still be **barycentered** unless `--no_bary` is also used!

    ```python
    pptoas.py -d psrfits_archive.fits -m model_from_ppspline_or_ppgauss -o my_tim_file.tim --fix_DM --no_bary
    ```

  - Add additonal flags (in pairs) to the output TOA lines:
    ```python
    pptoas.py -d psrfits_archive.fits -m model_from_ppspline_or_ppgauss --flags pta,pta_name,release,release_number
    ```

  - Use a different reduced chi-squared threshold for channel zapping, output paz commands to a file, which modify the original input archives that are read in from a metafile:
    ```python
    ppzap.py -d metafile_with_archive_names -m model_from_ppspline_or_ppgauss -t 1.3 -o paz_cmds.out --modify
    ```


[examplepy]: https://github.com/pennucci/PulsePortraiture/blob/master/examples/example.py
[examplenb]: https://github.com/pennucci/PulsePortraiture/blob/master/examples/example_make_model_and_TOAs.ipynb
