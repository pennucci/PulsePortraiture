#!/usr/bin/env python

from distutils.core import setup

setup(name='PulsePortraiture',
      version='0.0',
      description='Data analysis package for wideband pulsar timing',
      author='Tim Pennucci',
      author_email='tim.pennucci@nanograv.org',
      url='http://github.com/pennucci/PulsePortraiture',
      py_modules=['ppalign', 'ppgauss', 'pplib', 'ppspline', 'pptoas', 'pptoaslib', 'ppzap','telescope_codes'],
      scripts=['ppalign.py','ppgauss.py', 'ppspline.py', 'pptoas.py',
          'ppzap.py']
     )
