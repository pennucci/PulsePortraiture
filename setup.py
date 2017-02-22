#!/usr/bin/env python

from distutils.core import setup

setup(name='PulsePortraiture',
      version='0.0',
      description='Data analysis package for wide-band pulsar timing',
      author='Tim Pennucci',
      author_email='tim.pennucci@nanograv.org',
      url='http://github.com/pennucci/PulsePortraiture',
      py_modules=['pplib'],
      scripts=['ppalign.py','ppgauss.py', 'ppinterp.py', 'pptoas.py']
     )
