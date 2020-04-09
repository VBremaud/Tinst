"Importation des librairies"

import os
import matplotlib.pyplot as plt
import numpy as np
from spectractor.extractor.spectrum import Spectrum
from spectractor.tools import plot_spectrum_simple
from spectractor.tools import from_lambda_to_colormap
from spectractor.tools import wavelength_to_rgb
from scipy import signal
from scipy.interpolate import interp1d
from scipy import integrate
import scipy as sp
import gc
import sys
import glob
import statistics as sc

"""
startest="/home/tp-home005/vbremau/StageM1/auxtel_first_light-1_spectrum.fits"
"""
#HD107696

startest="/home/tp-home005/vbremau/StageM1/sim_20170530_175_spectrum.fits"

s=Spectrum(startest)
airmass=s.airmass
disperseur=s.disperser_label
star=s.header['TARGET']
lambda_obs=s.lambdas
intensite_obs=s.data
intensite_err=s.err
lambda_reel=s.target.wavelengths[0]
intensite_reel=s.target.spectra[0]

fig=plt.figure(figsize=[15,10])
plt.plot(lambda_reel,intensite_reel,c='red')
plt.plot(lambda_obs,intensite_obs,c='blue')
plt.axis([300,1100,min(intensite_obs)*1.1,max(intensite_obs)*1.1])
plt.xlabel('$\lambda$ (nm)',fontsize=20)
plt.ylabel('intensity',fontsize=20)
plt.title("Spectre test",fontsize=20)
plt.gca().get_xaxis().set_tick_params(labelsize=16)
plt.gca().get_yaxis().set_tick_params(labelsize=16)
plt.grid(True)
fig.tight_layout()
plt.show()

