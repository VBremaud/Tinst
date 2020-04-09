"Importation des librairies"

import os
import matplotlib.pyplot as plt
import numpy as np
from spectractor.extractor.spectrum import Spectrum
from spectractor.tools import plot_spectrum_simple
from spectractor.tools import from_lambda_to_colormap
from scipy import signal
from scipy.interpolate import interp1d
from scipy import integrate
import scipy as sp
import gc
import sys
import glob
import statistics as sc

def convertion_from_spectrator_to_txt(sim,fileday):
    if sim:
        list_spectrums=glob.glob(fileday+"/sim*spectrum*.fits")
        """liste des chemins contenu dans le fichier du jour à étudier en cherchant uniquement ceux qui proviennent
        des simulations du CTIO et qui sont des spectres"""

    else:
        list_spectrums=glob.glob(fileday+"/reduc*spectrum*.fits")
        """liste des chemins contenu dans le fichier du jour à étudier en cherchant uniquement ceux qui proviennent
        des mesures du CTIO et qui sont des spectres"""

    for i in range(len(list_spectrums)): #on peut divisier pour aller plus vite
        startest=list_spectrums[i]
        s=Spectrum(startest)
        airmass=s.airmass

        print(list_spectrums[i][:len(list_spectrums[i])-4])
        disperseur=s.disperser_label
        star=s.header['TARGET']
        lambda_obs=s.lambdas
        intensite_obs=s.data
        intensite_err=s.err
        if s.target.wavelengths==[]:
            fichier=open(list_spectrums[i][:len(list_spectrums[i])-5]+'.txt','w')

            fichier.write('#'+'\t'+star+'\t'+disperseur+'\t'+str(airmass)+'\t'+'\n')
            for j in range(len(lambda_obs)):
                fichier.write(str(lambda_obs[j])+'\t'+str(intensite_obs[j])+'\t'+str(intensite_err[j])+'\n')

            fichier.close()
        else:
            lambda_reel=s.target.wavelengths[0]
            intensite_reel=s.target.spectra[0]

            fichier=open(list_spectrums[i][:len(list_spectrums[i])-5]+'.txt','w')

            fichier.write('#'+'\t'+star+'\t'+disperseur+'\t'+str(airmass)+'\t'+'\n')
            for j in range(len(lambda_reel)):
                if len(lambda_obs)>j:
                    fichier.write(str(lambda_reel[j])+'\t'+str(intensite_reel[j])+'\t'+str(lambda_obs[j])+'\t'+str(intensite_obs[j])+'\t'+str(intensite_err[j])+'\n')
                else:
                    fichier.write(str(lambda_reel[j])+'\t'+str(intensite_reel[j])+'\n')

            fichier.close()

#convertision_from_spectrator_to_txt(True,"/home/tp-home005/vbremau/StageM1/extraction_test")