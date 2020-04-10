# coding: utf8
## Importation des librairies

import os #gestion de fichiers
import matplotlib.pyplot as plt #affichage
import numpy as np #calculs utilisant C
import glob
#A importer
"""
from spectractor.extractor.spectrum import Spectrum #importation des spectres
from spectractor.tools import from_lambda_to_colormap #couleurs des longueurs d'ondes
from spectractor.tools import wavelength_to_rgb #couleurs des longueurs d'ondes
from spectractor.simulation.simulator import AtmosphereGrid # grille d'atmosphère
"""
#CHECK SUR L'ORDI DU MAGISTERE

from scipy import signal #filtre savgol pour enlever le bruit
from scipy.interpolate import interp1d #interpolation
from scipy import integrate #integation
import scipy as sp #calculs
import statistics as sc #statistiques

## Convolution

def smooth(x,window_len,window,sigma):
    """
    Fonction: effectue la convolution d'un tableau de taille N et renvoie le tableau convolue de taille N.
    Entrees:
        x: tableau à convoluer de taille N
        window_len: taille de la fenêtre de convolution
        window: fonction de convolution (hamming, blackman, hanning)
        sigma: ecart type de la gaussienne dans le cas d'une convolution par une gaussienne

    Sortie:
        y: tableau x convolue de même taille N
    """

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    elif window == 'gaussian':
        if sigma==0:
            return x
        else:
            w=signal.gaussian(window_len,sigma)
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    if window_len%2==0: # even case
        y=y[int(window_len/2):-int(window_len/2)+1]
        return y
    else:           #odd case
        y=y[int(window_len/2-1)+1:-int(window_len/2-1)-1]
        return y


def plot_detection_raies(fichier=r'\Users\Vincent\Documents\Stage J.Neveu\Programmes et prod\CTIODataJune2017 prod4\sim_20170530_060_spectrum.txt',nb_tour=1,CALSPEC=False,trigger=2,lambdamin=250,lambdamax=1150,filtre1_window=61,filtre1_order=3,moy_raies=10,demi_taille_max=40):

    s=open(fichier,'r')
    "importation des donnees"
    intensite_obs=[] #liste des intensités brut
    lambda_obs=[] #liste des longueurs d'ondes brut
    VouF2=True
    for line in s:
        if VouF2:
            a=line.split()
            Airmass=float(a[3])
            VouF2=False
        else:
            if CALSPEC==True:
                a=line.split()
                if len(a)>1:
                    if float(a[0])>lambdamin and float(a[0])<lambdamax:
                        lambda_obs.append(float(a[0]))
                        intensite_obs.append(float(a[1]))


            else: #simu ou données
                a=line.split()
                if len(a)>3:
                    if float(a[2])>lambdamin and float(a[2])<lambdamax:
                        lambda_obs.append(float(a[2]))
                        intensite_obs.append(float(a[3]))


    "Recuperation des donnees du fichier spectre"

    intensite_obs_savgol=sp.signal.savgol_filter(intensite_obs,11,filtre1_order) #filtre savgol (enlève le bruit)
    intensite_obs_1=sp.interpolate.interp1d(lambda_obs,intensite_obs_savgol,kind='cubic')

    lambda_complet=np.linspace(lambda_obs[0],lambda_obs[-1],int((lambda_obs[-1]-lambda_obs[0])*10+1)) #précison Angtrom
    Intensite_obs=intensite_obs_1(lambda_complet)

    for z in range(nb_tour):
        intensite_tronque=[Intensite_obs[0]]
        lambda_tronque=[lambda_complet[0]]
        c=0
        for i in range(1,len(lambda_complet)-1):
            if (Intensite_obs[i+1]-Intensite_obs[i-1])/(lambda_complet[i+1]-lambda_complet[i-1])>0:
                c=1

            elif c==1 and (Intensite_obs[i+1]-Intensite_obs[i-1])/(lambda_complet[i+1]-lambda_complet[i-1])<0:
                intensite_tronque.append(Intensite_obs[i])
                lambda_tronque.append(lambda_complet[i])
                c=0

        intensite_tronque.append(Intensite_obs[-1])
        lambda_tronque.append(lambda_complet[-1])

        if CALSPEC==False:
            intensite_tronque=intensite_obs_savgol
            lambda_tronque=lambda_obs

        for j in range(100):
            intensite_tronque2=[Intensite_obs[0]]
            lambda_tronque2=[lambda_complet[0]]
            c=0
            for i in range(1,len(lambda_tronque)-1):
                if intensite_tronque[i-1]<intensite_tronque[i] or intensite_tronque[i+1]<intensite_tronque[i]:
                    intensite_tronque2.append(intensite_tronque[i])
                    lambda_tronque2.append(lambda_tronque[i])

            intensite_tronque2.append(Intensite_obs[-1])
            lambda_tronque2.append(lambda_complet[-1])

            intensite_tronque=intensite_tronque2
            lambda_tronque=lambda_tronque2

        Intensite_obss=sp.interpolate.interp1d(lambda_tronque,intensite_tronque,bounds_error=False,fill_value="extrapolate")
        Intensite_obs=Intensite_obss(lambda_complet)

    intensite_obss=Intensite_obs

    INTENSITE_OBS=intensite_obss

    INTENSITE_OBS=smooth(INTENSITE_OBS,155,'flat',1)
    INTENSITE_OBSS=INTENSITE_OBS
    INTENSITE_OBSS=sp.signal.savgol_filter(INTENSITE_OBS,filtre1_window*3,filtre1_order)


    fig=plt.figure(figsize=[15,10])
    plt.axis([300,1100,0,max(intensite_obs)*1.1])
    plt.plot(lambda_obs,intensite_obs,c='r')
    plt.plot(lambda_complet,INTENSITE_OBSS,c='black',linestyle='--')
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('u.a')
    plt.title('spectre observe final')
    plt.grid(True)

    plt.show()
