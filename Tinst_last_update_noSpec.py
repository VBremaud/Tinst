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
    if window == 'gaussian':
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

#CHECK

## Atmosphère

def plot_atmo(a1,a2,a3,coeff_sigma1=5,coeff_sigma2=3,coeff_sigma3=3,delta=5,demi_taille_max=120,coeff_sigmabis1=3,coeff_sigmabis2=3,coeff_sigmabis3=2,deltabis=10,demi_taille_maxbis=120,plot=True):
    """
    Fonction: calcul et renvoie l'atmosphère sans raie à partir d'un modèle d'atmosphère Libradtran ou affiche
    les differentes etapes du detecteur de raies (parcouru 2 fois). La valeur des paramètres par defaut est
    normalement suffisante pour l'extraction de l'atmosphère sans raies.

    Entrees:
    a1: paramètre d'ozone
    a2: paramètre PWV (vapeur d'eau preciptable)
    a3: paramètre d'aerosols

    coeff_sigma1: premier seuil de detection du detecteur de raies 1er passsage
    coeff_sigma2: deuxième seuil de detection du detecteur de raies 1er passsage
    coeff_sigma3: troisième seuil de detection du detecteur de raies 1er passsage
    coeff_sigmabis1: premier seuil de detection du detecteur de raies 2e passsage
    coeff_sigmabis2: deuxième seuil de detection du detecteur de raies 2e passsage
    coeff_sigmabis3: troisième seuil de detection du detecteur de raies 2e passsage

    delta: nombre de pixel avant le premier seuil et après le dernier que l'on considère comme faisant partie
    de la raie lors du 1er passage.
    deltabis: nombre de pixel avant le premier seuil et après le dernier que l'on considère comme faisant partie
    de la raie lors du 2e passage.

    demi_taille_max: plage maximale en pixel entre le premier seuil et le deuxième seuil de detection, 1er passage.
    demi_taille_maxbis: plage maximale en pixel entre le premier seuil et le deuxième seuil de detection, 2e passage.

    plot: bouleen decidant les sorties de la fonction (True: affichage, False: retourne l'atmosphère sans raies)

    Sorties:
    Si plot==True:
        affiche la fin de la première etape.
        affiche la fin de la deuxième etape.
        ne renvoie rien.

    Si plot==False:
        n'affiche rien.
        renvoie une fonction d'interpoation de l'atmosphère sans raies.
    """

    #fichier au hasard pour atmgrid (Attention: determination ici de l'airmass prise dans la simulation)
    atmgrid = AtmosphereGrid(filename="/Users/bremaud/CTIODataJune2017_reduced_RG715_v2_prod4/data_30may17_A2=0/reduc_20170530_135_atmsim.fits")
    a = atmgrid.simulate(a1,a2,a3)


    lambda_reel=np.arange(320,1050,1) #contient les longueurs d'ondes de l'atmosphère simulee.
    intensite_reel=a(lambda_reel) #contient la transmission de l'atmosphère simulee avec raies (taille N)

    intensite_reels=intensite_reel #on utilise le même algo que pour les spectres, ici on ne convolue par rien d'où le '='.
    intensite_reelss=sp.signal.savgol_filter(intensite_reels, 7, 3) #filtre savgol (enlève le bruit)

    "On commente un seul passage du detecteur de raie, le processus est le même dans les fonctions qui suivent."

    """
    Initialisation:
        D_intensite_reelss est un tableau contenant la derivee de l'atmosphère simulee, taille N (on rajoute un 0, sans doute inutile)
        Raies est un tableau de bouleen de taille N-1 indiquant pour chaque intensite si il s'agit d'une raie ou non.
        D_mean est un tableau contenant la valeur moyenne des 10 dernières valeurs de derivees avant l'indice considere qui ne correspondent pas à des raies.
        D_sigma est un tableau contenant l'ecart type des 10 dernières valeurs de derivees avant l'indice considere qui ne correspondent pas à des raies.
    """

    D_intensite_reelss=[]

    for i in range(len(intensite_reelss)-1):
        D_intensite_reelss.append((intensite_reelss[i+1]-intensite_reelss[i])/(lambda_reel[i+1]-lambda_reel[i]))

    D_intensite_reelss.append(0)

    D_mean=[]
    D_sigma=[]
    Raies=[]

    for i in range(10):
        Raies.append(False)
        D_mean.append(np.mean(intensite_reelss[:10]))
        D_sigma.append(np.std(intensite_reelss[:10]))

    i=10
    while i<len(D_intensite_reelss)-1:
        moy=[]
        j=1
        Raies.append(False)
        while len(moy)!=10:
            if Raies[i-j]==False:
                moy.append(D_intensite_reelss[i-j])
            j+=1
        D_mean.append(np.mean(moy))
        D_sigma.append(np.std(moy))

        if D_intensite_reelss[i]<D_mean[len(D_mean)-1]-coeff_sigma1*D_sigma[len(D_sigma)-1]:
            k=i
            while lambda_reel[k]-lambda_reel[i]<demi_taille_max and k<len(lambda_reel)-1:
                k+=1
            for j in range(i,k):
                if D_intensite_reelss[j]>D_mean[len(D_mean)-1]+coeff_sigma2*D_sigma[len(D_sigma)-1]:
                    if j+k-i<=len(lambda_reel):
                        indice=j+k-i
                    else:
                        indice=len(lambda_reel)
                    INDICE=indice
                    for v in range(j,indice):
                        if D_intensite_reelss[v]<D_mean[len(D_mean)-1]+coeff_sigma3*D_sigma[len(D_sigma)-1]:
                            indice=v
                            break

                    if indice!=INDICE:
                        if indice+delta>len(lambda_reel):
                            end=len(lambda_reel)
                        else:
                            end=indice+delta
                        for loop in range(i+1,indice+delta):
                            Raies.append(True)
                        for loop in range(i-delta,i+1):
                            Raies[i]=True
                        i=end
                        Raies.append(False)
                    break
        i+=1

    intensite_coupe=[]
    lambda_coupe=[]
    D_intensiteR_coupe=[]

    if len(intensite_reelss)<len(Raies):
        for i in range(len(Raies)-len(intensite_reelss)):
            Raies.pop()

    for i in range(len(Raies)):
        if Raies[i]==False:
            intensite_coupe.append(intensite_reelss[i])
            lambda_coupe.append(lambda_reel[i])
            D_intensiteR_coupe.append(D_intensite_reelss[i])
            endfalse=i

    intensite_reelSpline=sp.interpolate.interp1d(lambda_coupe,intensite_coupe)

    #-------Deuxième passage dans le detecteur--------#

    intensite_reelss=intensite_reelSpline(lambda_reel[10:-10])
    lambda_reel=lambda_reel[10:-10]

    D_intensite_reelss=[]
    Raies=[]

    for i in range(len(intensite_reelss)-1):
        D_intensite_reelss.append((intensite_reelss[i+1]-intensite_reelss[i])/(lambda_reel[i+1]-lambda_reel[i]))

    D_intensite_reelss.append(0)

    D_mean=[]
    D_sigma=[]
    Raies=[]

    for i in range(10):
        Raies.append(False)
        D_mean.append(np.mean(intensite_reelss[:10]))
        D_sigma.append(np.std(intensite_reelss[:10]))

    i=10
    while i<len(D_intensite_reelss)-1:
        moy=[]
        j=1
        Raies.append(False)
        while len(moy)!=10:
            if Raies[i-j]==False:
                moy.append(D_intensite_reelss[i-j])
            j+=1
        D_mean.append(np.mean(moy))
        D_sigma.append(np.std(moy))

        if D_intensite_reelss[i]<D_mean[len(D_mean)-1]-coeff_sigmabis1*D_sigma[len(D_sigma)-1]:
            k=i
            while lambda_reel[k]-lambda_reel[i]<demi_taille_maxbis and k<len(lambda_reel)-1:
                k+=1
            for j in range(i,k):
                if D_intensite_reelss[j]>D_mean[len(D_mean)-1]+coeff_sigmabis2*D_sigma[len(D_sigma)-1]:
                    if j+k-i<=len(lambda_reel):
                        indice=j+k-i
                    else:
                        indice=len(lambda_reel)
                    INDICE=indice
                    for v in range(j,indice):
                        if D_intensite_reelss[v]<D_mean[len(D_mean)-1]+coeff_sigmabis3*D_sigma[len(D_sigma)-1]:
                            indice=v
                            break

                    if indice!=INDICE:
                        if indice+deltabis>len(lambda_reel):
                            end=len(lambda_reel)
                        else:
                            end=indice+deltabis
                        for loop in range(i+1,indice+deltabis):
                            Raies.append(True)
                        for loop in range(i-deltabis,i+1):
                            Raies[i]=True
                        i=end
                        Raies.append(False)
                    break
        i+=1

    intensite_coupe=[]
    lambda_coupe=[]
    D_intensiteR_coupe=[]

    if len(intensite_reelss)<len(Raies):
        for i in range(len(Raies)-len(intensite_reelss)):
            Raies.pop()

    for i in range(len(Raies)):
        if Raies[i]==False:
            intensite_coupe.append(intensite_reelss[i])
            lambda_coupe.append(lambda_reel[i])
            D_intensiteR_coupe.append(D_intensite_reelss[i])
            endfalse=i

    #-----Detection manuel-----#

    i=0
    intensite_coupebis=[]
    lambda_coupebis=[]
    VouF=True
    for i in range(len(lambda_coupe)):
        if lambda_coupe[i]<780:
            if lambda_coupe[i]<755 or lambda_coupe[i]>770:
                lambda_coupebis.append(lambda_coupe[i])
                intensite_coupebis.append(intensite_coupe[i])


        if lambda_coupe[i]>860 and lambda_coupe[i]<900:
            lambda_coupebis.append(lambda_coupe[i])
            intensite_coupebis.append(intensite_coupe[i])

        if lambda_coupe[i]>900:
            if VouF:
                for e in np.linspace(lambda_coupe[i-1],894,20):
                    lambda_coupebis.append(e)
                    intensite_coupebis.append(intensite_coupe[i-1])
                VouF=False
            if lambda_coupe[i]>990:
                lambda_coupebis.append(lambda_coupe[i])
                intensite_coupebis.append(intensite_coupe[i])



    if plot==False:
        return sp.interpolate.interp1d(lambda_coupebis,intensite_coupebis)

    intensite_reelSpline=sp.interpolate.interp1d(lambda_coupebis,intensite_coupebis)
    plt.figure(figsize=[20,20])

    plt.subplot(2, 2, 1)

    plt.axis([300,1100,0,max(intensite_reel)*1.1])
    plt.plot(lambda_reel,intensite_reelss,lw=3,c='red')
    plt.xlabel('$\lambda$ (nm)',fontsize=24)
    plt.ylabel('u.a',fontsize=24)
    plt.title('Atmosphère après le premier passage',fontsize=24)
    plt.gca().get_xaxis().set_tick_params(labelsize=20)
    plt.gca().get_yaxis().set_tick_params(labelsize=20)
    plt.grid(True)

    plt.subplot(2, 2, 3)

    plt.axis([300,1100,min(D_intensite_reelss)*1.1,max(D_intensite_reelss)*1.1])
    plt.plot(lambda_reel,D_intensite_reelss,marker='.',zorder=1,label='derivee')
    plt.scatter(lambda_coupe,D_intensiteR_coupe,marker='.',c='r',zorder=2,label='conservee')
    plt.title("Derivee de l'atmosphère",fontsize=26)
    plt.grid(True)
    plt.xlabel('$\lambda$ (nm)',fontsize=24)
    plt.ylabel('u.a',fontsize=24)
    plt.gca().get_xaxis().set_tick_params(labelsize=20)
    plt.gca().get_yaxis().set_tick_params(labelsize=20)
    plt.legend(prop={'size':22},loc='upper right')

    plt.subplot(2, 2, 2)

    plt.axis([300,1100,0,max(intensite_reelss)*1.1])
    plt.plot(lambda_coupebis,intensite_coupebis,' .')
    plt.title('Atmosphère sans raies',fontsize=26)
    plt.grid(True)
    plt.xlabel('$\lambda$ (nm)',fontsize=24)
    plt.ylabel('u.a',fontsize=24)
    plt.gca().get_xaxis().set_tick_params(labelsize=20)
    plt.gca().get_yaxis().set_tick_params(labelsize=20)

    plt.subplot(2, 2, 4)

    plt.axis([300,1100,0,max(intensite_reelss)*1.1])
    plt.plot(lambda_reel,intensite_reelss,c='r',label='Atmosphère après premier passage')
    plt.plot(lambda_reel[20:-20],intensite_reelSpline(lambda_reel[20:-20]),c='black',lw=3,label='Atmosphère finale')
    plt.title('Atmosphère finale',fontsize=26)
    plt.grid(True)
    plt.xlabel('$\lambda$ (nm)',fontsize=24)
    plt.ylabel('u.a',fontsize=24)
    plt.gca().get_xaxis().set_tick_params(labelsize=20)
    plt.gca().get_yaxis().set_tick_params(labelsize=20)
    plt.legend(prop={'size':22},loc='lower right')

    #plt.savefig('/Users/bremaud/Documents/raies.pdf')
    plt.show()

#CHECK

## Lissage des raies

def plot_spectre(fichier,sigma):

    "Recuperation des donnees du fichier spectre"
    s=Spectrum(fichier)

    lambda_obs=s.lambdas
    intensite_obs=s.data
    lambda_reel=s.target.wavelengths[0]
    intensite_reel=s.target.spectra[0]

    intensite_reels=smooth(intensite_reel,4*sigma,'gaussian',sigma) #convolution par une gaussienne
    intensite_reelss=sp.signal.savgol_filter(intensite_reels, 7, 3) #filtre savgol (enlève le bruit)
    intensite_obss=sp.signal.savgol_filter(intensite_obs, 17, 3) #filtre savgol (enlève le bruit)

    "Detecteur de raies"

    D_intensite_reelss=[]
    Raies=[]

    for i in range(len(intensite_reelss)-1):
        D_intensite_reelss.append((intensite_reelss[i+1]-intensite_reelss[i])/(lambda_reel[i+1]-lambda_reel[i]))

    D_intensite_reelss.append(0)

    D_mean=[]
    D_sigma=[]
    Raies=[]

    for i in range(10):
        Raies.append(False)
        D_mean.append(np.mean(intensite_obss[:10]))
        D_sigma.append(np.std(intensite_obss[:10]))

    i=10
    while i<len(D_intensite_reelss)-1:
        moy=[]
        j=1
        Raies.append(False)
        while len(moy)!=10:
            if Raies[i-j]==False:
                moy.append(D_intensite_reelss[i-j])
            j+=1
        D_mean.append(np.mean(moy))
        D_sigma.append(np.std(moy))

        if D_intensite_reelss[i]<D_mean[len(D_mean)-1]-2*D_sigma[len(D_sigma)-1]:
            k=i
            while lambda_reel[k]-lambda_reel[i]<50:
                k+=1
            for j in range(i,k):
                if D_intensite_reelss[j]>D_mean[len(D_mean)-1]+2*D_sigma[len(D_sigma)-1]:
                    indice=j+k-i
                    for v in range(j,j+k-i):
                        if D_intensite_reelss[v]<D_mean[len(D_mean)-1]+2*D_sigma[len(D_sigma)-1]:
                            indice=v
                            break

                    if indice!=j+k-i:
                        for loop in range(i+1,indice+15):
                            Raies.append(True)
                        for loop in range(i-15,i+1):
                            Raies[i]=True
                        i=indice+15
                        Raies.append(False)
                    break
        i+=1

    intensite_coupe=[]
    lambda_coupe=[]
    D_intensiteR_coupe=[]
    for i in range(len(Raies)):
        if Raies[i]==False:
            D_intensiteR_coupe.append(D_intensite_reelss[i])
            intensite_coupe.append(intensite_reelss[i])
            lambda_coupe.append(lambda_reel[i])

    intensite_reelSpline=sp.interpolate.interp1d(lambda_coupe,intensite_coupe)

    alpha=1.5
    D_intensite_obss=[]
    Raies=[]

    for i in range(len(intensite_obss)-1):
        D_intensite_obss.append((intensite_obss[i+1]-intensite_obss[i])/(lambda_obs[i+1]-lambda_obs[i]))

    D_intensite_obss.append(0)

    D_mean=[]
    D_sigma=[]
    Raies=[]

    for i in range(10):
        Raies.append(False)
        D_mean.append(np.mean(intensite_obss[:10]))
        D_sigma.append(np.std(intensite_obss[:10]))

    i=10
    while i<len(D_intensite_obss)-1:
        moy=[]
        j=1
        Raies.append(False)
        while len(moy)!=10:
            if Raies[i-j]==False:
                moy.append(D_intensite_obss[i-j])
            j+=1
        D_mean.append(np.mean(moy))
        D_sigma.append(np.std(moy))

        if D_intensite_obss[i]<D_mean[len(D_mean)-1]-alpha*D_sigma[len(D_sigma)-1]:
            k=i
            while lambda_obs[k]-lambda_obs[i]<40 and k<len(lambda_obs)-1:
                k+=1
            for j in range(i,k):
                if D_intensite_obss[j]>D_mean[len(D_mean)-1]+alpha*D_sigma[len(D_sigma)-1]:
                    indice=j+k-i
                    for v in range(j,j+k-i):
                        if D_intensite_obss[v]<D_mean[len(D_mean)-1]+alpha*D_sigma[len(D_sigma)-1]:
                            indice=v
                            break

                    if indice!=j+k-i:
                        for loop in range(i+1,indice+4):
                            Raies.append(True)
                        for loop in range(i-4,i+1):
                            Raies[i]=True
                        i=indice+4
                        Raies.append(False)
                    break
        i+=1

    intensite_coupe_obs=[]
    lambda_coupe_obs=[]
    D_intensite_coupe=[]

    for i in range(len(Raies)):
        if Raies[i]==False:

            intensite_coupe_obs.append(intensite_obss[i])
            lambda_coupe_obs.append(lambda_obs[i])
            D_intensite_coupe.append(D_intensite_obss[i])

    intensite_obsSpline=sp.interpolate.interp1d(lambda_coupe_obs,intensite_coupe_obs)


    fig=plt.figure(figsize=[15,10])
    plt.axis([300,1040,0,1.1])
    plt.plot(lambda_obs,intensite_obss/max(intensite_obss),color='blue',label='spectre observe')
    plt.plot(lambda_reel,intensite_reel/max(intensite_reel),color='red',label='spectre CALSPEC')
    plt.xlabel('$\lambda$ (nm)',fontsize=20)
    plt.ylabel('flux normalise',fontsize=20)
    plt.title("Spectre observe et spectre CALSPEC",fontsize=20)
    plt.gca().get_xaxis().set_tick_params(labelsize=16)
    plt.gca().get_yaxis().set_tick_params(labelsize=16)
    plt.grid(True)
    plt.legend(prop={'size':16},loc='upper right')
    fig.tight_layout()
    #plt.savefig('/Users/bremaud/Documents/spec_obs_CALSPEC.pdf')
    plt.show()

    fig=plt.figure(figsize=[15,12])
    plt.subplot(2, 2, 3)

    plt.axis([300,1040,min(D_intensite_obss)*1.5,max(D_intensite_obss)*1.1])
    plt.plot(lambda_obs,D_intensite_obss,' .',label='derivee du spectre')
    plt.plot(lambda_coupe_obs,D_intensite_coupe,' .',c='r',label='spectre conserve')
    plt.title('Derivee du spectre observe',fontsize=26)
    plt.grid(True)
    plt.xlabel('$\lambda$ (nm)',fontsize=24)
    plt.ylabel('flux (erg/cm^2/nm)',fontsize=24)
    plt.gca().get_xaxis().set_tick_params(labelsize=20)
    plt.gca().get_yaxis().set_tick_params(labelsize=20)
    plt.text(550,max(D_intensite_obss)*0.65,'b)',fontsize=40)
    plt.legend(prop={'size':22},loc='upper right')

    plt.subplot(2, 2, 2)

    plt.axis([300,1040,0,max(intensite_obss)*1.1])
    plt.plot(lambda_coupe_obs,intensite_coupe_obs,' .')
    plt.title('Spectre observe sans raies',fontsize=26)
    plt.grid(True)
    plt.xlabel('$\lambda$ (nm)',fontsize=24)
    plt.ylabel('flux (erg/cm^2/nm)',fontsize=24)
    plt.text(750,max(intensite_obss)*0.85,'c)',fontsize=40)
    plt.gca().get_xaxis().set_tick_params(labelsize=20)
    plt.gca().get_yaxis().set_tick_params(labelsize=20)

    plt.subplot(2, 2, 4)

    plt.axis([300,1040,0,max(intensite_obss)*1.1])
    plt.plot(lambda_obs,intensite_obs,c='r',label='spectre observe initial')
    plt.plot(lambda_obs[20:len(lambda_obs)-20],intensite_obsSpline(lambda_obs[20:len(lambda_obs)-20]),c='black',label='spectre observe sans raies')
    plt.title('Spectre observe final',fontsize=26)
    plt.grid(True)
    plt.xlabel('$\lambda$ (nm)',fontsize=24)
    plt.ylabel('flux (erg/cm^2/nm)',fontsize=24)
    plt.gca().get_xaxis().set_tick_params(labelsize=20)
    plt.gca().get_yaxis().set_tick_params(labelsize=20)
    plt.text(750,max(intensite_obss)*0.85,'d)',fontsize=40)
    plt.legend(prop={'size':22},loc='lower right')
    fig.tight_layout()
    #plt.savefig('/Users/bremaud/Documents/raies5.pdf')

    plt.figure(figsize=[14,10])

    plt.axis([300,1100,0,max(intensite_obs)*1.1])
    plt.plot(lambda_obs,intensite_obs,' .')
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('u.a')
    plt.title('spectre observe')
    plt.grid(True)

    plt.figure(figsize=[14,10])

    plt.axis([300,1100,0,max(intensite_obs)*1.1])
    plt.plot(lambda_obs,intensite_obss,' .')
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('u.a')
    plt.title('spectre observe lisse savgol')
    plt.grid(True)

    plt.figure(figsize=[14,10])

    plt.axis([300,1100,min(D_intensite_obss)*1.1,max(D_intensite_obss)*1.1])
    plt.plot(lambda_obs,D_intensite_obss,' .')
    plt.plot(lambda_coupe_obs,D_intensite_coupe,' .',c='r')
    plt.title('derivee du spectre observe lisse')
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('u.a')
    plt.title('derivee du spectre observe lisse')
    plt.grid(True)

    plt.figure(figsize=[14,10])

    plt.axis([300,1100,0,max(intensite_obs)*1.1])
    plt.plot(lambda_coupe_obs,intensite_coupe_obs,' .')
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('u.a')
    plt.title('spectre observe sans raies')
    plt.grid(True)

    plt.figure(figsize=[14,10])

    plt.axis([300,1100,0,max(intensite_obs)*1.1])
    plt.plot(lambda_obs,intensite_obs,c='r')
    plt.plot(lambda_obs[20:len(lambda_obs)-20],intensite_obsSpline(lambda_obs[20:len(lambda_obs)-20]),c='black')
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('u.a')
    plt.title('spectre observe final')
    plt.grid(True)


    plt.show()

#CHECK

## Evaluation de la transmission instrumentale


def Plot_magabsorbe_bin(NAME,DEBUG,method,magabs,magabs_err,airmass,s,window,sigma,Bin,coeff_sigmaO,demi_taille_maxO,deltaO):

    """Fonction: effectue la division en bin de longueur d'onde du spectre observe et le calcul de l'integrale
    sur ces derniers après avoir effectue une convolution pour le spectre observe (provenant des donnees).
    Rempli les tableaux de magnitude et de masse d'air (methode).

    Entrees:
        magabs: matrice A x B avec A le nombre de bin de longueur d'onde et B le nombre de photo utilise contenant
        la magnitude du spectre observe par bin de longueur d'onde pour chaque photo.
        magabs_err: matrice A x B des incertitudes type associee à magabs.
        airmass: matrice A x B avec A le nombre de bin de longueur d'onde et B le nombre de photo utilise contenant
        la masse d'air du spectre observe par bin de longueur d'onde pour chaque photo.
        s: type spectractor.spectrum contenant l'ensemble des donnees pour une photo.
        window: type de fenêtre pour la convolution des spectres observe et tabule.
        width: taille de la fênetre pour la convolution.
        sigma: ecart type si la fenêtre utilisee est une 'gaussian'.
        Bin: tableau contenant les longueurs d'ondes divise en bin.

    Sortie:
        rempli les tableaux magabs, magabs_err, airmass. (methode)"""

    "importation des donnees"
    intensite_obs=[]
    lambda_obs=[]
    intensite_err=[]
    VouF2=True
    for line in s:
        if VouF2:
            a=line.split()
            Airmass=float(a[3])
            VouF2=False
        else:
            a=line.split()
            if len(a)>4:
                lambda_obs.append(float(a[2]))
                intensite_obs.append(float(a[3]))
                intensite_err.append(float(a[4]))

    if lambda_obs[len(lambda_obs)-1]>Bin[len(Bin)-1] and lambda_obs[0]<Bin[0]:
        """si la plage de longueur d'onde du spectre observe est plus etroite que les valeurs max et min de la division
        en bin on ne prend pas en compte la photo et on passe à la suivante"""

        if method=='raies':
            intensite_obss=sp.signal.savgol_filter(intensite_obs, 17, 3)

            D_intensite_obss=[]
            Raies=[]

            for i in range(len(intensite_obss)-1):
                D_intensite_obss.append((intensite_obss[i+1]-intensite_obss[i])/(lambda_obs[i+1]-lambda_obs[i]))

            D_intensite_obss.append(0)

            D_mean=[]
            D_sigma=[]
            Raies=[]

            for i in range(10):
                Raies.append(False)
                D_mean.append(np.mean(intensite_obss[:10]))
                D_sigma.append(np.std(intensite_obss[:10]))

            i=10
            while i<len(D_intensite_obss)-1:
                moy=[]
                j=1
                Raies.append(False)
                while len(moy)!=10:
                    if Raies[i-j]==False:
                        moy.append(D_intensite_obss[i-j])
                    j+=1
                D_mean.append(np.mean(moy))
                D_sigma.append(np.std(moy))

                if D_intensite_obss[i]<D_mean[len(D_mean)-1]-coeff_sigmaO*D_sigma[len(D_sigma)-1]:
                    k=i
                    while lambda_obs[k]-lambda_obs[i]<demi_taille_maxO and k<len(lambda_obs)-1:
                        k+=1
                    for j in range(i,k):
                        if D_intensite_obss[j]>D_mean[len(D_mean)-1]+coeff_sigmaO*D_sigma[len(D_sigma)-1]:
                            if j+k-i<=len(lambda_obs):
                                indice=j+k-i
                            else:
                                indice=len(lambda_obs)
                            for v in range(j,indice):
                                if D_intensite_obss[v]<D_mean[len(D_mean)-1]+coeff_sigmaO*D_sigma[len(D_sigma)-1]:
                                    indice=v
                                    break

                            if indice!=j+k-i:
                                if indice+deltaO>len(lambda_obs):
                                    end=len(lambda_obs)
                                else:
                                    end=indice+deltaO
                                for loop in range(i+1,end):
                                    Raies.append(True)
                                for loop in range(i-deltaO,i+1):
                                    Raies[i]=True
                                i=end
                                Raies.append(False)
                            break
                i+=1

            intensite_coupe_obs=[]
            lambda_coupe_obs=[]
            D_intensite_coupe=[]
            if len(intensite_obss)<len(Raies):
                for i in range(len(Raies)-len(intensite_obss)):
                    Raies.pop()

            for i in range(len(Raies)):
                if Raies[i]==False:
                    intensite_coupe_obs.append(intensite_obss[i])
                    lambda_coupe_obs.append(lambda_obs[i])
                    D_intensite_coupe.append(D_intensite_obss[i])
                    endfalse=i
            interpolation_obs=sp.interpolate.interp1d(lambda_coupe_obs,intensite_coupe_obs)


            if DEBUG:
                plt.figure(figsize=[6,6])
                plt.axis([300,1100,0,max(intensite_obs)*1.1])
                plt.plot(lambda_obs,intensite_obs,c='r')
                plt.plot(lambda_obs[1:endfalse],interpolation_obs(lambda_obs[1:endfalse]),c='black')
                plt.xlabel('$\lambda$ (nm)')
                plt.ylabel('u.a')
                plt.title('spectre observe sans raies')
                plt.grid(True)
                plt.show()

        elif method=='convoluate':

            intensite_obss=smooth(intensite_obs,6*sigma,window,sigma)
            "convolution par une fenêtre (en general gaussienne)"

            interpolation_obs=interp1d(lambda_obs, intensite_obss)
            "fonction d'interpolation du spectre observe convolue"

        fluxlum_Binobs=np.zeros(len(Bin)-1)
        fluxlumBin_err=np.zeros(len(Bin)-1)
        "tableaux de taille len(Bin)-1 et comportant les intensites par bin"
        "tableaux de taille len(Bin)-1 et comportant les erreurs d'intensites par bin"

        for v in range(len(Bin)-1):
            "On rempli les tableaux par bin de longueur d'onde"
            X=np.linspace(Bin[v],Bin[v+1],1000)
            Y=interpolation_obs(X)
            fluxlum_Binobs[v]=integrate.simps(Y,X,dx=1)/(Bin[v+1]-Bin[v])
            """Integration sur un bin de longueur d'onde avec 1000 points, on considèrera ensuite cette valeur comme
            valeur de l'intensite pour la longueur d'onde moyenne du bin"""


            "Determination des indices juste à droite de la frontière du bin_i pour les longueurs d'ondes observees"
            jmin=0
            while lambda_obs[jmin]<Bin[v]:
                jmin+=1

            jmax=jmin
            while lambda_obs[jmax]<Bin[v+1]:
                jmax+=1

            S=0
            "Propagation des incertitudes sur les intensites par bin, calcul sur les bords cf feuille"
            for j in range(jmin,jmax):
                if j==jmin:
                    S+=(intensite_err[j]*((lambda_obs[j]-Bin[v])/2
                    +(Bin[v]-lambda_obs[j-1])*(lambda_obs[j]-Bin[v])/(lambda_obs[j]-lambda_obs[j-1])/2)
                    +intensite_err[j]*(lambda_obs[j+1]-lambda_obs[j])/2)**2
                    +(intensite_err[j-1]*((lambda_obs[j]-Bin[v])/2
                    +(Bin[v]-lambda_obs[j-1])*(lambda_obs[j]-Bin[v])/(lambda_obs[j]-lambda_obs[j-1])/2))**2

                if j==jmax-1:
                    S+=(intensite_err[j]*((Bin[v+1]-lambda_obs[j])/2
                    +(lambda_obs[j+1]-Bin[v+1])*(Bin[v+1]-lambda_obs[j])/(lambda_obs[j+1]-lambda_obs[j])/2)
                    +intensite_err[j]*(lambda_obs[j]-lambda_obs[j-1])/2)**2
                    +(intensite_err[j+1]*((Bin[v+1]-lambda_obs[j])/2
                    +(lambda_obs[j+1]-Bin[v+1])*(Bin[v+1]-lambda_obs[j])/(lambda_obs[j+1]-lambda_obs[j])/2))**2

                else:
                    S+=(intensite_err[j]*(lambda_obs[j+1]-lambda_obs[j-1])/2)**2

            fluxlumBin_err[v]=np.sqrt(S)/(Bin[v+1]-Bin[v])

        "Remplissage des tableaux de magnitude absorbee et de masse d'air"
        airmassi=Airmass


        for v in range(len(fluxlum_Binobs)):
            "On rempli les 3 tableaux en iterant sur les differents bin de longueur d'onde"
            magabs[v].append(np.log(fluxlum_Binobs[v])) #passage en magnitude avec le log
            airmass[v].append(airmassi)
            magabs_err[v].append(fluxlumBin_err[v]/fluxlum_Binobs[v])

    else:
        print("----> image non utilisee à cause de la plage de longueur d'onde"+'\n')
        NAME.pop()
#CHECK

def Plot_magabsorbe_star(DEBUG,method,sim,fileday,star,disperseur,window,sigma,binwidth,lambda_min,lambda_max,coeff_sigmaO,demi_taille_maxO,deltaO,coeff_sigma,demi_taille_max,delta):

    """Fonction: renvoie les tableaux magabas, airmass, fluxlumBin_reel, Bin.

    Entree:
        sim: True si il s'agit d'une simulation, False sinon.
        fileday: chemin où se trouve le fichier contenant les spectres à etudier.
        star: nom de l'etoile à etudier.
        disperseur: nom du disperseur à etudier (Ron400,Thor300,HoloPhP,HoloPhAg,HoloAmAg)
        binwidth: taille des bin de longueur d'onde.
        lambda_min: longueur d'onde minimum utilisee lors de la division en bin.
        lambda_max: longueur d'onde maximum utilisee lors de la division en bin.
        idem que precedemment.

    Sortie:
        Tableaux associees ainsi que le tableau Bin et celui des incertitudes"""

    Bin=np.arange(lambda_min,lambda_max+binwidth,binwidth)
    "Division de la plage spectrale en bin de longueur d'onde de longueur binwidth"

    if sim:
        list_spectrums=glob.glob(fileday+"/sim*spectrum*.txt") #depend de la prod
        """liste des chemins contenu dans le fichier du jour à etudier en cherchant uniquement ceux qui proviennent
        des simulations du CTIO et qui sont des spectres"""

    else:
        list_spectrums=glob.glob(fileday+"/reduc*spectrum*.txt") #depend de la prod
        """liste des chemins contenu dans le fichier du jour à etudier en cherchant uniquement ceux qui proviennent
        des mesures du CTIO et qui sont des spectres"""

    magabs=[]
    airmass=[]
    magabs_err=[]
    "Constitution de liste vide de bonne taille c'est à dire correspondant au nombre de bin de longueurs d'ondes"
    for j in range(len(Bin)-1):
        magabs.append([])
        airmass.append([])
        magabs_err.append([])

    "spec est un compteur parcourant la liste des spectres du fichier du jour, on l'initialise au debut"
    spec=0
    NAME=[]
    """VouF: Vrai ou Faux pour que les calculs concernant le spectre tabule ne soit fait q'une seule fois. En effet
    le spectre tabule est le même puisqu'on ne considère qu'une seule etoile"""
    VouF=True

    while spec<int(len(list_spectrums)):

        s=open(list_spectrums[spec],'r')

        for line in s:
            a=line.split()
            print(a)
            Star=a[1]
            Disperseur=a[2]
            Airmass=float(a[3])
            break

        if VouF and Star==star: #uniquement lors du premier passage
            intensite_reel=[]
            lambda_reel=[]
            for line in s:
                a=line.split()
                intensite_reel.append(float(a[1]))
                lambda_reel.append(float(a[0]))

            if method=='raies':

                intensite_reels=smooth(intensite_reel,6*sigma,'gaussian',sigma) #convolution par une gaussienne
                intensite_reelss=sp.signal.savgol_filter(intensite_reels, 7, 3) #filtre savgol (enlève le bruit)

                D_intensite_reelss=[]
                Raies=[]

                for i in range(len(intensite_reelss)-1):
                    D_intensite_reelss.append((intensite_reelss[i+1]-intensite_reelss[i])/(lambda_reel[i+1]-lambda_reel[i]))

                D_intensite_reelss.append(0)

                D_mean=[]
                D_sigma=[]
                Raies=[]

                for i in range(10):
                    Raies.append(False)
                    D_mean.append(np.mean(intensite_reelss[:10]))
                    D_sigma.append(np.std(intensite_reelss[:10]))

                i=10
                while i<len(D_intensite_reelss)-1:
                    moy=[]
                    j=1
                    Raies.append(False)
                    while len(moy)!=10:
                        if Raies[i-j]==False:
                            moy.append(D_intensite_reelss[i-j])
                        j+=1
                    D_mean.append(np.mean(moy))
                    D_sigma.append(np.std(moy))

                    if D_intensite_reelss[i]<D_mean[len(D_mean)-1]-coeff_sigma*D_sigma[len(D_sigma)-1]:
                        k=i
                        while lambda_reel[k]-lambda_reel[i]<demi_taille_max:
                            k+=1
                        for j in range(i,k):
                            if D_intensite_reelss[j]>D_mean[len(D_mean)-1]+coeff_sigma*D_sigma[len(D_sigma)-1]:
                                indice=j+k-i
                                for v in range(j,j+k-i):
                                    if D_intensite_reelss[v]<D_mean[len(D_mean)-1]+2*D_sigma[len(D_sigma)-1]:
                                        indice=v
                                        break

                                if indice!=j+k-i:
                                    for loop in range(i+1,indice+delta):
                                        Raies.append(True)
                                    for loop in range(i-delta,i+1):
                                        Raies[i]=True
                                    i=indice+delta
                                    Raies.append(False)
                                break
                    i+=1

                intensite_coupe=[]
                lambda_coupe=[]
                D_intensiteR_coupe=[]

                for i in range(len(Raies)):
                    if Raies[i]==False:
                        D_intensiteR_coupe.append(D_intensite_reelss[i])
                        intensite_coupe.append(intensite_reelss[i])
                        lambda_coupe.append(lambda_reel[i])

                interpolation_reel=sp.interpolate.interp1d(lambda_coupe,intensite_coupe)

                if DEBUG:
                    plt.figure(figsize=[6,6])
                    plt.axis([300,1100,0,max(intensite_reelss)*1.1])
                    plt.plot(lambda_reel,intensite_reel,c='r')
                    plt.plot(lambda_reel[5:-5],interpolation_reel(lambda_reel[5:-5]),c='black')
                    plt.xlabel('$\lambda$ (nm)')
                    plt.ylabel('u.a')
                    plt.title('spectre reel final')
                    plt.grid(True)
                    plt.show()

            elif method=='convoluate':

                intensite_reels=smooth(intensite_reel,6*sigma,'gaussian',sigma)

                interpolation_reel=interp1d(lambda_reel, intensite_reels)
                "fonction d'interpolation du spectre tabule convolue"

            fluxlum_Binreel=np.zeros(len(Bin)-1)
            "tableaux de taille len(Bin)-1 et comportant les intensites tabulees par bin"

            for v in range(len(Bin)-1):
                "On rempli les tableaux par bin de longueur d'onde"
                X=np.linspace(Bin[v],Bin[v+1],1000)
                Y=interpolation_reel(X)
                fluxlum_Binreel[v]=integrate.simps(Y,X,dx=1)/(Bin[v+1]-Bin[v])
                """Integration sur un bin de longueur d'onde avec 1000 points, on considèrera ensuite cette valeur
                comme valeur de l'intensite tabulee pour la longueur d'onde moyenne du bin"""

            VouF=False

        if Star==star and Disperseur==disperseur:
            name_data=os.path.split(list_spectrums[spec])[1]
            NAME.append(name_data)
            print(name_data)
            s=open(list_spectrums[spec],'r')
            Plot_magabsorbe_bin(NAME,DEBUG,method,magabs,magabs_err,airmass,s,window,sigma,Bin,coeff_sigmaO,demi_taille_maxO,deltaO)

        spec+=1
        s.close()


    """On elimine les spectres qui donnent des resultats absurdes, c'est à dire très different de la valeurs moyenne
    des autres magnitudes. Arbitrairement on fixe cette difference à 1 en moyenne sur l'ensemble des bins"""

    def f(x,a,b):
        return a*x+b

    L=[0]
    while L!=[]:
        L=[] #Contient la liste des indices des spectres problematiques
        for i in range(len(magabs[0])): #test de chaque spectre etudie
            S=0
            for j in range(len(magabs)): #on effectue une moyenne de la difference sur chaque bin
                S=abs(np.mean(magabs[j])-magabs[j][i])
                if sim and S>1:
                    L.append(i)
                    break
                elif sim==False and S>1:
                    L.append(i)
                    break

        if L==[]:
            for i in range(len(magabs[0])): #test de chaque spectre etudie
                S=0
                for j in range(len(magabs)): #on effectue une moyenne de la difference sur chaque bin
                    popt, pcov=sp.optimize.curve_fit(f,airmass[j],magabs[j],sigma=magabs_err[j])
                    residu_m=abs(magabs[j][i]-f(airmass[j][i],*popt))
                    S=residu_m
                    if sim:
                        if S>3*np.std(magabs[j]):
                            L.append(i)
                            break

                    else:
                        if S>3*np.std(magabs[j]):
                            L.append(i)
                            break


        "Suppression des 3 listes des spectres problematiques"
        for i in range(len(L)):
            print('\n')
            print(NAME[L[i]])
            print("----> image non utilisee à cause d'un flux en intensite anormal"+'\n')
            for j in range(len(magabs)):
                magabs[j].pop(L[i]-i)
                airmass[j].pop(L[i]-i)
                magabs_err[j].pop(L[i]-i)

    return(airmass,magabs,magabs_err,Bin,fluxlum_Binreel)
#CHECK


def droites_Bouguer(DEBUG,method,sim,fileday,star,disperseur,window,sigma,binwidth,lambda_min,lambda_max,coeff_sigmaO,demi_taille_maxO,deltaO,coeff_sigma,demi_taille_max,delta):
    """Fonction: effectue pour toute les photos d'une même etoile, le trace de magabs par bin de longueur d'onde

    Entree:
        idem que precedemment.

    Sortie:
        Trace des droites de Bouguer.
        Trace de la magnitude absorbee en fonction de la masse d'air.
        Tableau des coefficients ordonnees à l'origine et pente (il s'agit d'une matrice de taille nombre de bin x 2)
        Tableaux des bins, des incertitudes, et de l'intensite par bin tabulee"""

    "Recuperation des tableaux renvoyes par Plot_magabsorbe_star"
    airmass, magabs, magabs_err, Bin, fluxlum_Binreel=Plot_magabsorbe_star(DEBUG,method,sim,fileday,star,disperseur,window,sigma,binwidth,lambda_min,lambda_max,coeff_sigmaO,demi_taille_maxO,deltaO,coeff_sigma,demi_taille_max,delta)

    "Definition de la fonction lineaire qu'on va utiliser pour tracer les droites de Bouguer."
    def f(x,a,b):
        return a*x+b

    Z=np.linspace(0,2.2,1000) #nombre de points pour le trace des droites

    #indice j du debut de l'ordre 2 environ.
    for i in range(len(Bin)):
        if Bin[i]>750:
            j=i
            break

    coeff=np.zeros((len(magabs),2)) #on initialise la liste des coefficients à la liste vide.
    err=np.zeros(len(magabs)) #on initialise la liste des erreurs sur les ordonnees à l'origine à la liste vide.
    coeffbis=np.zeros((len(magabs)-j,2)) #coefficient pour le fit.


    fig=plt.figure(figsize=[15, 10])

    new_lambda=0.5*(Bin[1:]+Bin[:-1])

    #date=os.path.split(fileday)[1]
    "date: chaine de caractère correspondant à la date du fichier etudie à partir de l'emplacement du fichier"

    "On trace les droites et on recupère l'ordonnee à l'origine pour chaque bin de longueur d'onde"
    for i in range(len(magabs)):
        popt, pcov=sp.optimize.curve_fit(f,airmass[i],magabs[i],sigma=magabs_err[i]) #fit propageant les incertitudes
        "On recupère les coefficients et les incertitudes types sur l'ordonnee à l'origine"
        coeff[i][0],coeff[i][1]=popt[0],popt[1]
        err[i]=np.sqrt(pcov[1][1])

        "Affichage"
        MAG=f(Z, *popt)
        MAG_sup=f(Z,popt[0]+np.sqrt(pcov[0][0]),popt[1]-np.sqrt(pcov[1][1]))
        MAG_inf=f(Z,popt[0]-np.sqrt(pcov[0][0]),popt[1]+np.sqrt(pcov[1][1]))
        #plt.plot(Z,MAG,c=wavelength_to_rgb(new_lambda[i]))
        plt.plot(Z,MAG,c='black')

        # Attention à modifier
        #plt.plot(Z,MAG_sup,c=wavelength_to_rgb(new_lambda[i]),linestyle=':')
        #plt.plot(Z,MAG_inf,c=wavelength_to_rgb(new_lambda[i]),linestyle=':')
        plt.plot(Z,MAG_sup,c='black',linestyle=':')
        plt.plot(Z,MAG_inf,c='black',linestyle=':')
        #plt.fill_between(Z,MAG_sup,MAG_inf,color=[wavelength_to_rgb(new_lambda[i])])
        "la commande ci-dessus grise donne la bonne couleur la zone où se trouve la droite de Bouguer"

        #plt.scatter(airmass[i],magabs[i],c=[wavelength_to_rgb(new_lambda[i])],
            #label=f'{Bin[i]}-{Bin[i+1]} nm', marker='o',s=30)

        plt.scatter(airmass[i],magabs[i],c='black',
            label=f'{Bin[i]}-{Bin[i+1]} nm', marker='o',s=30)
        #plt.errorbar(airmass[i],magabs[i], xerr=None, yerr = magabs_err[i],fmt = 'none',
            #capsize = 1, ecolor=(wavelength_to_rgb(new_lambda[i])), zorder = 2,elinewidth = 2)
        plt.errorbar(airmass[i],magabs[i], xerr=None, yerr = magabs_err[i],fmt = 'none',
            capsize = 1, ecolor=('black'), zorder = 2,elinewidth = 2)
    #Interpolation des paramètres des droites de bouguer:

    a_lambda=interp1d(new_lambda,coeff.T[0])
    b_lambda=interp1d(new_lambda,coeff.T[1])

    Lambda=np.arange(new_lambda[0],new_lambda[len(new_lambda)-1],1)
    coeffbis=coeff #à supprimer

    """
    #apparte concernant l'atmosphère (et celle trouvee avec les droites de Bouguer)

    atmgrid = AtmosphereGrid(filename="/Users/bremaud/CTIODataJune2017_reduced_RG715_v2_prod4/data_30may17_A2=0/reduc_20170530_135_atmsim.fits")

    a = plot_atmo(300,4,0.05,plot=False) #fonction de l'atmosphère lissee
    b = atmgrid.simulate(300,4,0.05) #fonction de l'atmosphère non lissee

    #Normalisation
    for i in range(len(new_lambda)):
        if new_lambda[i]>751:
            coeff[-1][0]=np.log(np.exp(coeff[i][0]*1.137)/(a(new_lambda[i])/max(a(new_lambda))))/1.137
            break

    #On force la pente des droites de Bouguer à valoir la pente theorique
    for i in range(len(new_lambda)):
        if new_lambda[i]>751:
            coeff.T[0][i]=np.log(np.max(np.exp(coeff.T[0]*1.137))*a(new_lambda[i])/max(a(Lambda)))/1.137

    #Nouvelle fonction d'interpolation avec les nouvelles pentes.
    a_lambdabis=interp1d(new_lambda,coeff.T[0])

    #fit de l'ordre 2

    alpha=np.zeros(len(magabs)-j+1)
    for i in range(j,len(magabs)):
        a_2=a_lambdabis(Bin[i]/2)
        b_2=b_lambda(Bin[i]/2)
        a_1=coeff[i][0]
        def g(x,b_1,alpha):
            return a_1*x+b_1+np.log(1+alpha*np.exp((a_2-a_1)*x+b_2-b_1))

        popt, pcov=sp.optimize.curve_fit(g,airmass[i],magabs[i],p0=(coeff[i][1]-0.3,80e-2),bounds=([coeff[i][1]-0.9,0],[coeff[i][1],100e-2]))
        #Attention aux limites de alpha

        coeffbis[i-j]=[coeff[i][0],popt[0]]
        alpha[i-j]=popt[1]
        MAG=g(Z, *popt)
        #plt.plot(Z,MAG,c='black',lw=2) (optionnel pour afficher le nouveau fit mais ça n'apporte pas grand chose)

    "Determination des limites du graphique"
    Min = np.min(magabs)
    Max = np.max(magabs)

    plt.xlabel("airmass",fontsize=24)
    plt.ylabel('ln(flux)',fontsize=24)
    plt.title('ln(flux) par bin: '30may', '+star+', '+disperseur+', convolution, $\sigma$='+str(sigma)+', binwidth='+str(binwidth),fontsize=22)
    plt.axis([0,2.2,Min-0.2,Max+0.3])
    plt.gca().get_xaxis().set_tick_params(labelsize=18)
    plt.gca().get_yaxis().set_tick_params(labelsize=18)
    plt.legend(prop={'size':12},loc='right')
    plt.grid(True)
    fig.tight_layout()
    #plt.savefig('/Users/bremaud/Documents/rep_instru/simulations/'+disperseur+'/'+method+'/binwidth='+str(binwidth)+'/sigma='+str(sigma)+'/image/droite_bouguer1.pdf')
    plt.show()
    #fin de la figure des droites de Bouguer.

    #debut figure atmosphère.

    fig = plt.figure(figsize=[15, 10])

    a = plot_atmo(300,4,0.05,plot=False)
    b = atmgrid.simulate(300,4,0.05)
    c= atmgrid.simulate(300,4,0)
    d=atmgrid.simulate(300,4,0.1)
    #plt.plot(Lambda, a(Lambda)/max(a(Lambda)),c='black',lw=2,label='atmosphère sans raies')
    #plt.plot(Lambda, b(Lambda)/max(b(Lambda)),c='red',lw=2,label='atmosphère en entree de la simu')
    plt.plot(Lambda, b(Lambda)/max(b(Lambda)),c='red',label='VAOD = 0.05',lw=5)
    plt.plot(Lambda, c(Lambda)/max(c(Lambda)),c='blue',linestyle='dotted',label='VAOD = 0',lw=5)
    plt.plot(Lambda, d(Lambda)/max(d(Lambda)),c='black',linestyle='dotted',label='VAOD = 0.1',lw=5)
    #plt.plot(Lambda,np.exp(a_lambdabis(Lambda)*1.137)/np.max(np.exp(a_lambdabis(Lambda)*1.137)),c='blue',lw=2,label='atmosphère en sortie après modification')
    #plt.plot(Lambda,np.exp(a_lambda(Lambda)*1.137)/np.max(np.exp(a_lambda(Lambda)*1.137)),c='black',lw=2,label='atmosphère en sortie de la simu')

    plt.xlabel('$\lambda$ (nm)',fontsize=28)
    plt.ylabel('Transmission atmospherique normalisee',fontsize=28)
    plt.gca().get_xaxis().set_tick_params(labelsize=22)
    plt.gca().get_yaxis().set_tick_params(labelsize=22)
    plt.title("Dependance du modèle d'atmosphère aux aerosols",fontsize=28) #à changer selon l'affichage souhaite
    plt.axis([Bin[0],Bin[len(Bin)-1],0.2,1.1])
    plt.legend(prop={'size':26},loc='lower left')
    plt.grid(True)
    #plt.savefig('/Users/bremaud/Documents/rep_instru/simulations/'+disperseur+'/'+method+'/binwidth='+str(binwidth)+'/sigma='+str(sigma)+'/coeff_sigmaO='+str(coeff_sigmaO)+'/demi_taille_maxO='+str(demi_taille_maxO)+'/deltaO='+str(deltaO)+'/coeff_sigma='+str(coeff_sigma)+'/demi_taille_max='+str(demi_taille_max)+'/delta='+str(delta)+'/image/transmission_atmo4.pdf')
    plt.show()

    #Residus aux droites de Bouguer.

    def f(x,a,b):
        return a*x+b
    fig = plt.figure(figsize=[25, 25])
    residu_m=np.zeros((len(magabs),len(magabs[0])))
    for i in range(len(magabs)):
        residu_m[i]=np.array(magabs[i])-f(np.array(airmass[i]),coeff[i][0],coeff[i][1])
        plt.plot(airmass[i],residu_m[i],marker='.',c=(wavelength_to_rgb(new_lambda[i])))

    plt.axis([min(airmass[i]),max(airmass[i]),np.min(residu_m),np.max(residu_m)])
    plt.gca().get_xaxis().set_tick_params(labelsize=18)
    plt.gca().get_yaxis().set_tick_params(labelsize=18)
    plt.xlabel('airmass',fontsize=24)
    plt.title('Residus aux droites: '+date+', '+star+', '+disperseur+', '+method+', $\sigma$='+str(sigma)+', binwidth='+str(binwidth),fontsize=24)
    plt.ylabel('magnitude',fontsize=24)
    #plt.savefig('/Users/bremaud/Documents/rep_instru/simulations/'+disperseur+'/'+method+'/binwidth='+str(binwidth)+'/sigma='+str(sigma)+'/coeff_sigmaO='+str(coeff_sigmaO)+'/demi_taille_maxO='+str(demi_taille_maxO)+'/deltaO='+str(deltaO)+'/coeff_sigma='+str(coeff_sigma)+'/demi_taille_max='+str(demi_taille_max)+'/delta='+str(delta)+'/image/residus.pdf')
    plt.show()
    """
    return (coeff,coeffbis,Bin,err,fluxlum_Binreel)
#CHECK


def reponse_instrumentale(method,sim,fileday,star,disperseur,DEBUG=False,window='gaussian',sigma=5,binwidth=20,lambda_min=370,lambda_max=1030,coeff_sigmaO=1.5,demi_taille_maxO=40,deltaO=4,coeff_sigma=2.5,demi_taille_max=30,delta=12):
    """Fonction: trace la reponse instrumentale obtenue ainsi que celle de Sylvie, Augustion et Nick et enregistre les
    donnees dans des fichiers pdf (pour la reponse instrumentale) et txt cree à l'execution.
    Si sim: uniquement la reponse de Sylvie.

    Entree:
        idem que precedemment avec ici des valeurs par defauts des paramètres modifiables à tester.

    Sortie:
        Trace de la reponse instrumentale.
        Trace de celle de Sylvie, Nick et Augustin si il s'agit du disperseur Ronchie400.
        Enregistre les donnees dans un chemin et avec un nom lies aux paramètres d'entree de la fonction.

    Si sim:
        Trace de la reponse instrumentale.
        Trace de celle de Sylvie si il s'agit du disperseur Ronchie400.
        Trace de la correlation entre les deux reponses instrumentales.
        Trace de l'ecart relatif entre la reponse instrumentale obtenue et celle des donnees.
        Trace de la reponse instrumentale ainsi que de celle de Sylvie si il s'agit du disperseur Ronchie400.
        Enregistre les donnees dans un chemin et avec un nom lies aux paramètres d'entree de la fonction."""


    "Recuperation des tableaux renvoyes par droites_Bouguer"
    coeff, coeffbis, Bin, err, fluxlum_Binreel=droites_Bouguer(DEBUG,method,sim,fileday,star,disperseur,window,sigma,binwidth,lambda_min,lambda_max,coeff_sigmaO,demi_taille_maxO,deltaO,coeff_sigma,demi_taille_max,delta)

    """On calcul les tableaux rep_instru correspondant à la reponse instrumentale, new_err l'erreur associee et
    new_lambda la longueur d'onde associe à la reponse instrumentale."""
    rep_instru=np.exp(coeff.T[1])/fluxlum_Binreel
    new_err=err*np.exp(coeff.T[1])/fluxlum_Binreel

    new_lambda=0.5*(Bin[1:]+Bin[:-1])

    #"bis" correspond aux valeurs avec le fit.
    rep_instru_bis=[np.exp(coeffbis[i][1])/fluxlum_Binreel[i+len(coeff)-len(coeffbis)] for i in range(len(coeffbis))]
    new_lambda_bis=[(Bin[len(coeff)-len(coeffbis)+i+1]+Bin[len(coeff)-len(coeffbis)+i])/2 for i in range(len(coeffbis))]

    Max=np.max(rep_instru)

    #date=os.path.split(fileday)[1]

    "Affichage"
    if sim:
        gs_kw = dict(height_ratios=[4,1], width_ratios=[1])
        fig, ax = plt.subplots(2,1,sharex="all",figsize=[13, 11], constrained_layout=True, gridspec_kw=gs_kw)
        ax[0].scatter(new_lambda,rep_instru/Max,c='black', marker='o',label='T_inst Vincent',zorder=2,s=35)
        ax[0].errorbar(new_lambda,rep_instru/Max, xerr=None, yerr = new_err/max(rep_instru),fmt = 'none', capsize = 1, ecolor = 'black', zorder = 2,elinewidth = 2)

        #Affichage du fit:
        ax[0].scatter(new_lambda_bis,rep_instru_bis/Max,c='red', marker='o',label='fit ordre 2',zorder=2,s=30)

    else:
        #Affichage rep instru comparees + atmosphère

        fig=plt.figure(figsize=[15,10])
        ax2 = fig.add_subplot(111)
        #ax2 = ax1.twinx()
        ax2.annotate("", xy = (950, 0.9), xytext = (950, 1.1),
                arrowprops = {'facecolor': 'black', 'edgecolor': 'black',
                              'width': 6, 'headwidth': 15
                              }, color = 'black')
        ax2.annotate("", xy = (905, 0.99), xytext = (905, 1.1),
                arrowprops = {'facecolor': 'black', 'edgecolor': 'black',
                              'width': 6, 'headwidth': 15
                              }, color = 'black')

        ax2.text(890,1.12,"raies de l'eau",color='black',fontsize=20)

        ax2.axvline(385,0.57,1.18,linestyle='dotted',c='black')
        ax2.axvline(490,0.70,1.18,linestyle='dotted',c='black')

        ax2.text(410,1.12,"aerosols",color='black',fontsize=20)

        ax2.axvline(650,0.75,1.18,linestyle='dotted',c='black')
        ax2.axvline(515,0.72,1.18,linestyle='dotted',c='black')

        ax2.text(565,1.12,"ozone",color='black',fontsize=20)


        ax2.annotate("", xy = (764, 0.97), xytext = (764, 1.1),
                arrowprops = {'facecolor': 'black', 'edgecolor': 'black',
                              'width': 6, 'headwidth': 10
                              }, color = 'black')

        ax2.text(710,1.12,"raie du dioxygène",color='black',fontsize=20)

        ax2.annotate("", xy= (385, 1.13), xytext = (400, 1.13),
                arrowprops = {'facecolor': 'black', 'edgecolor': 'black',
                              'width': 1, 'headwidth': 10
                              }, color = 'black')

        ax2.annotate("", xy= (490, 1.13), xytext  = (475, 1.13),
                arrowprops = {'facecolor': 'black', 'edgecolor': 'black',
                             'width': 1, 'headwidth': 10
                              }, color = 'black')

        ax2.annotate("", xy = (515, 1.13), xytext = (555, 1.13),
                arrowprops = {'facecolor': 'black', 'edgecolor': 'black',
                             'width': 1, 'headwidth': 10
                              }, color = 'black')

        ax2.annotate("", xy= (650, 1.13), xytext  = (610, 1.13),
                arrowprops = {'facecolor': 'black', 'edgecolor': 'black',
                              'width': 1, 'headwidth': 10
                              }, color = 'black')


        #ax1.scatter(new_lambda,rep_instru/Max,c='black', marker='o',label='T_inst Vincent',zorder=2,s=35)
        #ax1.errorbar(new_lambda,rep_instru/Max, xerr=None, yerr = new_err/max(rep_instru),fmt = 'none', capsize = 1, ecolor = 'black', zorder = 2,elinewidth = 2)
        """
        Lambda=np.arange(new_lambda[0],new_lambda[len(new_lambda)-1],1)
        atmgrid = AtmosphereGrid(filename="/Users/bremaud/CTIODataJune2017_reduced_RG715_v2_prod4/data_30may17_A2=0/reduc_20170530_135_atmsim.fits")
        b = atmgrid.simulate(300,5,0.05)
        ax2.plot(Lambda, b(Lambda)/max(b(Lambda)),color='blue')
        """

    "Si le disperseur correspond au Ronchi400 on compare avec les autres reponses instrumentales"

    if disperseur=='Ron400':
        if sim:
            x='ctio_throughput_300517_v1.txt'

            a=np.loadtxt(x)
            ax[0].scatter(a.T[0],a.T[1]/max(a.T[1]),c='deepskyblue', marker='.',label='T_inst exacte')
            ax[0].errorbar(a.T[0],a.T[1]/max(a.T[1]), xerr=None, yerr = a.T[2]/max(a.T[1]),fmt = 'none', capsize = 1, ecolor = 'deepskyblue', zorder = 1,elinewidth = 2)

        else:
            #à rajouter si on souhaite comparer les reponses de tout le monde

            x='/Users/bremaud/Documents/rep instru CTIO S-A-N/T_throughput/t_tel_r400.list'
            a=np.loadtxt(x)
            ax1.scatter(a.T[0],a.T[1],c='r', marker='.',label='T_inst Guyonnet')

            x='/Users/bremaud/Documents/rep instru CTIO S-A-N/T_throughput/ctio_throughput_300517_v1.txt'
            a=np.loadtxt(x)
            ax1.scatter(a.T[0],a.T[1]/max(a.T[1]),c='deepskyblue', marker='.',label='T_inst Dagoret-Campagne')
            ax1.errorbar(a.T[0],a.T[1]/max(a.T[1]), xerr=None, yerr = a.T[2]/max(a.T[1]),fmt = 'none', capsize = 1, ecolor = 'deepskyblue', zorder = 1,elinewidth = 2)

            x='/Users/bremaud/Documents/rep instru CTIO S-A-N/20171006_RONCHI400_clear_45_median_tpt.txt'
            a=np.loadtxt(x)
            x='/Users/bremaud/Documents/rep instru CTIO S-A-N/CBP_throughput.dat'
            b=np.loadtxt(x)
            def takePREMIER(elem):
                return elem[0]
            A=[[a.T[0][i],a.T[1][i]] for i in range(len(a.T[0]))]
            A.sort(key=takePREMIER)
            a.T[0]=[A[i][0] for i in range(len(A))]
            a.T[1]=[A[i][1] for i in range(len(A))]

            if a.T[0][0]<b.T[0][0]:
                L=np.linspace(a.T[0][0]-1,b.T[0][0],int(b.T[0][0]-a.T[0][0]+1))
                Y_L=[0.006]*int((b.T[0][0]-a.T[0][0]+1))
                X=np.concatenate((L,b.T[0]))
                Z=np.concatenate((Y_L,b.T[1]))

            if a.T[0][len(a.T[0])-1]>X[len(X)-1]:
                L=np.linspace(X[len(X)-1],a.T[0][len(a.T[0])-1],int(a.T[0][len(a.T[0])-1]-X[len(X)-1]+1))
                Y_L=[0.0021]*int((a.T[0][len(a.T[0])-1]-X[len(X)-1]+1))
                M=np.concatenate((X,L))
                N=np.concatenate((Z,Y_L))

            interpolation=interp1d(M,N)
            Y=interpolation(a.T[0])
            Ynew=[a.T[1][i]/Y[i] for i in range(len(Y))]
            ax1.scatter(a.T[0],Ynew/Ynew[int(len(Ynew)/2)],c='g', marker='.',label='T_inst Mondrik')

    if sim:
        ax[0].set_xlabel('$\lambda$ (nm)',fontsize=20)
        if disperseur=='Ron400':
            ax[0].set_ylabel('Transmission du CTIO normalisee',fontsize=22)
        else:
            ax[0].set_ylabel('Transmission du CTIO + '+disperseur+' normalisee',fontsize=20)

        ax[0].set_title('Transmission instrumentale avec ordre 2',fontsize=24)
        ax[0].axis([Bin[0],Bin[len(Bin)-1],0,1.1])
        ax[0].get_xaxis().set_tick_params(labelsize=20)
        ax[0].get_yaxis().set_tick_params(labelsize=15)
        ax[0].grid(True)
        ax[0].legend(prop={'size':22},loc='upper right')

        rep_ideal,rep_sim,lambda_ideal,lambda_sim=a.T[1]/max(a.T[1]),rep_instru/Max,a.T[0],new_lambda
        rep_sim_bis=rep_instru_bis/Max
        """On cherche les points de la reponse ideale (celle de Sylvie) les plus proches des longueurs d'ondes de la rep
        simulee"""
        new_rep_ideal=np.zeros(len(lambda_sim)) #nouvelle valeur de la reponse ideale pour la même plage de longueur d'onde que rep simulation
        new_rep_ideal_bis=np.zeros(len(new_lambda_bis))


        #Determination des indices de rep_Sylvie pour le calcul des ecarts relatifs
        for i in range(len(lambda_sim)):
            j=0
            while lambda_ideal[j]<lambda_sim[i]:
                j+=1
            if (lambda_ideal[j]-lambda_sim[i])<(abs(lambda_ideal[j-1]-lambda_sim[i])):
                new_rep_ideal[i]=rep_ideal[j]
            else:
                new_rep_ideal[i]=rep_ideal[j-1]

        for i in range(len(new_lambda_bis)):
            j=0
            while lambda_ideal[j]<new_lambda_bis[i]:
                j+=1
            if (lambda_ideal[j]-new_lambda_bis[i])<(abs(lambda_ideal[j-1]-new_lambda_bis[i])):
                new_rep_ideal_bis[i]=rep_ideal[j]
            else:
                new_rep_ideal_bis[i]=rep_ideal[j-1]

        "Tableaux avec les ecarts relatifs"
        new_rep_ideal_norm=np.ones(len(new_rep_ideal))
        rep_sim_norm=(rep_sim/new_rep_ideal-1)*100

        rep_sim_bis_norm=(rep_sim_bis/new_rep_ideal_bis-1)*100

        zero=np.zeros(1000)

        "Affichage"
        X_2=0
        X_2_bis=0
        for i in range(len(rep_sim_norm)-len(rep_sim_bis_norm)):
            X_2+=rep_sim_norm[i]**2
            X_2_bis+=rep_sim_norm[i]**2

        for i in range(len(rep_sim_bis_norm)):
            X_2+=rep_sim_norm[i+len(rep_sim_norm)-len(rep_sim_bis_norm)]**2
            X_2_bis+=rep_sim_bis_norm[i]**2

        X_2=np.sqrt(X_2/len(rep_sim_norm)) #correspond au sigma
        X_2_bis=np.sqrt(X_2_bis/len(rep_sim_norm)) #correspond au sigmabis

        ax[1].plot(np.linspace(Bin[0],Bin[len(Bin)-1],1000),zero,c='black')

        for i in range(len(rep_sim)):
            ax[1].scatter(lambda_sim[i],rep_sim_norm[i],c='red',marker='o')
            NewErr=new_err/new_rep_ideal/max(rep_instru)*100
            ax[1].errorbar(lambda_sim[i],rep_sim_norm[i], xerr=None, yerr = NewErr[i],fmt = 'none', capsize = 1, ecolor = 'red', zorder = 2,elinewidth = 2)

        #ax[1].scatter(new_lambda_bis,rep_sim_bis_norm,c='blue',marker='o') (fit de l'ordre 2)
        ax[1].set_xlabel('$\lambda$ (nm)',fontsize=22)
        ax[1].set_ylabel('Ecart relatif (%)',fontsize=22)
        ax[1].get_xaxis().set_tick_params(labelsize=18)
        ax[1].get_yaxis().set_tick_params(labelsize=12)
        ax[0].axvline(760,0,1,linestyle='dotted',c='black')
        ax[1].axvline(760,min(rep_sim_norm)*2,max(rep_sim_norm)*2,linestyle='dotted',c='black')
        ax[1].grid(True)
        ax[0].text(745,0.6,"debut de l'ordre 2",rotation=90,fontsize=20,color='black')
        ax[1].yaxis.set_ticks(range(int(min(rep_sim_norm))-2,int(max(rep_sim_norm))+4,(int(max(rep_sim_norm))+6-int(min(rep_sim_norm)))//8))
        ax[1].text(550,max(rep_sim_norm)*3/4,'$\sigma$= '+str(X_2)[:4]+'%',color='black',fontsize=20)
        #ax[1].text(650,max(rep_sim_norm)*3/4,'$\sigma_{bis}$= '+str(X_2_bis)[:4]+'%',color='blue',fontsize=20) (fit de l'ordre 2)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        """
        if method=='raies':
            plt.savefig('/Users/bremaud/Documents/rep_instru/simulations/'+disperseur+'/'+method+'/binwidth='+str(binwidth)+'/sigma='+str(sigma)+'/coeff_sigmaO='+str(coeff_sigmaO)+'/demi_taille_maxO='+str(demi_taille_maxO)+'/deltaO='+str(deltaO)+'/coeff_sigma='+str(coeff_sigma)+'/demi_taille_max='+str(demi_taille_max)+'/delta='+str(delta)+'/image/transmission_instrumentale07fit.pdf')
        else:
            plt.savefig('/Users/bremaud/Documents/rep_instru/simulations/'+disperseur+'/'+method+'/binwidth='+str(binwidth)+'/sigma='+str(sigma)+'/image/transmission_instrumentale03.pdf')
        """
        plt.show()

        #Trace des correlations (finalement peu utile)
        fig2 = plt.figure(figsize=(10,10))
        for i in range(len(rep_sim)):
            #plt.scatter(new_rep_ideal[i],rep_sim[i],c=(wavelength_to_rgb(lambda_sim[i])),marker='o')
            plt.scatter(new_rep_ideal[i],rep_sim[i],c='black',marker='o')
        plt.xlabel('Transmission instrumentale Sylvie',fontsize=15)
        plt.ylabel('Transmission instrumentale Vincent',fontsize=15)
        plt.gca().get_xaxis().set_tick_params(labelsize=12)
        plt.gca().get_yaxis().set_tick_params(labelsize=12)
        plt.text(0.45,0.9,'r= '+str(np.corrcoef(new_rep_ideal,rep_sim)[0][1])[:6],fontsize=18)
        "Coefficient de correlation entre les deux reponses"

        def f(x,a,b):
            return a*x+b
        popt, pcov=sp.optimize.curve_fit(f,new_rep_ideal,rep_sim)
        Z=np.linspace(0,1,10000)
        Fit=f(Z,*popt)

        "On trace un fit de la reponse simulation pour voir l'ecart avec la première bissectrice du plan"
        plt.plot(Z,Fit,label='fit',c='red',linestyle = 'solid')
        plt.plot(Z,Z,label='T_ideal',c='black',linestyle = 'solid')
        plt.title('Correlation entre les transmissions: '+'30mai'+', '+star+', '+disperseur+', '+method+', $\sigma$='+str(sigma)+', binwidth='+str(binwidth),fontsize=16)
        plt.grid(True)
        plt.axis([0,1,0,1])
        plt.legend(prop={'size':15},loc='upper left')
        #plt.savefig('/Users/bremaud/Documents/rep_instru/simulations/'+disperseur+'/'+method+'/binwidth='+str(binwidth)+'/sigma='+str(sigma)+'/coeff_sigmaO='+str(coeff_sigmaO)+'/demi_taille_maxO='+str(demi_taille_maxO)+'/deltaO='+str(deltaO)+'/coeff_sigma='+str(coeff_sigma)+'/demi_taille_max='+str(demi_taille_max)+'/delta='+str(delta)+'/image/correlation.pdf')
        plt.show()

        #rempli des fichiers avec la valeur des reponses instrumentales calculees (à partir des simu)
        """
        if method=='raies':
            fichier=open('/Users/bremaud/Documents/rep_instru/simulations/'+disperseur+'/'+method+'/binwidth='+str(binwidth)+'/sigma='+str(sigma)+'/coeff_sigmaO='+str(coeff_sigmaO)+'/demi_taille_maxO='+str(demi_taille_maxO)+'/deltaO='+str(deltaO)+'/coeff_sigma='+str(coeff_sigma)+'/demi_taille_max='+str(demi_taille_max)+'/delta='+str(delta)+'/texte/transmission_instrumentale02.txt','w')
            fichierbis=open('/Users/bremaud/Documents/rep_instru/simulations/'+disperseur+'/'+method+'/Comparaisons/02rep_binwidth='+str(binwidth)+'_sigma='+str(sigma)+'_coeff_sigmaO='+str(coeff_sigmaO)+'_demi_taille_maxO='+str(demi_taille_maxO)+'_deltaO='+str(deltaO)+'_coeff_sigma='+str(coeff_sigma)+'_demi_taille_max='+str(demi_taille_max)+'_delta='+str(delta)+'.txt','w')
            fichier.write(str(X_2)+'\n')
            fichierbis.write(str(X_2)+'\t'+str(binwidth)+'\t'+str(sigma)+'\t'+str(coeff_sigmaO)+'\t'+str(demi_taille_maxO)+'\t'+str(deltaO)+'\t'+str(coeff_sigma)+'\t'+str(demi_taille_max)+'\t'+str(delta)+'\n')
            for i in range(len(rep_instru)):
                fichier.write(str(new_lambda[i])+'\t'+str(rep_instru[i]/Max)+'\t'+str(new_err[i]/Max)+'\n')
            fichier.close()
            fichierbis.close()
        else:
            fichier=open('/Users/bremaud/Documents/rep_instru/simulations/'+disperseur+'/'+method+'/binwidth='+str(binwidth)+'/sigma='+str(sigma)+'/texte/transmission_instrumentale02.txt','w')
            fichierbis=open('/Users/bremaud/Documents/rep_instru/simulations/'+disperseur+'/'+method+'/Comparaisons/02rep_binwidth='+str(binwidth)+'_sigma='+str(sigma)+'.txt','w')
            fichier.write(str(X_2)+'\n')
            fichierbis.write(str(X_2)+'\t'+str(binwidth)+'\t'+str(sigma)+'\n')
            for i in range(len(rep_instru)):
                fichier.write(str(new_lambda[i])+'\t'+str(rep_instru[i]/Max)+'\t'+str(new_err[i]/Max)+'\n')
            fichier.close()
            fichierbis.close()"""

    else:
        ax2.set_xlabel('$\lambda$ (nm)',fontsize=24)
        if disperseur=='Ron400':
            #ax1.set_ylabel('Transmission du CTIO normalisee',fontsize=22)
            ax2.set_ylabel("Transmission normalisee",fontsize=22)
        #else:
            #ax1.set_ylabel('Transmission du CTIO + '+disperseur+' normalisee',fontsize=22)

        ax2.set_title("Illustration d'une transmission atmospherique simulee",fontsize=22)
        #ax1.axis([Bin[0],Bin[len(Bin)-1],0,1.25])
        ax2.axis([Bin[0],Bin[len(Bin)-1],0,1.25])
        ax2.get_xaxis().set_tick_params(labelsize=20)
        #ax1.get_yaxis().set_tick_params(labelsize=20)
        ax2.get_yaxis().set_tick_params(labelsize=20)

        plt.grid(True)
        #ax2.legend(prop={'size':18},loc='lower right')
        #ax1.legend(prop={'size':18},loc='lower left')
        fig.tight_layout()

        #rempli des fichiers avec la valeur des reponses instrumentales calculees (à partir des donnnees)
        if method=='raies':
            #plt.savefig('/Users/bremaud/Documents/rep_instru/donnees CTIO/'+disperseur+'/'+method+'/binwidth='+str(binwidth)+'/sigma='+str(sigma)+'/coeff_sigmaO='+str(coeff_sigmaO)+'/demi_taille_maxO='+str(demi_taille_maxO)+'/deltaO='+str(deltaO)+'/coeff_sigma='+str(coeff_sigma)+'/demi_taille_max='+str(demi_taille_max)+'/delta='+str(delta)+'/image/transmission_instrumentale.pdf')
            fichier=open('/Users/bremaud/Documents/rep_instru/donnees CTIO/'+disperseur+'/'+method+'/binwidth='+str(binwidth)+'/sigma='+str(sigma)+'/coeff_sigmaO='+str(coeff_sigmaO)+'/demi_taille_maxO='+str(demi_taille_maxO)+'/deltaO='+str(deltaO)+'/coeff_sigma='+str(coeff_sigma)+'/demi_taille_max='+str(demi_taille_max)+'/delta='+str(delta)+'/texte/transmission_instrumentale.txt','w')
            for i in range(len(rep_instru)):
                fichier.write(str(new_lambda[i])+'\t'+str(rep_instru[i]/Max)+'\t'+str(new_err[i]/Max)+'\n')
            fichier.close()

        else:
            #plt.savefig('/Users/bremaud/Documents/rep_instru/donnees CTIO/'+disperseur+'/'+method+'/binwidth='+str(binwidth)+'/sigma='+str(sigma)+'/image/transmission_instrumentale12.pdf')
            plt.show()
            """
            fichier=open('/Users/bremaud/Documents/rep_instru/donnees CTIO/'+disperseur+'/'+method+'/binwidth='+str(binwidth)+'/sigma='+str(sigma)+'/texte/transmission_instrumentale.txt','w')
            for i in range(len(rep_instru)):
                fichier.write(str(new_lambda[i])+'\t'+str(rep_instru[i]/Max)+'\t'+str(new_err[i]/Max)+'\n')
            fichier.close()"""
#CHECK

reponse_instrumentale('convoluate',True,"\\Users\\Vincent\\Documents\\Stage J.Neveu\\Programmes et prod\\CTIODataJune2017 prod4",'HD111980','Ron400',sigma=5,binwidth=20)