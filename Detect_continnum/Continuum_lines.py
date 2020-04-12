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
from scipy import misc

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


def plot_detection_raies(fichier=r'\Users\Vincent\Documents\Stage J.Neveu\Programmes et prod\data_30may17_A2=0\reduc_20170530_075_spectrum.txt',nb_tour=1,CALSPEC=False,trigger=2,lambdamin=250,lambdamax=1150,filtre1_window=11,filtre1_order=3,moy_raies=10,demi_taille_max=40):

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

    intensite_obs_savgol=sp.signal.savgol_filter(intensite_obs,filtre1_window,filtre1_order) #filtre savgol (enlève le bruit)
    intensite_obs_savgol=smooth(intensite_obs_savgol,7,'flat',1)

    intensite_obs_savgol1=sp.interpolate.interp1d(lambda_obs,intensite_obs_savgol,kind='quadratic')

    for i in range(len(lambda_obs)):
        if lambda_obs[i]>350:
            k=i
            break

    intensite_obs_savgol2=sp.signal.savgol_filter(intensite_obs[k:],353,6)

    intensite_obs_sagol_3=sp.interpolate.interp1d(lambda_obs[k:],intensite_obs_savgol2,kind='quadratic',bounds_error=False,fill_value="extrapolate")

    fig=plt.figure(figsize=[15,10])
    plt.axis([300,1100,0,max(intensite_obs)*1.1])
    plt.plot(lambda_obs[k:],intensite_obs_savgol2,c='b')
    plt.plot(lambda_obs,intensite_obs,c='r')

    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('u.a')
    plt.title('spectre observe final')
    plt.grid(True)
    plt.show()

    lambda_complet=np.linspace(lambda_obs[0],lambda_obs[-1],int((lambda_obs[-1]-lambda_obs[0])*10+1)) #précison Angtrom

    INTENSITE_OBSS=intensite_obs_savgol

    for z in range(nb_tour):
        print(z)

        intensite_obss=INTENSITE_OBSS

        if z==0:

            D_intensite_obss=[(intensite_obss[1]-intensite_obss[0])/(lambda_obs[1]-lambda_obs[0])]
            for i in range(1,len(intensite_obss)-1):
                D_intensite_obss.append((intensite_obss[i+1]-intensite_obss[i-1])/(lambda_obs[i+1]-lambda_obs[i-1]))

            D_intensite_obss.append(0)

            intensite_derivee=sp.interpolate.interp1d(lambda_obs,D_intensite_obss)

            intensite_obss=intensite_obs_savgol1(lambda_complet)

        else:
            D_intensite_obss=[(intensite_obss[1]-intensite_obss[0])/(lambda_complet[1]-lambda_complet[0])]
            for i in range(1,len(intensite_obss)-1):
                D_intensite_obss.append((intensite_obss[i+1]-intensite_obss[i-1])/(lambda_complet[i+1]-lambda_complet[i-1]))

            D_intensite_obss.append(0)
            print(len(lambda_complet),len(intensite_obss))
            intensite_derivee=sp.interpolate.interp1d(lambda_complet,D_intensite_obss)

        D_intensite_obss=intensite_derivee(lambda_complet)

        D_mean=misc.derivative(intensite_obs_sagol_3,lambda_complet[10:-10])

        S=np.std(D_mean[:50])
        D_sigma=[]
        for i in range(50):
            D_sigma.append(S)

        for i in range(50,len(D_mean)-50):
            D_sigma.append(np.std(D_mean[i-50:i+50]))

        for i in range(len(D_mean)-50,len(D_mean)):
            D_sigma.append(np.std(D_mean[-50:]))

        Raies=[]
        Raies+=[False,False,False,False,False,False,False,False,False]

        i=10
        while i<len(D_intensite_obss)-10:
            var_signe=0
            Raies.append(False)

            if D_intensite_obss[i]<D_mean[i-10]-trigger*D_sigma[i-10]:

                k=i
                while lambda_complet[k]-lambda_complet[i]<demi_taille_max and k<len(lambda_complet)-10:
                    k+=1

                for j in range(i,k):
                    if D_intensite_obss[j+1]-D_intensite_obss[j]>0 and var_signe==0:
                        var_signe=1

                    if var_signe==1 and D_intensite_obss[j+1]-D_intensite_obss[j]<0:
                        var_signe=2

                    if var_signe==2 and D_intensite_obss[j+1]-D_intensite_obss[j]>0:
                        var_signe=3

                    if var_signe==3 and D_intensite_obss[j+1]-D_intensite_obss[j]<0:
                        break

                    if D_intensite_obss[j]>D_mean[j-10]+trigger*D_sigma[j-10]:

                        if len(lambda_complet)-10>j+k-i:
                            indice=j+k-i
                        else:
                            indice=len(lambda_complet)-10
                        for v in range(j,indice):

                            if D_intensite_obss[v+1]-D_intensite_obss[v]<0:
                                if var_signe==1:
                                    var_signe=2

                                if var_signe==3:
                                    var_signe=4

                            if D_intensite_obss[v+1]-D_intensite_obss[v]>0:
                                if var_signe==2:
                                    var_signe=3

                                if var_signe==4:
                                    break

                            if D_intensite_obss[v]<D_mean[v-10]+trigger*D_sigma[v-10]:
                                indice=v
                                break

                        if indice!=j+k-i and indice!=len(lambda_complet)-1:
                            if var_signe==2 or var_signe==4:
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

        Raies+=[False,False,False,False,False,False]

        print(len(Raies))
        print(len(lambda_complet))
        for i in range(10,len(Raies)):
            if Raies[i]==False:
                stop=0
                for j in range(i-3,i+1):
                    if Raies[j]==True:
                        stop=1
                if stop==1:
                    for j in range(i+1,i+4):
                        if Raies[j]==True:
                            stop=2
                if stop==2:
                    print(lambda_complet[i])
                    Raies[i]=True

        for i in range(len(Raies)):
            if Raies[i]==False:
                if lambda_complet[i]>972 or lambda_complet[i]<922:
                    intensite_coupe_obs.append(intensite_obss[i])
                    lambda_coupe_obs.append(lambda_complet[i])
                    D_intensite_coupe.append(D_intensite_obss[i])

        if len(Raies)<len(lambda_complet):
            for i in range(len(Raies),len(lambda_complet)):
                intensite_coupe_obs.append(intensite_obss[i])
                lambda_coupe_obs.append(lambda_complet[i])
                D_intensite_coupe.append(D_intensite_obss[i])

        intensite_obsSpline=sp.interpolate.interp1d(lambda_coupe_obs,intensite_coupe_obs)


        INTENSITE_OBS=intensite_obsSpline(lambda_complet)

        for j in range(len(lambda_complet)):
            if intensite_obs_savgol1(lambda_complet)[j]>INTENSITE_OBS[j]:
                INTENSITE_OBS[j]=intensite_obs_savgol1(lambda_complet)[j]
            if lambda_complet[j]<385:
                INTENSITE_OBS[j]=intensite_obs_savgol1(lambda_complet)[j]

        print(len(INTENSITE_OBS))
        INTENSITE_OBSS=smooth(INTENSITE_OBS,40,'flat',1)

    fig=plt.figure(figsize=[15,10])
    plt.axis([300,1100,min(D_intensite_obss)*1.1,max(D_intensite_obss)*1.1])
    plt.plot(lambda_complet,intensite_derivee(lambda_complet),c='blue')
    plt.plot(lambda_coupe_obs,D_intensite_coupe,' .',c='r')
    plt.plot(lambda_complet[10:-10],D_mean,c='g')
    plt.plot(lambda_complet[10:-10],np.array(D_mean)+trigger*np.array(D_sigma),c='purple')
    plt.plot(lambda_complet[10:-10],np.array(D_mean)-trigger*np.array(D_sigma),c='purple')
    plt.xlabel('$\lambda$ (nm)',fontsize=20)
    plt.ylabel('dérivée',fontsize=20)
    plt.title("Spectre dérivée interpolé",fontsize=20)
    plt.gca().get_xaxis().set_tick_params(labelsize=16)
    plt.gca().get_yaxis().set_tick_params(labelsize=16)
    plt.grid(True)
    plt.legend(prop={'size':16},loc='upper right')
    fig.tight_layout()

    fig=plt.figure(figsize=[15,10])
    plt.axis([300,1100,0,max(intensite_obs)*1.1])

    plt.plot(lambda_obs,intensite_obs,c='r')
    plt.plot(lambda_complet,INTENSITE_OBSS,c='black',linestyle='--')
    plt.plot(lambda_complet,intensite_obs_savgol1(lambda_complet),c='g')
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('u.a')
    plt.title('spectre observe final')
    plt.grid(True)


    plt.show()
















