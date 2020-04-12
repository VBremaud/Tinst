# coding: utf8
## Importation des librairies

import os #gestion de fichiers
import matplotlib.pyplot as plt #affichage
import numpy as np #calculs utilisant C
import glob
#A importer

from scipy import signal #filtre savgol pour enlever le bruit
from scipy.interpolate import interp1d #interpolation
from scipy import integrate #integation
import scipy as sp #calculs
import statistics as sc #statistiques
from scipy import misc

## Convolution


def smooth(x,window_len,window,sigma=1):
    """
    Fonction: effectue la convolution d'un tableau de taille N et renvoie le tableau convolue de taille N.
    On gère les bords en dupliquant ceux-ci sur un intervalle plus grand.
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

#CHECK


## Evaluation de la transmission instrumentale


def Plot_magabsorbe_bin(NAME,magabs,magabs_err,airmass,s,Bin,method,DEBUG,sigma1,window1,LAMBDA_MIN,LAMBDA_MAX,trigger,filtre1_window,filtre1_order,moy_raies,demi_taille_max,filtre1avg_window,filtre_global,order_global,debut_filtre_global,debord_raies,seuilEAU_1,seuilEAU_2,bord_droit,bord_gauche,filtre4avg_window):

    """Fonction: effectue la division en bin de longueur d'onde du spectre observe et le calcul de l'integrale
    sur ces derniers après avoir effectue les opérations suivantes sur le spectre observe:
        -
        -
        -
        -
        -

    Après la conversion en magnitude on rempli les tableaux de magnitude et de masse d'air (méthode).

    Entrees:
        6 params changeant au cours de l'extraction de Tinst:

        magabs: matrice A x B avec A le nombre de bin de longueur d'onde et B le nombre de photo utilise contenant
        la magnitude du spectre observe par bin de longueur d'onde pour chaque photo.
        magabs_err: matrice A x B des incertitudes type associee à magabs.
        airmass: matrice A x B avec A le nombre de bin de longueur d'onde et B le nombre de photo utilise contenant
        la masse d'air du spectre observe par bin de longueur d'onde pour chaque photo.
        s: type spectractor.spectrum contenant l'ensemble des donnees pour un spectre.
        NAME: liste contenant le nom des fichiers des spectres traités.
        Bin:

        21 params (fixe):

        DEBUG:
        method:
        sigma1: (utile que pour la méthode convoluate)
        window1: (utile que pour la méthode convoluate)
        trigger:
        filtre1_window:
        filtre1_order:
        moy_raies:
        demi_taille_max:
        filtre1avg_window:
        filtre_global:
        order_global:
        debut_filtre_global:
        debord_raies:
        seuilEAU_1:
        seuilEAU_2:
        bord_droit:
        bord_gauche:
        LAMBDA_MIN:
        LAMBDA_MAX:
        filtre4avg_window:

    Sortie:
        rempli les tableaux magabs, magabs_err, airmass. (methode)"""

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

    if lambda_obs[-1]>Bin[-1] and lambda_obs[0]<Bin[0]:
        """si la plage de longueur d'onde du spectre observe est plus etroite que les valeurs max et min de la division
        en bin on ne prend pas en compte la photo et on passe à la suivante"""

        if method=='raies':

            intensite_obs_savgol=sp.signal.savgol_filter(intensite_obs,filtre1_window,filtre1_order) #filtre savgol (enlève le bruit)
            #filtre moy 2
            intensite_obs_savgol=smooth(intensite_obs_savgol,filtre1avg_window,'flat') #entre 2 pts, 1.4 nm

            intensite_obs_savgol1=sp.interpolate.interp1d(lambda_obs,intensite_obs_savgol,kind='quadratic')

            for i in range(len(lambda_obs)):
                #début du filtre "global"
                if lambda_obs[i]>debut_filtre_global:
                    k=i
                    break

            intensite_obs_savgol2=sp.signal.savgol_filter(intensite_obs[k:],filtre_global,order_global)

            intensite_obs_sagol_3=sp.interpolate.interp1d(lambda_obs[k:],intensite_obs_savgol2,kind='quadratic',bounds_error=False,fill_value="extrapolate")

            lambda_complet=np.linspace(lambda_obs[0],lambda_obs[-1],int((lambda_obs[-1]-lambda_obs[0])*10+1)) #précison Angtrom

            INTENSITE_OBSS=intensite_obs_savgol

            intensite_obss=INTENSITE_OBSS

            D_intensite_obss=[(intensite_obss[1]-intensite_obss[0])/(lambda_obs[1]-lambda_obs[0])]
            for i in range(1,len(intensite_obss)-1):
                D_intensite_obss.append((intensite_obss[i+1]-intensite_obss[i-1])/(lambda_obs[i+1]-lambda_obs[i-1]))

            D_intensite_obss.append(0)

            intensite_derivee=sp.interpolate.interp1d(lambda_obs,D_intensite_obss)

            intensite_obss=intensite_obs_savgol1(lambda_complet)


            D_intensite_obss=intensite_derivee(lambda_complet)

            D_mean=misc.derivative(intensite_obs_sagol_3,lambda_complet[moy_raies:-moy_raies])

            S=np.std(D_mean[:moy_raies*5])
            D_sigma=[]
            for i in range(moy_raies*5):
                D_sigma.append(S)

            for i in range(moy_raies*5,len(D_mean)-moy_raies*5):
                D_sigma.append(np.std(D_mean[i-moy_raies*5:i+moy_raies*5]))

            for i in range(len(D_mean)-moy_raies*5,len(D_mean)):
                D_sigma.append(np.std(D_mean[-moy_raies*5:]))


            Raies=[False,False,False,False,False,False,False,False,False,False]

            i=moy_raies
            while i<len(D_intensite_obss)-moy_raies:
                var_signe=0
                Raies.append(False)

                if D_intensite_obss[i]<D_mean[i-moy_raies]-trigger*D_sigma[i-moy_raies]:

                    k=i
                    while lambda_complet[k]-lambda_complet[i]<demi_taille_max and k<len(lambda_complet)-moy_raies:
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

                        if D_intensite_obss[j]>D_mean[j-moy_raies]+trigger*D_sigma[j-moy_raies]:

                            if len(lambda_complet)-moy_raies>j+k-i:
                                indice=j+k-i
                            else:
                                indice=len(lambda_complet)-moy_raies
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

                                if D_intensite_obss[v]<D_mean[v-moy_raies]+trigger*D_sigma[v-moy_raies]:
                                    indice=v
                                    break

                            if indice!=j+k-i and indice!=len(lambda_complet)-1:
                                if var_signe==2 or var_signe==4:
                                    for loop in range(i+1,indice+debord_raies+1):
                                        Raies.append(True)
                                    for loop in range(i-debord_raies-1,i+1):
                                        Raies[i]=True
                                    i=indice+4
                                    Raies.append(False)
                            break
                i+=1

            intensite_coupe_obs=[]
            lambda_coupe_obs=[]
            D_intensite_coupe=[]

            if len(Raies)<len(lambda_complet):
                for j in range(len(Raies),len(lambda_complet)):
                    Raies.append(False)

            for i in range(10,len(Raies)):
                if Raies[i]==False:
                    stop=0
                    for j in range(i-debord_raies,i+1):
                        if Raies[j]==True:
                            stop=1
                    if stop==1:
                        for j in range(i+1,i+debord_raies+1):
                            if Raies[j]==True:
                                stop=2
                    if stop==2:
                        Raies[i]=True

            for i in range(len(Raies)):
                if Raies[i]==False:
                    if lambda_complet[i]>seuilEAU_2 or lambda_complet[i]<seuilEAU_1:
                        intensite_coupe_obs.append(intensite_obss[i])
                        lambda_coupe_obs.append(lambda_complet[i])
                        D_intensite_coupe.append(D_intensite_obss[i])

            intensite_obsSpline=sp.interpolate.interp1d(lambda_coupe_obs,intensite_coupe_obs,bounds_error=False,fill_value="extrapolate")
            INTENSITE_OBS=intensite_obsSpline(lambda_complet)

            for j in range(len(lambda_complet)):
                if intensite_obs_savgol1(lambda_complet)[j]>INTENSITE_OBS[j] or lambda_complet[j]>bord_droit:

                    INTENSITE_OBS[j]=intensite_obs_savgol1(lambda_complet)[j]
                if lambda_complet[j]<bord_gauche:

                    INTENSITE_OBS[j]=intensite_obs_savgol1(lambda_complet)[j]


            INTENSITE_OBSS=smooth(INTENSITE_OBS,filtre4avg_window,'flat',1)

            interpolation_obs=sp.interpolate.interp1d(lambda_complet,INTENSITE_OBSS)

            if DEBUG:
                plt.figure(figsize=[6,6])
                plt.axis([LAMBDA_MIN,LAMBDA_MAX,0,max(intensite_obs)*1.1])
                plt.plot(lambda_obs,intensite_obs,c='r')
                plt.plot(lambda_complet,INTENSITE_OBSS,c='black')
                plt.xlabel('$\lambda$ (nm)')
                plt.ylabel('u.a')
                plt.title('spectre observe sans raies')
                plt.grid(True)
                plt.show()


        elif method=='convoluate':

            intensite_obss=smooth(intensite_obs,6*sigma1,window1,sigma1)
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

def Plot_magabsorbe_star(method,sim,fileday,star,disperseur,binwidth,lambda_min,lambda_max,sigma_max,mag_diffmax,filtre2_order,filtre3_order,filtre2avg_window,filtre2_window,filtre3avg_window,filtre3_window,lambda_mid,DEBUG,sigma1,window1,LAMBDA_MIN,LAMBDA_MAX,trigger,filtre1_window,filtre1_order,moy_raies,demi_taille_max,filtre1avg_window,filtre_global,order_global,debut_filtre_global,debord_raies,seuilEAU_1,seuilEAU_2,bord_droit,bord_gauche,filtre4avg_window):
    #filtre 2 window --> trigger2
    #order2 --> nb tour

    #16 params

    """Fonction: renvoie les tableaux magabas, airmass, fluxlumBin_reel, Bin.

    Entree:

        16 params (fixe):

        sim: True si il s'agit d'une simulation, False sinon.
        fileday: chemin où se trouve le fichier contenant les spectres à etudier.
        star: nom de l'etoile à etudier.
        disperseur: nom du disperseur à etudier (Ron400,Thor300,HoloPhP,HoloPhAg,HoloAmAg)
        binwidth: taille des bin de longueur d'onde.
        lambda_min: longueur d'onde minimum utilisee lors de la division en bin.
        lambda_max: longueur d'onde maximum utilisee lors de la division en bin.
        sigma_max:
        mag_diffmax:
        filtre2_order:
        filtre3_order:
        filtre2avg_window:
        filtre2_window:
        filtre3avg_window:
        filtre3_window:
        lambda_mid:

        6 params déjà vu dans Plot_magabsorbe_bin (fixe):

        DEBUG:
        method:
        sigma1: (utile que pour la méthode convoluate)
        window1: (utile que pour la méthode convoluate)
        LAMBDA_MIN:
        LAMBDA_MAX:

        15 params utilent pour une fonction interne cf Plot_magabsorbe_bin

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
                if float(a[0])>lambda_min-50 and float(a[0])<lambda_max+50:
                    intensite_reel.append(float(a[1]))
                    lambda_reel.append(float(a[0]))

            if method=='raies':

                #méthode maxmin

                intensite_reel_savgol=sp.signal.savgol_filter(intensite_reel,filtre3_window,filtre3_order) #filtre savgol (enlève le bruit)
                intensite_reel_moy=smooth(intensite_reel,filtre3avg_window,'flat',1)

                intensite_reel_1=(intensite_reel_savgol+intensite_reel_moy)/2

                j=0
                for i in range(len(intensite_reel_1)):
                    if lambda_reel[i]>lambda_mid:
                        j=i
                        break

                intensite_reel_1[j:]=intensite_reel_savgol[j:]

                intensite_reel_1=sp.interpolate.interp1d(lambda_reel,intensite_reel_1,kind='cubic')

                lambda_complet=np.linspace(lambda_reel[0],lambda_reel[-1],int((lambda_reel[-1]-lambda_reel[0])*10+1)) #précision Angtrom
                Intensite_reel=intensite_reel_1(lambda_complet)

                intensite_tronque=[Intensite_reel[0]]
                lambda_tronque=[lambda_complet[0]]
                c=0
                for i in range(1,len(lambda_complet)-1):
                    if (Intensite_reel[i+1]-Intensite_reel[i-1])/(lambda_complet[i+1]-lambda_complet[i-1])>0:
                        c=1

                    elif c==1 and (Intensite_reel[i+1]-Intensite_reel[i-1])/(lambda_complet[i+1]-lambda_complet[i-1])<0:
                        intensite_tronque.append(Intensite_reel[i])
                        lambda_tronque.append(lambda_complet[i])
                        c=0

                intensite_tronque.append(Intensite_reel[-1])
                lambda_tronque.append(lambda_complet[-1])


                for j in range(100):
                    intensite_tronque2=[Intensite_reel[0]]
                    lambda_tronque2=[lambda_complet[0]]
                    c=0
                    for i in range(1,len(lambda_tronque)-1):
                        if intensite_tronque[i-1]<intensite_tronque[i] or intensite_tronque[i+1]<intensite_tronque[i]:
                            intensite_tronque2.append(intensite_tronque[i])
                            lambda_tronque2.append(lambda_tronque[i])

                    intensite_tronque2.append(Intensite_reel[-1])
                    lambda_tronque2.append(lambda_complet[-1])

                    intensite_tronque=intensite_tronque2
                    lambda_tronque=lambda_tronque2

                Intensite_reels=sp.interpolate.interp1d(lambda_tronque,intensite_tronque,bounds_error=False,fill_value="extrapolate")
                Intensite_reel=Intensite_reels(lambda_complet)
                INTENSITE_reel=smooth(Intensite_reel,filtre2avg_window,'flat',1)
                INTENSITE_reelS=sp.signal.savgol_filter(INTENSITE_reel,filtre2_window,filtre2_order)

                interpolation_reel=sp.interpolate.interp1d(lambda_complet,INTENSITE_reelS)

                if DEBUG:
                    plt.figure(figsize=[6,6])
                    plt.axis([LAMBDA_MIN,LAMBDA_MAX,0,max(intensite_reel)*1.1])
                    plt.plot(lambda_reel,intensite_reel,c='r')
                    plt.plot(lambda_complet,INTENSITE_reelS,c='black')
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
            Plot_magabsorbe_bin(NAME,magabs,magabs_err,airmass,s,Bin,method,DEBUG,sigma1,window1,LAMBDA_MIN,LAMBDA_MAX,trigger,filtre1_window,filtre1_order,moy_raies,demi_taille_max,filtre1avg_window,filtre_global,order_global,debut_filtre_global,debord_raies,seuilEAU_1,seuilEAU_2,bord_droit,bord_gauche,filtre4avg_window)

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
                if sim and S>mag_diffmax:
                    L.append(i)
                    break
                elif sim==False and S>mag_diffmax:
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
                        if S>sigma_max*np.std(magabs[j]):
                            L.append(i)
                            break

                    else:
                        if S>sigma_max*np.std(magabs[j]):
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


def droites_Bouguer(method,sim,fileday,star,disperseur,binwidth,lambda_min,lambda_max,sigma_max,mag_diffmax,filtre2_order,filtre3_order,filtre2avg_window,filtre2_window,filtre3avg_window,filtre3_window,lambda_mid,DEBUG,sigma1,window1,LAMBDA_MIN,LAMBDA_MAX,trigger,filtre1_window,filtre1_order,moy_raies,demi_taille_max,filtre1avg_window,filtre_global,order_global,debut_filtre_global,debord_raies,seuilEAU_1,seuilEAU_2,bord_droit,bord_gauche,filtre4avg_window):

    """Fonction: effectue pour toute les photos d'une même etoile, le trace de magabs par bin de longueur d'onde

    Entree:
        37 params utilent pour une fonction interne cf Plot_magabsorbe_star

    Sortie:
        Trace des droites de Bouguer.
        Trace de la magnitude absorbee en fonction de la masse d'air.
        Tableau des coefficients ordonnees à l'origine et pente (il s'agit d'une matrice de taille nombre de bin x 2)
        Tableaux des bins, des incertitudes, et de l'intensite par bin tabulee"""

    "Recuperation des tableaux renvoyes par Plot_magabsorbe_star"
    airmass, magabs, magabs_err, Bin, fluxlum_Binreel=Plot_magabsorbe_star(method,sim,fileday,star,disperseur,binwidth,lambda_min,lambda_max,sigma_max,mag_diffmax,filtre2_order,filtre3_order,filtre2avg_window,filtre2_window,filtre3avg_window,filtre3_window,lambda_mid,DEBUG,sigma1,window1,LAMBDA_MIN,LAMBDA_MAX,trigger,filtre1_window,filtre1_order,moy_raies,demi_taille_max,filtre1avg_window,filtre_global,order_global,debut_filtre_global,debord_raies,seuilEAU_1,seuilEAU_2,bord_droit,bord_gauche,filtre4avg_window)

    "Definition de la fonction lineaire qu'on va utiliser pour tracer les droites de Bouguer."
    def f(x,a,b):
        return a*x+b

    Z=np.linspace(0,2.2,1000) #nombre de points pour le trace des droites


    coeff=np.zeros((len(magabs),2)) #on initialise la liste des coefficients à la liste vide.
    err=np.zeros(len(magabs)) #on initialise la liste des erreurs sur les ordonnees à l'origine à la liste vide.

    fig=plt.figure(figsize=[15, 10])

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
        plt.plot(Z,MAG,c='black')
        plt.plot(Z,MAG_sup,c='black',linestyle=':')
        plt.plot(Z,MAG_inf,c='black',linestyle=':')

        plt.scatter(airmass[i],magabs[i],c='black',
            label=f'{Bin[i]}-{Bin[i+1]} nm', marker='o',s=30)

        plt.errorbar(airmass[i],magabs[i], xerr=None, yerr = magabs_err[i],fmt = 'none',
            capsize = 1, ecolor=('black'), zorder = 2,elinewidth = 2)


    return (coeff,Bin,err,fluxlum_Binreel)
#CHECK


def reponse_instrumentale(method,sim,fileday,star,disperseur,sigma2=10,window_len2=20,binwidth=20,lambda_min=350,lambda_max=1030,sigma_max=3,mag_diffmax=1,filtre2_order=3,filtre3_order=3,filtre2avg_window=10,filtre2_window=61,filtre3avg_window=50,filtre3_window=11,lambda_mid=475,DEBUG=False,sigma1=5,window1='gaussian',LAMBDA_MIN=350,LAMBDA_MAX=1099,trigger=2,filtre1_window=17,filtre1_order=3,moy_raies=10,demi_taille_max=40,filtre1avg_window=12,filtre_global=353,order_global=6,debut_filtre_global=350,debord_raies=3,seuilEAU_1=922,seuilEAU_2=972,bord_droit=980,bord_gauche=389,filtre4avg_window=40):

    """Fonction: trace la reponse instrumentale obtenue ainsi que celle de Sylvie, Augustion et Nick et enregistre les
    donnees dans des fichiers pdf (pour la reponse instrumentale) et txt cree à l'execution.
    Si sim: uniquement la reponse de Sylvie.

    Entree:
        2 params (fixe):

        sigma2:
        winwow_len2:

        37 params utilent pour une fonction interne cf Plot_magabsorbe_star

    Sortie:
        Trace de la reponse instrumentale.
        Trace de celle de Sylvie, Nick et Augustin si il s'agit du disperseur Ronchie400.
        Enregistre les donnees dans un chemin et avec un nom lies aux paramètres d'entree de la fonction."""

    "Recuperation des tableaux renvoyes par droites_Bouguer"
    coeff, Bin, err, fluxlum_Binreel=droites_Bouguer(method,sim,fileday,star,disperseur,binwidth,lambda_min,lambda_max,sigma_max,mag_diffmax,filtre2_order,filtre3_order,filtre2avg_window,filtre2_window,filtre3avg_window,filtre3_window,lambda_mid,DEBUG,sigma1,window1,LAMBDA_MIN,LAMBDA_MAX,trigger,filtre1_window,filtre1_order,moy_raies,demi_taille_max,filtre1avg_window,filtre_global,order_global,debut_filtre_global,debord_raies,seuilEAU_1,seuilEAU_2,bord_droit,bord_gauche,filtre4avg_window)

    """On calcul les tableaux rep_instru correspondant à la reponse instrumentale, new_err l'erreur associee et
    new_lambda la longueur d'onde associe à la reponse instrumentale."""
    rep_instru=np.exp(coeff.T[1])/fluxlum_Binreel
    new_err=err*np.exp(coeff.T[1])/fluxlum_Binreel

    new_lambda=0.5*(Bin[1:]+Bin[:-1])


    Max=np.max(rep_instru)

    "Affichage"
    if sim:
        gs_kw = dict(height_ratios=[4,1], width_ratios=[1])
        fig, ax = plt.subplots(2,1,sharex="all",figsize=[14,12], constrained_layout=True, gridspec_kw=gs_kw)
        ax[0].scatter(new_lambda,rep_instru/Max,c='black', marker='o',label='T_inst Vincent',zorder=2,s=35)
        ax[0].errorbar(new_lambda,rep_instru/Max, xerr=None, yerr = new_err/max(rep_instru),fmt = 'none', capsize = 1, ecolor = 'black', zorder = 2,elinewidth = 2)

        x='T_throughput/ctio_throughput_300517_v1.txt'

        a=np.loadtxt(x)
        ax[0].scatter(a.T[0],a.T[1]/max(a.T[1]),c='deepskyblue', marker='.',label='T_inst exacte')
        ax[0].errorbar(a.T[0],a.T[1]/max(a.T[1]), xerr=None, yerr = a.T[2]/max(a.T[1]),fmt = 'none', capsize = 1, ecolor = 'deepskyblue', zorder = 1,elinewidth = 2)


    else:
        #Affichage rep instru comparees + atmosphère
        fig=plt.figure(figsize=[14,9])
        ax2 = fig.add_subplot(111)

        ax2.scatter(new_lambda,rep_instru/Max,c='black', marker='o',zorder=2,s=15)
        ax2.errorbar(new_lambda,rep_instru/Max, xerr=None, yerr = new_err/max(rep_instru),fmt = 'none', capsize = 1, ecolor = 'black', zorder = 2,elinewidth = 2)

        new_lambda1=[new_lambda[i] for i in range(len(new_lambda))]
        new_lambda=[new_lambda[i] for i in range(len(new_lambda))]
        rep_instru=[rep_instru[i] for i in range(len(rep_instru))]
        endlambda=int(new_lambda[-1])+1
        endrep=rep_instru[-1]
        for i in range(endlambda,LAMBDA_MAX-1):
            new_lambda.append(i)
            rep_instru.append(endrep)

        new_lambda=np.array(new_lambda)
        rep_instru=np.array(rep_instru)

        Tinst=sp.interpolate.interp1d(new_lambda,rep_instru,bounds_error=False,fill_value="extrapolate")

        lambda_tinst=np.linspace(LAMBDA_MIN,LAMBDA_MAX,LAMBDA_MAX-LAMBDA_MIN+1)
        rep_instru_vincent=Tinst(lambda_tinst)

        rep_instru_vincent=smooth(rep_instru_vincent,window_len2,'gaussian',sigma2)

        yerr=sp.interpolate.interp1d(new_lambda1,new_err,bounds_error=False,fill_value="extrapolate")
        yerR=yerr(lambda_tinst)

        Max=max(rep_instru_vincent)
        ax2.scatter(lambda_tinst,rep_instru_vincent/Max,c='red',marker='.',label='Tinst_Vincent')
        ax2.errorbar(lambda_tinst,rep_instru_vincent/Max, xerr=None, yerr = yerR/Max,fmt = 'none', capsize = 1, ecolor = 'red', zorder = 1,elinewidth = 2)

        """
        x='T_throughput/ctio_throughput_300517_v1.txt'

        a=np.loadtxt(x)
        ax2.scatter(a.T[0],a.T[1]/max(a.T[1]),c='deepskyblue', marker='.',label='Tinst_Sylvie')
        ax2.errorbar(a.T[0],a.T[1]/max(a.T[1]), xerr=None, yerr = a.T[2]/max(a.T[1]),fmt = 'none', capsize = 1, ecolor = 'deepskyblue', zorder = 1,elinewidth = 2)
        """
        """
        fichier=open('/Users/Vincent/Documents/Stage J.Neveu/Programmes et prod/Pyzo/T_throughput/Ctio_'+disperseur+'.txt','w')
        for i in range(len(rep_instru_vincent)):
            fichier.write(str(lambda_tinst[i])+'\t'+str(rep_instru_vincent[i])+'\t'+str(yerR[i])+'\n')
        fichier.close()
        """
    if sim:
        ax[0].set_xlabel('$\lambda$ (nm)',fontsize=20)
        if disperseur=='Ron400':
            ax[0].set_ylabel('Transmission du CTIO normalisee',fontsize=17)
        else:
            ax[0].set_ylabel('Transmission du CTIO + '+disperseur+' normalisee',fontsize=20)

        ax[0].set_title('Transmission instrumentale sans ordre 2, '+star+', '+disperseur+', F7V-VI',fontsize=18)
        ax[0].axis([Bin[0],Bin[-1],0,1.1])
        ax[0].get_xaxis().set_tick_params(labelsize=17)
        ax[0].get_yaxis().set_tick_params(labelsize=14)
        ax[0].grid(True)
        ax[0].legend(prop={'size':22},loc='upper right')

        rep_ideal,rep_sim,lambda_ideal,lambda_sim=a.T[1]/max(a.T[1]),rep_instru/Max,a.T[0],new_lambda

        """On cherche les points de la reponse ideale (celle de Sylvie) les plus proches des longueurs d'ondes de la rep
        simulee"""
        new_rep_ideal=np.zeros(len(lambda_sim)) #nouvelle valeur de la reponse ideale pour la même plage de longueur d'onde que rep simulation



        #Determination des indices de rep_Sylvie pour le calcul des ecarts relatifs
        for i in range(len(lambda_sim)):
            j=0
            while lambda_ideal[j]<lambda_sim[i]:
                j+=1
            if (lambda_ideal[j]-lambda_sim[i])<(abs(lambda_ideal[j-1]-lambda_sim[i])):
                new_rep_ideal[i]=rep_ideal[j]
            else:
                new_rep_ideal[i]=rep_ideal[j-1]


        "Tableaux avec les ecarts relatifs"
        new_rep_ideal_norm=np.ones(len(new_rep_ideal))
        rep_sim_norm=(rep_sim/new_rep_ideal-1)*100

        zero=np.zeros(1000)

        "Affichage"
        X_2=0
        for i in range(len(rep_sim_norm)):
            X_2+=rep_sim_norm[i]**2


        X_2=np.sqrt(X_2/len(rep_sim_norm)) #correspond au sigma

        ax[1].plot(np.linspace(Bin[0],Bin[-1],1000),zero,c='black')

        for i in range(len(rep_sim)):
            ax[1].scatter(lambda_sim[i],rep_sim_norm[i],c='red',marker='o')
            NewErr=new_err/new_rep_ideal/max(rep_instru)*100
            ax[1].errorbar(lambda_sim[i],rep_sim_norm[i], xerr=None, yerr = NewErr[i],fmt = 'none', capsize = 1, ecolor = 'red', zorder = 2,elinewidth = 2)

        ax[1].set_xlabel('$\lambda$ (nm)',fontsize=17)
        ax[1].set_ylabel('Ecart relatif (%)',fontsize=15)
        ax[1].get_xaxis().set_tick_params(labelsize=18)
        ax[1].get_yaxis().set_tick_params(labelsize=10)

        ax[1].grid(True)

        ax[1].yaxis.set_ticks(range(int(min(rep_sim_norm))-2,int(max(rep_sim_norm))+4,(int(max(rep_sim_norm))+6-int(min(rep_sim_norm)))//8))
        ax[1].text(550,max(rep_sim_norm)*3/4,'$\sigma$= '+str(X_2)[:4]+'%',color='black',fontsize=20)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()



    else:
        ax2.set_xlabel('$\lambda$ (nm)',fontsize=24)
        ax2.set_ylabel("Transmission normalisee",fontsize=22)
        ax2.set_title("Transmission instrumentale du CTIO, "+disperseur,fontsize=22)
        ax2.axis([Bin[0],Bin[-1],0,1.25])
        ax2.get_xaxis().set_tick_params(labelsize=20)
        ax2.get_yaxis().set_tick_params(labelsize=20)
        ax2.legend(prop={'size':17},loc='upper right')
        plt.grid(True)
        fig.tight_layout()

        plt.show()

#CHECK

#reponse_instrumentale('raies',True,"/home/tp-home005/vbremau/StageM1/data_30may17_A2=0",'HD111980','Ron400')
reponse_instrumentale('raies',False,r"\Users\Vincent\Documents\Stage J.Neveu\Programmes et prod\data_30may17_A2=0",'HD111980','HoloAmAg',DEBUG=True)
