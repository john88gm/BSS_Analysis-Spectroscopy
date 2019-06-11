# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 12:57:59 2019

@author: gmaggioni3
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:22:16 2019

@author: John-Gioma
"""
# Import the required modules
import numpy as np
import scipy, glob, os, fnmatch, re, sklearn, time, sys
import matplotlib.pyplot as plt
from numpy import matlib

np.set_printoptions(suppress=True,precision=4)
# ---------- Home made modules

Pythondirectory = os.getcwd()[:-16];
sys.path.append(Pythondirectory)

import auxiliary_funs

# Minimisation, fitting
from scipy import optimize
from scipy import signal
from sklearn import feature_selection
from sklearn.decomposition import FastICA, PCA 

from scipy.interpolate import interp1d
from sklearn import linear_model

from pymcr.mcr import McrAR
from pymcr.regressors import OLS, NNLS
from pymcr.constraints import ConstraintNonneg, ConstraintNorm
from sklearn.linear_model.ridge import Ridge

import pandas as pd 
import xlrd as xl 
from pandas import ExcelWriter
from pandas import ExcelFile 

"""
Set which system to analyse

"""
date_index = 0
subfolder_index = 0
subsubfolder_index = 3

date = ['Mixtures_01012019','Mixtures_28012019','Mixtures_29012019',
            'Mixtures_02022019','Mixtures_05022019','Mixtures_10022019','Mixtures_14032019','Mixtures_15032019','Mixtures_02042019','Mixtures_26022019'];

subfolder = ['Raman','Set1','Set2',''];  

subsubfolder = ['NO3','NO3NO2','NO3CO3',''];      
        
python_folder_name = date[date_index] 
if subfolder_index == 1 or  subfolder_index == 2 :
    python_folder_name += '/' + subfolder[subfolder_index]
raw_folder_name = date[date_index] + '/' + subfolder[subfolder_index] + '/' + subsubfolder[subsubfolder_index]

spectroscopictechnique = 'Raman'

"""
Set Options for the Savitzky-Golay smoothening
"""
DerOrd  = 2;    # Derivative Order of the Spectrum
SGpol   = 2     # Interpolant Polynomial DegrAee
SGw     = 35;   # Number of poaints per Window
scaling = 1;    # Type of Scaling
  
"""
Set the relevant Directories
"""
# *.spc files directory
TXTSpectradirectory = Pythondirectory[:-6] + 'PostDoc/Data/' + raw_folder_name;

savedirectory = Pythondirectory[:-6] + 'Python/Data_txt/' + raw_folder_name;

MainDirectory = Pythondirectory + '/Ternary_Systems/' + python_folder_name;

"""
Load the Library of pure components
"""  
data = np.load('Library.npz')
for j in range(np.size(data.files)):
    x = data.files[j]        
#Create a variable with the name associated to that saved in the file    
for j in range(np.size(data.files)):    
    vars()[data.files[j]] = data[data.files[j]]
newspectra = np.zeros((sources.shape[0],sources.shape[1]*2))
for i in range(sources.shape[0]): 
    newspectra[i,:] = np.interp(np.linspace(wavelength[0],wavelength[-1],sources.shape[1]*2),wavelength,sources[i,:])
    f = interp1d(wavelength,sources[i,:],kind='cubic')
    newspectra[i,:] = f(np.linspace(wavelength[0],wavelength[-1],sources.shape[1]*2)) 
sources = newspectra
"""
Load the Raw and Baselined Spectra
"""      
data = np.load('Data/Measured_Data_Raman.npz')    
for j in range(np.size(data.files)):
    x = data.files[j]    
#Create a variable with the name associated to that saved in the file        
for j in range(np.size(data.files)):
    #Create a variable with the name associated to that saved in the file
    vars()[data.files[j]] = data[data.files[j]] 
newspectra = np.zeros((X0.shape[0],sources.shape[1]))
for i in range(X0.shape[0]): 
    newspectra[i,:] = np.interp(np.linspace(wavelength[0],wavelength[-1],sources.shape[1]),wavelength,X0[i,:])
    f = interp1d(wavelength,X0[i,:],kind='cubic')
    newspectra[i,:] = f(np.linspace(wavelength[0],wavelength[-1],sources.shape[1]))    
X0 = newspectra
# Values of wavenumbers
wavelength = np.linspace(wavelength[0],wavelength[-1],sources.shape[1])

# Define some copies of the Data for convenience
Xe = 1*X0;
Xe /= Xe[:,np.where(wavelength>=1640)[0][0]].reshape(-1,1)
#Xe[Xe<0] = 0.
Xo = 1*X0;
Xo -= Xo.min(1).reshape(-1,1)
X0 = 1*Xo
"""
Load Temperature and pH data from the Excel file
""" 
DataF=pd.read_excel("Data/IControl.xlsx",sheet_name='trends')
    
class Data:
    pass
IoSControl = Data

setattr(IoSControl,'rpm',DataF.values[1:,5]);
setattr(IoSControl,'ph',DataF.values[1:,2]);
setattr(IoSControl,'time',DataF.values[1:,3]);
setattr(IoSControl,'T',DataF.values[1:,4]);

# Compute the interpolated time vector for the IoS control quantities       
t0 = np.linspace(0,IoSControl.time[-1]/60,X0.shape[0])  

"""
Pre-processing: Used for ICA
"""    
# Total area under the first unscaled spectrum
SF = np.trapz(X0[10,:],wavelength)
# Mean of each spectrum
mu = X0.mean(1)
# square roof of the standard deviation, before subtracting the mean
si = np.sqrt(X0.std(1))
for i in range(X0.shape[0]):       
    X0[i,:] /= (X0[i,np.where(wavelength>=1640)[0][0]])
     # Scaling, according to 
    if scaling == 0: # Autoscaling
        X0[i,:] = ( X0[i,:]-mu[i] )/si[i]**2
    elif scaling == 1: # Pareto scaling
        X0[i,:] = ( X0[i,:]-mu[i] ) / si[i] 
    elif scaling == 2: # Vast Scaling
        X0[i,:] = ( X0[i,:]-mu[i] )/(si[i] **2)*(mu[i]/(si[i] **2))
    elif scaling == 3: # Total Area under the spcetrum  
        X0[i,:] /= np.trapz(X0[i,:],wavelength); 
    elif scaling == 4: # Total Area under the spcetrum  
        X0[i,:] = ( X0[i,:] ) / si[i]
    elif scaling == 5: # Total Area under the spcetrum  
        X0[i,:] = ( X0[i,:] )  


# Match the indices from the Raman and the IControl datasets
index = (10,16,21,30,72,97,132,182,1207,1270,1275,1295,1325,2470,2500,2530,2565,2591)
index0 = np.zeros(len(index))
for i in range(0,len(index)): 
    index0[i] = int(np.array([np.where(abs(t0[index[i],]-IoSControl.time/60)<1)[0][0]]))
index0 = index0.astype(int)      
# Plot the data fpr pH, T, and rpm
fig = plt.figure(10)
fig.set_size_inches(18.5, 25.5)
plt.subplot(3,1,1)  
plt.plot(IoSControl.time[index0,]/60,IoSControl.ph[index0,],'dk')
plt.subplot(3,1,2)  
plt.plot(IoSControl.time[index0,]/60,IoSControl.T[index0,],'dk')
plt.subplot(3,1,3)  
plt.plot(IoSControl.time[index0,]/60,IoSControl.rpm[index0,],'dk')

# Estimation of Average SNR, Noise directly estimated from Raman device used
SNR = Xo[index,:].var(1).mean()/100**2
# Local index of the SNR 
SNRi = (Xo**2)/(((Xo**2).mean())/SNR)
# Estimate the amount of noise
EstCovNoise = (Xo[index,:]**2).mean()/SNR;

# Compute the SG - Derivatives of the mixed signals
Xd = X0*1.
for kk in range(X0.shape[0]):#range(0,X0.shape[0]):
    j = 1;
    while j<=DerOrd:
        Xd[kk,:]   = scipy.signal.savgol_filter(Xd[kk,:], SGw, SGpol, deriv=1, mode='nearest');
        j +=1
"""
Determination of the number of Components via SVD
"""
Xnica = Xo[index,:]
U, Sv, V = np.linalg.svd( Xnica, full_matrices=True)
S = np.zeros(Xnica.shape)
np.fill_diagonal(S,Sv, wrap=True)
N = np.linalg.norm(Xnica,ord=1)
E = np.zeros(Xnica.shape[0])
for nn in range(0,Xnica.shape[0]):
    Rec = U@S[:,0:nn+1]@V[0:nn+1,:]
    E[nn] = np.linalg.norm(Xnica-Rec,ord=1)/N
DE = np.append(-E[0],np.diff(E)) 
nica = np.max([(sum(E>=1e-2)+1),sum(DE<-1e-2)])
print(nica)
"""
FastICA
"""
tic = time.time()
# Compute ICA
ica = FastICA(fun='exp',n_components=nica,tol=1e-8,max_iter=500)
ica.fit_transform(Xd[index,:].T)  # Reconstruct signals, needs the transpose of the matrix
A_ = ica.mixing_  # Get estimated miXeg matrix
toc = time.time()
runningtime_fastica = toc-tic # How long the decomposition took

shift = np.linalg.pinv(A_)@(mu[index,].reshape(-1,1)/si[index,].reshape(-1,1));

S0ica = (np.linalg.pinv(A_)@X0[index,:]) + shift # Reconstruct signals, needs the transpose of the matrix
Sdica = (np.linalg.pinv(A_)@Xd[index,:])  # Reconstruct signals, needs the transpose of the matrix
A_ *= si[index,].reshape(-1,1)
Aica = A_*1.
"""
MCR-ALS
"""
"""
# MCR assumes a system of the form: D = CS^T
#
# Data that you will provide (hyperspectral context):
# D [n_pixels, n_frequencies]  # Hyperspectral image unraveled in space (2D)
#
# initial_spectra [n_components, n_frequencies]  ## S^T in the literature
# OR
# initial_conc [n_pixels, n_components]   ## C in the literature

# If you have an initial estimate of the spectra
#mcrals.fit(D, ST=initial_spec)
# Otherwise, if you have an initial estimate of the concentrations
#mcrals.fit(D, C=initial_conc)

Examples to set the different options
In the following, we initiate a McrAls object 
with a st_regr setup with NNLS and 
c_reg to OLS.

One can select regressor with a string, or can import the class and instanstiate

mcrals = McrAls(max_iter=100, st_regr='NNLS', c_regr=OLS(), 
                c_constraints=[ConstraintNonneg(), ConstraintNorm()])
"""  
tic = time.time()
mcrals = McrAR(c_regr=linear_model.ElasticNet(alpha=1e-5,l1_ratio=0.75),max_iter=700,tol_err_change=Xe[index,:].max()*1e-8,st_regr='NNLS',c_constraints=[ConstraintNonneg()])
mcrals.fit(Xe[index,:], ST= S0ica**2 )
toc = time.time()
runningtime_mcrals = toc-tic # How long the decomposition took

S0mcr = mcrals.ST_opt_;
Amcr  = mcrals.C_opt_; 
Sdmcr = (np.linalg.pinv(Amcr)@Xd[index,:])

"""
Species Identification
"""
# Comparison with Library 
Cor = auxiliary_funs.correlation(S0mcr,sources,nica)
I = Cor.argmax(1);
Ivalues = Cor.max(1);
I.sort()
I = np.unique(I)

# The spectral matrix with the 
L = sources[I,:]
# The associated coefficients of the mixing matrix (proportional or equal to concentrations)
reg = linear_model.Lasso(alpha=1e-1,max_iter=2e4,positive=True)
reg.fit(L.T,Xe[index,:].T)
G = reg.coef_
Gn = G/G.sum(1)[:,None]*100
print(Gn)
err3 = (Xe[index,:]-G@L)
print(G/G.sum(1)[:,None]*100)

"""
LSQ regression
"""
Q = Xe[index,:]@np.linalg.pinv(sources)
errQ = (Xe[index,:]-Q@sources)
Qn = Q/Q.sum(1)[:,None]*100
"""
LASSO regression
"""
reg = linear_model.Lasso(alpha=1e-1,max_iter=2e4,positive=True)
reg.fit(sources.T,Xe[index,:].T)
K = reg.coef_
Kn = K/K.sum(1)[:,None]*100
errK = (Xe[index,:]-K@sources)

"""
Plots
"""
fig = plt.figure(1)
fig.set_size_inches(18.5, 25.5)
plt.subplot(3,1,1)
plt.plot(wavelength,Xe[index,:].T)
plt.subplot(3,1,2)  
for i in range(0,len(index)):
    plt.plot(wavelength,X0[index[i],:].T,color=(i/len(index),0,0))
plt.subplot(3,1,3)  
for i in range(0,len(index)):
    plt.plot(wavelength,Xd[index[i],:].T,color=(i/len(index),0,0))    
# Final Results
fig = plt.figure(4)
fig.set_size_inches(15.5, 10.5)
plt.subplot(3,1,1)  
plt.title('X, Raw Spectra')  
plt.plot(wavelength,Xe[index,:].T)
plt.subplot(3,1,2)  
plt.title('Sources - FASTICA')  
plt.plot(wavelength,S0ica.T)
plt.subplot(3,1,3)
plt.title('Sources - MCRALS')  
plt.plot(wavelength,S0mcr.T/S0mcr.max(1))

"""
Save the results
"""
    
np.savez('nica_'+ str(nica) + '_Raman',
         mu = mu, si = si,
         X0=X0,Xd=Xd, i_Data = index, Xe=Xe, Xo=Xo,       
         EstCovNoise = EstCovNoise,         
         SNR= SNR,
         Aica= Aica, S0ica=S0ica, Sdica=Sdica,
         Amcr= Amcr, S0mcr=S0mcr, Sdmcr=Sdmcr,
         SGpol=SGpol,SGw=SGw, scaling=scaling,
         wavelength=wavelength,nica=nica,
         runningtime_fastica=runningtime_fastica,
         runningtime_mcrals=runningtime_mcrals,
         I= I, Ivalues= Ivalues, L= L, G = G, K = K, Q = Q,
         DerOrd=DerOrd)

scipy.io.savemat('nica_'+ str(nica) + '_Raman',
         mdict={
                'mu' : mu, 'si' : si,
         'X0':X0,'Xd':Xd, 'i_Data' : index, 'Xe' :Xe, 'Xo' :Xo,
         'sources' : sources,
         'EstCovNoise' : EstCovNoise,
         'SNR': SNR,
         'Aica' : Aica, 'S0ica':S0ica,'Sdica':Sdica,
         'Amcr' : Amcr, 'S0mcr':S0mcr,'Sdmcr':Sdmcr,
         'SGpol':SGpol,'SGw':SGw, 'scaling':scaling,
         'wavelength':wavelength,'nica':nica,
         'DerOrd':DerOrd,
         'runningtime_fastica':runningtime_fastica,
         'runningtime_mcrals':runningtime_mcrals,
         'I': I, 'Ivalues': Ivalues, 'L': L, 'G': G, 'K': K, 'Q': Q})