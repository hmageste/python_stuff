#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple Kriging
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Rob Knight, Gavin Huttley, and Peter Maxwell"
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Rob Knight", "Peter Maxwell", "Gavin Huttley",
                    "Matthew Wakefield"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Rob Knight"
__email__ = "rob@spot.colorado.edu"
__status__ = "Production" #types: "Prototype", "Development", or "Production"

def KS(dados,blocos,variograma):
    resultado = np.zeros(blocos)
    
    cov_angulos = np.zeros((dados.shape[0],dados.shape[0]))
    cov_distancias = np.zeros((dados.shape[0],dados.shape[0]))
    K = np.zeros((dados.shape[0],dados.shape[0]))
    for i in xrange(dados.shape[0]-1):
        cov_angulos[i,i:]=np.arctan2((dados[i:,1]-dados[i,1]),(dados[i:,0]-dados[i,0]))
        cov_distancias[i,i:]=np.sqrt((dados[i:,0]-dados[i,0])**2+(dados[i:,1]-dados[i,1])**2)
    for i in xrange(dados.shape[0]):
        for j in xrange(dados.shape[0]):
            if cov_distancias[i,j]!=0:
                amp=np.sqrt((variograma[1]*np.cos(cov_angulos[i,j]))**2+(variograma[0]*np.sin(cov_angulos[i,j]))**2)
                K[i,j]=dados[:,2].var()*(1-np.e**(-3*cov_distancias[i,j]/amp))
    K = K + K.T
    
    for x in xrange(resultado.shape[0]):
        for y in xrange(resultado.shape[1]):
             distancias = np.sqrt((x-dados[:,0])**2+(y-dados[:,1])**2)
             angulos = np.arctan2(y-dados[:,1],x-dados[:,0])
             amplitudes = np.sqrt((variograma[1]*np.cos(angulos[:]))**2+(variograma[0]*np.sin(angulos[:]))**2)
             M = dados[:,2].var()*(1-np.e**(-3*distancias[:]/amplitudes[:]))
             W = LA.solve(K,M)
             resultado[x,y] = np.sum(W*(dados[:,2]-dados[:,2].mean()))+dados[:,2].mean()
    return resultado
