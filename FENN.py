# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:04:04 2020

@author: PYN
"""
import numpy as np
import types
from scipy.linalg import sqrtm
class KNNwithM(object):

    def __init__(self):
      self.P=None

    def train(self, X, y,neibor=5,t=1,miu=0.25):
        num_train=X.shape[0]
        dim=X.shape[1]

        dist=((X[:,np.newaxis,:]-X)**2).sum(axis=2)
        nei=(dist.argsort(axis=1))[:,:neibor]

        #compute weight V
        V=np.zeros_like(dist)
        for i in range(num_train):

          V[i, nei[i, y[i] == y[nei[i]]]] = 1   
          V[i, nei[i, y[i] != y[nei[i]]]] = -1  
          V[i,i]=0

        D=np.zeros_like(V)
        for i in range(num_train):
          D[i,i]=V[i].sum()
        L=D-V
        # XA=(X.T)@D@X
        XA=(X.T)@X
        XB=(X.T)@L@X

        XA_sqrt=sqrtm(XA)
        # print(XA,XA_sqrt,XB)
        XA_msqrt=np.linalg.inv(XA_sqrt)

        XB = XA_msqrt @ XB @ (XA_msqrt)
        XB = np.triu(XB)
        XB += XB.T - np.diag(XB.diagonal())
    
        B_eVal,B_eVec= np.linalg.eig(XB)

        ubu = np.zeros(dim)
        for i in range(dim):
            ui = B_eVec[:, i]
            ubu[i] = np.exp(-ui@XB@(ui.T)/miu)

        S_eVal=(ubu/ubu.sum())

        S = np.zeros_like(XB)
        for i in range(dim):
            ui = B_eVec[:, i]
            ui = ui[:,np.newaxis]
            S +=S_eVal[i] * (ui.T*ui)

        # print(S,S_eVal,B_eVec)
        self.P=sqrtm(S).dot(XA_msqrt)

        self.train_data = X
        self.train_label =y


        su=np.argsort(B_eVal)
        self.sv=B_eVec[:,su]


    def predict(self,X_test,k=1):
        X_test=X_test@(self.P.T)
        X_train=self.train_data@(self.P.T)
        dist=((X_test[:,np.newaxis,:]-X_train)**2).sum(axis=2)

        num_test = dist.shape[0]
        preds = np.zeros(num_test)
        for i in range(num_test):
            closest_indx = dist[i].argsort(axis=0)[:k]
            # print(i,closest_indx,self.train_label[closest_indx])
            preds[i] = np.bincount(self.train_label[closest_indx]).argmax()
        return preds

    def dim_reduce(self,X,dim):
        return X@(self.P.T)@(self.sv[:,:dim])


