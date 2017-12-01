#!/opt/anaconda3/bin/python3 
#-*- coding:utf-8 -*-
import sys
import time
import numpy as np
from sklearn.decomposition import PCA
import dmin

pca = PCA(n_components=10)

def acp(data):
    return pca.transform(data)

#data and label are np.array
def apprentissage(data, lbl):
    t1 = time.clock()
    ACPdata = pca.fit_transform(data)
    print('Dans ACP :')
    barycentre = dmin.apprentissage(ACPdata, lbl)
    t2 = time.clock()
    print('ACP : apprentissage ('+ str(len(data)) +' samples) -> ' + str(t2-t1)+'s')
    return barycentre

def labelDev(data, label, barycentre):
    print('Dans ACP :')
    ACPdata = acp(data)
    return dmin.labelDev(ACPdata, label, barycentre)

def labeler(data, barycentre):
    t1 = time.clock()
    ACPdata = acp(data)
    print('Dans ACP :')
    lblRes = dmin.labeler(ACPdata, barycentre)
    t2 = time.clock()
    print('ACP : label ('+ str(len(data)) +' samples) -> ' + str(t2-t1)+'s')
    return lblRes;

if __name__ == "__main__":
    args = {'td' : 'data/trn_img.npy',
            'tl' : 'data/trn_lbl.npy',
            'dd' : 'data/dev_img.npy',
            'dl' : 'data/dev_lbl.npy',
            'ld' : 'data/tst_img.npy',
            'v' : True,
            'train' : True,
            'dev' : True,
            'label' : True
            }

    for i in range(10,200,10):
        pca = PCA(n_components=i)
        print("ACP : nb_compo = " + str(i))
        barycentre = []
        if args["train"]:
            trData = np.load(args['td'])
            trLabel = np.load(args['tl']) 
            barycentre = apprentissage(trData, trLabel)
        if args["dev"]:
            devData = np.load(args['dd'])
            devLabel = np.load(args['dl']) 
            err = labelDev(devData, devLabel, barycentre)
            if args['v']:
                print(err)
        if args["label"]:
            lbData = np.load(args['dd'])
            res = labeler(lbData,barycentre)
