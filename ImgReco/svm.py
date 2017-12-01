#!/opt/anaconda3/bin/python3 
#-*- coding:utf-8 -*-
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
import pca

clf = svm.LinearSVC(dual=False)
pca = PCA(n_components = 10)

#data and label are np.array
def apprentissage(data, lbl, acp=-1):
    t1 = time.clock()
    if acp != -1:
        global pca 
        pca = PCA(n_components=acp)
        data = pca.fit_transform(data)
    clf.fit(data, lbl) 
    t2 = time.clock()
    print('SVM : apprentissage ('+ str(len(data)) +' samples) -> ' + str(t2-t1)+'s')

def labelDev(data, label, acp=-1):
    lblRes = labeler(data, acp)
    err = 0
    for i, lbl in enumerate(lblRes):
        if lbl != label[i]:
            err += 1
    print(str(err) + " erreurs sur " + str(len(lblRes)) + " données à classer")
    return float(err)/float(len(lblRes))

def labeler(data, acp=-1):
    t1 = time.clock()
    if acp != -1:
        data = pca.transform(data)
    lblRes = clf.predict(data)
    t2 = time.clock()
    print('SVM : label ('+ str(len(data)) +' samples) -> ' + str(t2-t1)+'s')
    return lblRes;

if __name__ == "__main__":
    args = {'td' : 'data/trn_img.npy',
            'tl' : 'data/trn_lbl.npy',
            'dd' : 'data/dev_img.npy',
            'dl' : 'data/dev_lbl.npy',
            'ld' : 'data/tst_img.npy',
            'v'  : True,
            'train' : True,
            'dev' : True,
            'label' : True
            }

    x = [-1]
    x.extend(range(10,200,10))
    err = []
    res = []
    for i in x:
        if args["train"]:
            trData = np.load(args['td'])
            trLabel = np.load(args['tl']) 
            apprentissage(trData, trLabel, i) 
        if args["dev"]:
            devData = np.load(args['dd'])
            devLabel = np.load(args['dl']) 
            err.append(labelDev(devData, devLabel, i))
        if args["label"]:
            lbData = np.load(args['dd'])
            lbLabel = np.load(args['dl']) 
            res.append(labeler(lbData, i))

    plt.plot(x,err)
    plt.show()

    best = res[err.index(min(err))]
    best = np.array(best)
    np.save('test.npy', best)
