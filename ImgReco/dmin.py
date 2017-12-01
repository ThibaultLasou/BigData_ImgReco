#!/usr/bin/python3
# -*- coding : utf-8 -*-

import sys
import time
import numpy as np
import matplotlib.pyplot as plt

labels={0 : "T-shirt/top",
        1 : "Trouser",
        2 : "Pullover",
        3 : "Dress",
        4 : "Coat",
        5 : "Sandal",
        6 : "Shirt",
        7 : "Sneaker",
        8 : "Bag",
        9 : "Ankle boot"}

#data and label are np.array
def apprentissage(data, lbl):
    t1 = time.clock()
    classes = []
    barycentre = []
    for i in range (0,10):
        classes.append([])

    for i, img in enumerate(data):
        classes[lbl[i]].append(img)

    for i in range (0,len(classes)):
        classes[i] = np.array(classes[i])
        barycentre.append(np.mean(classes[i], 0))
    t2 = time.clock()
    print('DMIN : apprentissage ('+ str(len(data)) +' samples) -> ' + str(t2-t1)+'s')
    return barycentre

def labelDev(data, label, barycentre):
    print('DMIN : dev')
    lblRes = labeler(data, barycentre)
    err = 0
    for i, lbl in enumerate(lblRes):
        if lbl != label[i]:
            err += 1
    print(str(err) + " erreurs sur " + str(len(lblRes)) + " données à classer")
    return float(err)/float(len(lblRes))

def labeler(data, barycentre):
    t1 = time.clock()
    lblRes = []
    for img in data:
        dist = []
        for b in barycentre:
            dist.append(np.linalg.norm(img - b))
        lblRes.append(dist.index(min(dist)))
    t2 = time.clock()
    print('DMIN : label ('+ str(len(data)) +' samples) -> ' + str(t2-t1)+'s')
    return lblRes


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
        lbLabel = np.load(args['dl']) 
        res = labeler(lbData,barycentre)
