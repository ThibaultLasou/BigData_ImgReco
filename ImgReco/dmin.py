#!/usr/bin/python3
# -*- coding : utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sys

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
    classes = []
    for i in range (0,10):
        classes.append([])

    for i, img in enumerate(data):
        classes[lbl[i]].append(img)

    for i in range (0,len(classes)):
        classes[i] = np.array(classes[i])
        barycentre.append(np.mean(classes[i], 0))
    return barycentre

def labelDev(data, label, barycentre):
    lblRes = labeler(data, barycentre)
    err = 0
    for i, lbl in enumerate(lblRes):
        if lbl != label[i]:
            err += 1
    print(err)
    print(len(lblRes))
    return float(err)/float(len(lblRes))

def labeler(data, barycentre):
    lblRes = []
    for img in data:
        dist = []
        for b in barycentre:
            dist.append(np.linalg.norm(img - b))
        lblRes.append(dist.index(min(dist)))
    return lblRes


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args ={ '-td' : '',
                '-tl' : '',
                '-dd' : '',
                '-dl' : '',
                '-ld' : '',
                '-ll' : '',
                '-v' : False,
                'train' : False,
                'dev' : False,
                'label' : False
                }
        for i, arg in enumerate(sys.argv):
            if arg in args.keys():
                if arg in ['-v', 'train', 'dev', 'label']:
                    args[arg] = True
                elif sys.argv[i+1] not in args.keys():
                    args[arg] = sys.argv[i+1]
                else:
                    print("Wrong arguments")
    else:
        args = {'-td' : 'data/trn_img.npy',
                '-tl' : 'data/trn_lbl.npy',
                '-dd' : 'data/dev_img.npy',
                '-dl' : 'data/dev_lbl.npy',
                '-ld' : 'data/tst_img.npy',
                '-ll' : 'data/tst_lbl.npy',
                '-v' : True,
                'train' : True,
                'dev' : True,
                'label' : True
                }

    barycentre = []
    if args["train"]:
        trData = np.load(args['-td'])
        trLabel = np.load(args['-tl']) 
        barycentre = apprentissage(trData, trLabel)
    if args["dev"]:
        devData = np.load(args['-dd'])
        devLabel = np.load(args['-dl']) 
        err = labelDev(devData, devLabel, barycentre)
        if args['-v']:
            print(err)
    if args["label"]:
        lbData = np.load(args['-dd'])
        lbLabel = np.load(args['-dl']) 
        res = labeler(lbData,barycentre)
#        print(res)

     #   imgP = img.reshape(28,28)
     #   plt.imshow(imgP, plt.cm.gray)
     #   plt.show()

