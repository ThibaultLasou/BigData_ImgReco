#!/home/tlasou/anaconda3/bin/python3 
import sys
import time
import numpy as np
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

        if args["train"]:
            trData = np.load(args['-td'])
            trLabel = np.load(args['-tl']) 
            apprentissage(trData, trLabel, 10) #ici
        if args["dev"]:
            devData = np.load(args['-dd'])
            devLabel = np.load(args['-dl']) 
            err = labelDev(devData, devLabel, 1) 
            if args['-v']:
                print(err)
        if args["label"]:
            lbData = np.load(args['-dd'])
            lbLabel = np.load(args['-dl']) 
            res = labeler(lbData, 1) 
    #        print(res)

     #   imgP = img.reshape(28,28)
     #   plt.imshow(imgP, plt.cm.gray)
     #   plt.show()
