#!/usr/bin/python3
# -*-coding:Utf-8 -*

import re
import numpy as np
import matplotlib.pyplot as plt

def barycentre(w):
    return (np.mean(w, 0))

def droiteSep(u1, u2):
    v = u1-u2
    w = (u2 + u1)/2
    U = np.dot(np.array([[0,1],[-1,0]]),v)
    B = w+U
    return w,B

if __name__ == '__main__':

    w1 = []
    w2 = []

    with open("data1",'rt') as dataFile:
        for row in dataFile:
            r = re.split('\s+', row)
            p = [int(r[0]), int(r[1])]
            if int(r[2]) == 1:
                w1.append(p)
            elif int(r[2]) == 2:
                w2.append(p)
    w1 = np.array(w1)
    w2 = np.array(w2)
    u1 = barycentre(w1)
    u2 = barycentre(w2)
    print (u1, u2)
    droiteSep(u1, u2)
    
    
    plt.plot(u1[0],u1[1],'ro')
    plt.plot(u2[0],u2[1],'ro')
    plt.axis([0.,5,0.,5])
    u,B = droiteSep(u1,u2)
    plt.plot([u[0],B[0]],[u[1],B[1]],'r-')
    plt.show()
