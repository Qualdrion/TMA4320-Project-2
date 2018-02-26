import numpy as np
from math import *
from matplotlib import pyplot as plt
import random

def protfold(n, d):
    grid = np.zeros((n, n))
    for i in range(n):
        grid[int((n-1)/2)][i] = i+1
    s = np.sum(grid)
    c = 0
    while c < d:
        r1 = random.randint(1, n-1)
        if r1 >= (n+1)/2:
            r1 += 1
        r2 = random.randint(0, 1)
        #print(r1, r2)
        copygrid = twist(grid, r1, r2, n)
        if islegaltwist(copygrid, s):
            grid = copygrid
            c += 1
    plt.imshow(grid)
    plt.show()


def twist(A, r1, r2, n):
    copy = np.copy(A)
    if r1 < (n+1)/2:
        a, b = np.where(r1+1 == A)
        a, b = a[0], b[0]
        for i in range(1, r1+1):
            c, d = np.where(i == A)
            copy[c[0]][d[0]] = 0
            if r2 == 0:
                copy[b-d[0]+a][c[0]-a+b] = i
            if r2 == 1:
                copy[d[0]-b+a][a-c[0]+b] = i
    if r1 > (n+1)/2:
        a, b = np.where(r1-1 == A)
        a, b = a[0], b[0]
        for i in range(r1, n+1):
            c, d = np.where(i == A)
            copy[c[0]][d[0]] = 0
            if r2 == 0:
                copy[b-d[0]+a][c[0]-a+b] = i
            if r2 == 1:
                copy[d[0]-b+a][a-c[0]+b] = i
    return copy

def islegaltwist(A, s):
    if s == np.sum(A):
        return True
    return False


protfold(31, 100)