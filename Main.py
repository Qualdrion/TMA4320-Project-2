import numpy as np
from math import *
from matplotlib import pyplot as plt
import random
import time

def protfold(n, d):
    if n%2 == 0:
        grid = np.zeros((n+1, n+1))
    if n%2 == 1:
        grid = np.zeros((n, n))
    center = int(ceil((n+1)/2))
    for i in range(n):
        grid[center-1][i] = i+1
    s = np.sum(grid)
    c = 0
    while c < d:
        r1 = random.randint(1, n-1)
        if r1 >= center:
            r1 += 1
        r2 = random.randint(0, 1)
        #print(r1, r2)
        copygrid = twist(grid, r1, r2, n, center)
        if islegaltwist(copygrid, s):
            grid = copygrid
            c += 1
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != 0:
                grid[i][j] += 2
    plt.imshow(grid)
    plt.show()
    return grid

def protfoldenergy(n, d, nt, st, Tmax):
    t = time.time()
    U = np.zeros((n, n))
    for x in np.nditer(U, op_flags=['readwrite']):
        x[...] = random.uniform(-3.47*10**(-21), -10.4*10**(-21))
    if n%2 == 0:
        grud = np.zeros((n+1, n+1))
    else:
        grud = np.zeros((n, n))
    center = int(ceil((n+1)/2))
    for i in range(n):
        grud[center-1][i] = i+1
    s = np.sum(grud)
    E = np.zeros((nt, d))
    L = np.zeros((nt, d))
    L[0][0] = n
    T = np.linspace(0, Tmax, nt)
    for i in range(nt):
        print(i, time.time()-t)
        t = time.time()
        grid = np.copy(grud)
        c = 0
        d2 = d*exp(-st*T[i])
        while c < d2:
            Enrg = energy(grid, n, U)
            r1 = random.randint(1, n-1)
            if r1 >= center:
                r1 += 1
            r2 = random.randint(0, 1)
            copygrid = twist(grid, r1, r2, n, center)
            if islegaltwist(copygrid, s):
                Enew = energy(copygrid, n, U)
                if Enew <= Enrg:
                    grid = copygrid
                    E[i][c] = Enew
                elif i > 0:
                    if random.random() < exp(-1/(1.38*10**(-23) * T[i])*(Enew-Enrg)):
                        grid = copygrid
                        E[i][c] = Enew
                    else:
                        E[i][c] = Enrg
                else:
                    E[i][c] = Enrg
                L[i][c] = diam(grid, n)
                c += 1
    return E, L, T

def protfoldenergy2(n, d, dT):
    t = time.time()
    U = np.zeros((n, n))
    for x in np.nditer(U, op_flags=['readwrite']):
        x[...] = random.uniform(-3.47*10**(-21), -10.4*10**(-21))
    if n%2 == 0:
        grid = np.zeros((n+1, n+1))
    else:
        grid = np.zeros((n, n))
    center = int(ceil((n+1)/2))
    for i in range(n):
        grid[center-1][i] = i+1
    T = np.arange(0, 1501, dT).tolist()
    T = T[::-1]
    s = np.sum(grid)
    E = np.zeros(d*len(T))
    L = np.zeros(d*len(T))
    L[0] = n
    c2 = 0
    for i in range(len(T)):
        print(i, time.time()-t)
        t = time.time()
        c = 0
        while c < d:
            Enrg = energy(grid, n, U)
            r1 = random.randint(1, n-1)
            if r1 >= center:
                r1 += 1
            r2 = random.randint(0, 1)
            copygrid = twist(grid, r1, r2, n, center)
            if islegaltwist(copygrid, s):
                Enew = energy(copygrid, n, U)
                if Enew <= Enrg:
                    grid = copygrid
                    E[c2] = Enew
                elif T[i] != 0:
                    if random.random() < exp(-(Enew-Enrg)/(1.38*10**(-23) * T[i])):
                        grid = copygrid
                        E[c2] = Enew
                    else:
                        E[c2] = Enrg
                else:
                    E[c2] = Enrg
                L[c2] = diam(grid, n)
                c += 1
                c2 += 1
    return E, L, T, grid

def twist(A, r1, r2, n, center):
    copy = np.copy(A)
    if r1 < center:
        a, b = np.where(r1+1 == A)
        a, b = a[0], b[0]
        for i in range(1, r1+1):
            c, d = np.where(i == A)
            copy[c[0]][d[0]] = 0
            if r2 == 0:
                copy[b-d[0]+a][c[0]-a+b] = i
            if r2 == 1:
                copy[d[0]-b+a][a-c[0]+b] = i
    if r1 > center:
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

def energy(grid, n, U):
    sum = 0
    for i in range(1, n+1):
        for j in range(i, n+1):
            a, b = np.where(i == grid)
            c, d = np.where(j == grid)
            a, b, c, d = a[0], b[0], c[0], d[0]
            if abs(a-c) == 1 and abs(b-d) == 0 and abs(grid[a][b]-grid[c][d]) != 1:
                sum += U[i-1][j-1] + U[j-1][i-1]
            if abs(a-c) == 0 and abs(b-d) == 1 and abs(grid[a][b]-grid[c][d]) != 1:
                sum += U[i-1][j-1] + U[j-1][i-1]
    return sum

def diam(grid, n):
    max = 0
    for i in range(1, n+1):
        for j in range(i, n+1):
            a, b = np.where(i == grid)
            c, d = np.where(j == grid)
            length = sqrt((c[0]-a[0])**2+(d[0]-b[0])**2)
            if length > max:
                max = length
    return max

#protfold(10, 0)
#protfold(10, 1)
#protfold(10, 2)

dmax = 15000
st = 1/1500

'''E, L, T = protfoldenergy(15, dmax, 16, st, 1500)

S = []
for i in range(len(E)):
    S.append(sum(E[i])/(floor(dmax*(exp(-st*T[i])))))
plt.plot(T, S)
plt.xlabel('Temperature')
plt.ylabel('<E>')
plt.show()

S2 = []
for i in range(len(L)):
    S2.append(sum(L[i])/floor(dmax*(exp(-st*T[i]))))
plt.plot(T, S2)
plt.xlabel('Temperature')
plt.ylabel('Diameter')
plt.show()'''

'''A = np.linspace(0, dmax-1, dmax)

plt.subplot(1, 2, 1)
plt.plot(A, [E[0][int(x)] for x in A])
plt.xlabel('Twists')
plt.ylabel('Energy')

plt.subplot(1, 2, 2)
plt.plot(A, [E[1][int(x)] for x in A])
plt.xlabel('Twists')
plt.show()'''

E, L, T, grid = protfoldenergy2(30, 600, 30)

xposition = np.linspace(0, 30600, 51)
for xc in xposition:
    plt.axvline(x=xc, color='k', linestyle='-', linewidth=0.2)
plt.plot(np.linspace(0, len(E)-1, len(E)), E, linewidth=0.4)
plt.plot()
plt.xlabel('Twists')
plt.ylabel('Energy')
plt.show()

'''R = []
S = []
for i in range(len(T)):
    R.append(sum(E[600*i:600*(i+1)])/600)
    S.append(sum(L[600*i:600*(i+1)])/600)
print(S)

plt.plot(T, R)
plt.xlabel('Temperature')
plt.ylabel('<E>')
plt.gca().invert_xaxis()
plt.show()

plt.plot(T, S)
plt.xlabel('Temperature')
plt.ylabel('Diameter')
plt.gca().invert_xaxis()
plt.show()

for i in range(len(grid)):
    for j in range(len(grid[i])):
        if grid[i][j] != 0:
            grid[i][j] += 2
plt.imshow(grid)
plt.show()'''