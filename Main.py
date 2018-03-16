import numpy as np
from math import *
from matplotlib import pyplot as plt
import random
import time

#Simple function that takes a monomer of length n and folds it d times.
def protfold(n, d):
    #Initialises the matrix that the monomer lives in.
    if n%2 == 0:
        grid = np.zeros((n+1, n+1))
    if n%2 == 1:
        grid = np.zeros((n, n))
    center = int(ceil((n+1)/2))
    for i in range(n):
        grid[center-1][i] = i+1
    #Saves the sum of the elements to compare to (used for checking if a twist is legal, as the sum of the elements should remain the same if it is).
    s = np.sum(grid)
    c = 0
    while c < d:
        #Note that a twist is defineda as a point that is twisted (that isn't the center), and a direction that it is twisted (0, 1).
        r1 = random.randint(1, n-1)
        if r1 >= center:
            r1 += 1
        r2 = random.randint(0, 1)
        copygrid = twist(grid, r1, r2, n, center)
        if islegaltwist(copygrid, s):
            grid = copygrid
            c += 1
    #Just adding 2 to all the non.zero elements to make the start of the monomer more visible.
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != 0:
                grid[i][j] += 2
    plt.imshow(grid)
    plt.show()
    return grid

#More advanced version of the folding function. This one takes inputs monomer length n, numer of folds d, number of points in plot nt,
#st which causes the function to evaluate fewer values at higher temperatures and Tmax which defines the ending temperature.
def protfoldenergy(n, d, nt, st, Tmax):
    t = time.time()
    #Initialises the matrix containing the energies of monomers when they are adjacent to each other.
    U = np.zeros((n, n))
    for x in np.nditer(U, op_flags=['readwrite']):
        x[...] = random.uniform(-3.47*10**(-21), -10.4*10**(-21))
    #Initialises the grid with the monomer, with a different name from before (grud) so I can copy it under the name grid.
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
        #Prints the time spent calculating the monomer for the previous temperature. The first print will have time spent initializing.
        #Expected time as of right now for each step seems to be roughly 90 seconds for 15 monomers and 900 seconds for 30 monomer.
        #This could suggest O(N^3) runtime which isn't ideal, but comparing at length 5 (7 seconds) and length 10 (29 seconds) does
        #suggest that the runtime actually is O(N^2). This could suggest that the problem actually instead is related to for instance
        #how a larger percentage of potential twists aren't legal twists with a longer polymer length which can dramatically increase
        #the runtime.
        print(i, time.time()-t)
        t = time.time()
        grid = np.copy(grud)
        c = 0
        d2 = d*exp(-st*T[i])
        Enrg = energy(grid, n, U)
        while c < d2:
            r1 = random.randint(1, n-1)
            if r1 >= center:
                r1 += 1
            r2 = random.randint(0, 1)
            copygrid = twist(grid, r1, r2, n, center)
            #First we check if the twist is legal
            if islegaltwist(copygrid, s):
                Enew = energy(copygrid, n, U)
                #Then we check if it decreases the energy
                if Enew <= Enrg:
                    grid = copygrid
                    E[i][c] = Enew
                    Enrg = Enew
                #And if not we check if we twist anyways due to the temperature
                elif i > 0:
                    if random.random() < exp(-1/(1.38*10**(-23) * T[i])*(Enew-Enrg)):
                        grid = copygrid
                        E[i][c] = Enew
                        Enrg = Enew
                    else:
                        E[i][c] = Enrg
                else:
                    E[i][c] = Enrg
                #And we make sure to save the energy and diameter in any case
                L[i][c] = diam(grid, n)
                c += 1
    return E, L, T

#This function is used for the problems where we are asked to not reset the grid before folding again. It is essentially mostly the
#same as the previous function, except for that change. As a result of that change the only input values for this function is
#the monomer length n, the number of twists d (typically 30600) and the change in temperature (typically 30).
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
        Enrg = energy(grid, n, U)
        while c < d:
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
                    Enrg = Enew
                elif T[i] != 0:
                    if random.random() < exp(-(Enew-Enrg)/(1.38*10**(-23) * T[i])):
                        grid = copygrid
                        E[c2] = Enew
                        Enrg = Enew
                    else:
                        E[c2] = Enrg
                else:
                    E[c2] = Enrg
                L[c2] = diam(grid, n)
                c += 1
                c2 += 1
    return E, L, T, grid

#This function takes as input the matrix a with the monomer in it, the number r1 which is the monomer number we twist around,
#the direction r2 to twist around, and the length and center of the monomer which are useful so we don't have to calculate them
#inside this function.
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

#Checks if the sum og the matrix is the same as the sum of the original matrix. If it is then the twist was legal (since we didn't
#overwrite any of the polymers).
def islegaltwist(A, s):
    if s == np.sum(A):
        return True
    return False

#Calculates the energy in the grid by finding each pair of monomers and checking if they are next to each other, but not 2 adjacent numbers.
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

#The same as for the energy function but instead of adding to the sum for each pair that fits we just calculate the length between
#each pair of 2 monomers.
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

dmax = 25000
st = 1/1000

E, L, T = protfoldenergy(30, dmax, 61, st, 1500)

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
plt.show()

'''A = np.linspace(0, dmax-1, dmax)

plt.subplot(1, 2, 1)
plt.plot(A, [E[0][int(x)] for x in A])
plt.xlabel('Twists')
plt.ylabel('Energy')

plt.subplot(1, 2, 2)
plt.plot(A, [E[1][int(x)] for x in A])
plt.xlabel('Twists')
plt.show()'''

'''E, L, T, grid = protfoldenergy2(30, 600, 30)

xposition = np.linspace(0, 30600, 51)
for xc in xposition:
    plt.axvline(x=xc, color='k', linestyle='-', linewidth=0.2)
plt.plot(np.linspace(0, len(E)-1, len(E)), E, linewidth=0.4)
plt.plot()
plt.xlabel('Twists')
plt.ylabel('Energy')
plt.show()'''

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