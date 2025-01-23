import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy


grid = 6
label = 'plot for '+str(grid)+' x '+str(grid)+' grid'
imax = grid
jmax = grid
nmax = 100
rw = 0.1
rx = 1
ry = 1
rz = 0.1
thetaB = 0
thetaN = 90
eps = .00001
rexparam = 1.5
pi = 3.1415927

x = np.zeros((grid,grid))
y = np.zeros((grid,grid))
M_AB = np.zeros((grid,grid))
N_AB = np.zeros((grid,grid))
M_BC = np.zeros((grid,grid))
N_BC = np.zeros((grid,grid))
M_CD = np.zeros((grid,grid))
N_CD = np.zeros((grid,grid))
M_DA = np.zeros((grid,grid))
N_DA = np.zeros((grid,grid))
phi = np.zeros((grid,grid))
phi_x = np.zeros((grid,grid))

n_wx = imax-1
n_zy = jmax-1
nm_wx = n_wx
nm_zy = n_zy
dr_wx = (rx-rw)/nm_wx
dr_zy = (ry-rz)/nm_wx
d_th = (thetaN-thetaB)/nm_zy


for j in range(1,jmax+1):
    cntj = j-1
    thetaj = (thetaB + cntj*d_th)*pi/180
    if (j<grid): cosj = math.cos(thetaj)
    elif (j==grid): cosj = 0
    sinj = math.sin(thetaj)
    dr = dr_wx + (dr_zy-dr_wx)*cntj/nm_zy
    r_wz = rw + (rz-rw)*cntj/nm_zy
    
    for i in range(1,imax+1):
        cnti = i-1
        r = r_wz + cnti*dr
        x[-j][i-1] = r*cosj
        y[-j][i-1] = r*sinj
        phi[-j][i-1] = sinj/r
phi_x = copy(phi)

x=x.round(4)
y=y.round(4)

for j in range(1,grid-1):
    for i in range(1,grid-1):
        
        xA = 0.25*(x[-i][j-1] + x[-i][j]+ x[-(i+1)][j] + x[-(i+1)][j-1]).round(5)
        yA = 0.25*(y[-i][j-1] + y[-i][j]+ y[-(i+1)][j] + y[-(i+1)][j-1]).round(5)
        
        xB = 0.25*(x[-i][j] + x[-i][j+1]+ x[-(i+1)][j+1] + x[-(i+1)][j]).round(5)
        yB = 0.25*(y[-i][j] + y[-i][j+1]+ y[-(i+1)][j+1] + y[-(i+1)][j]).round(5)
        
        xC = 0.25*(x[-(i+1)][j] + x[-(i+1)][j+1]+ x[-(i+2)][j+1] + x[-(i+2)][j]).round(5)
        yC = 0.25*(y[-(i+1)][j] + y[-(i+1)][j+1]+ y[-(i+2)][j+1] + y[-(i+2)][j]).round(5)
        
        xD = 0.25*(x[-(i+1)][j-1] + x[-(i+1)][j]+ x[-(i+2)][j] + x[-(i+2)][j-1]).round(5)
        yD = 0.25*(y[-(i+1)][j-1] + y[-(i+1)][j]+ y[-(i+2)][j] + y[-(i+2)][j-1]).round(5)
        
        '''AB'''
        dxA = xB-xA
        dyA = yB-yA
        dxJ = x[-(i+1)][j]-x[-i][j]
        dyJ = y[-(i+1)][j]-y[-i][j]
        sAB = abs(dxA*dyJ - dxJ*dyA)
        M_AB[-(i+1)][j] = (dxA**2 + dyA**2)/sAB
        N_AB[-(i+1)][j] = (dxA*dxJ + dyA*dyJ)/sAB
        
        '''BC'''
        dxB = xC-xB
        dyB = yC-yB
        dxI = x[-(i+1)][j]-x[-(i+1)][j+1]
        dyI = y[-(i+1)][j]-y[-(i+1)][j+1]
        sBC = abs(dxB*dyI - dxI*dyB)
        M_BC[-(i+1)][j] = (dxB**2 + dyB**2)/sBC
        N_BC[-(i+1)][j] = (dxB*dxI + dyB*dyI)/sBC
        
        '''CD'''
        dxC = xD-xC
        dyC = yD-yC
        dxJ = x[-(i+1)][j]-x[-(i+2)][j]
        dyJ = y[-(i+1)][j]-y[-(i+2)][j]
        sCD = abs(dxC*dyJ - dxJ*dyC)
        M_CD[-(i+1)][j] = (dxC**2 + dyC**2)/sCD
        N_CD[-(i+1)][j] = (dxC*dxJ + dyC*dyJ)/sCD
        
        '''AB'''
        dxD = xA-xD
        dyD = yA-yD
        dxI = x[-(i+1)][j]-x[-(i+1)][j-1]
        dyI = y[-(i+1)][j]-y[-(i+1)][j-1]
        sDA = abs(dxD*dyI - dxI*dyD)
        M_DA[-(i+1)][j] = (dxD**2 + dyD**2)/sDA
        N_DA[-(i+1)][j] = (dxD*dxI + dyD*dyI)/sDA
        
for n in range(1,nmax):
    sum = 0
    for j in range(1, grid-1):
        for i in range(1, grid-1):
            phiNX = 0.25*(N_CD[-(i+1)][j]-N_CD[-(i+1)][j])*phi[-(i+2)][j-1]
            phiNX = phiNX + (M_CD[-(i+1)][j]+0.25*(N_BC[-(i+1)][j]-N_DA[-(i+1)][j]))*phi[-(i+2)][j]
            phiNX = phiNX + 0.25*(N_BC[-(i+1)][j]-N_CD[-(i+1)][j])*phi[-(i+2)][j+1]
            phiNX = phiNX + (M_DA[-(i+1)][j] + 0.25*(N_CD[-(i+1)][j]-N_AB[-(i+1)][j]))*phi[-(i+1)][j-1]
            phiNX = phiNX + (M_BC[-(i+1)][j] + 0.25*(N_AB[-(i+1)][j]-N_CD[-(i+1)][j]))*phi[-(i+1)][j+1]
            phiNX = phiNX + 0.25*(N_DA[-(i+1)][j]-N_AB[-(i+1)][j])*phi[-i][j-1]
            phiNX = phiNX + (M_AB[-(i+1)][j] + 0.25*(N_DA[-(i+1)][j]-N_BC[-(i+1)][j]))*phi[-i][j]
            phiNX = phiNX + 0.25*(N_AB[-(i+1)][j] - N_BC[-(i+1)][j])*phi[-i][j+1]
            phiNX = phiNX / (M_AB[-(i+1)][j] + M_BC[-(i+1)][j] + M_CD[-(i+1)][j] + M_DA[-(i+1)][j])
            
            diff = phiNX - phi[-(i+1)][j]
            sum = sum + diff**2
            phi[-(i+1)][j] = phi[-(i+1)][j] + rexparam*diff
            mn_sq = sum/(nm_wx-1)
    rms = math.sqrt(mn_sq)
    if (rms<eps):
#         print("Solution converged after steps: ",n)
#         print('rms: ',rms)
        break
if (rms<eps):
    sum2 = 0
    for j in range(1,grid-1): 
        for i in range(1,grid-1):
            diff2 = phi[-(i+1)][j] - phi_x[-(i+1)][j]
            sum2 = sum2 + diff2**2
    print("Phi distribution for "+str(grid)+" x "+str(grid)+" grid \n")
    for c in range(1,grid+1):
        print('j=',c)
        print('Phi: ', phi[-c])
        print('Phi_X: ', phi_x[-c])
        print('\n')
    rms2 = math.sqrt(sum2/(nm_wx-1))
    print("Solution converged after steps: ",n)
    print('RMS: ',rms2)
else:
    print('Convergence not achieved')  


fig, ax1 = plt.subplots()
fig, ax2 = plt.subplots()
cp1 = ax1.contourf(x,y,phi_x,levels=10)
cp2 = ax2.contour(x,y,phi_x,levels=10,cmap='hot')
ax1.clabel(cp2,inline = True, fontsize = 7)
plt.title(label)
plt.show()

            