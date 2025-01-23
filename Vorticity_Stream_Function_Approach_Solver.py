'''
Created on 11-Nov-2020

@author: PAUL AKASH
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from copy import copy
import sys
np.set_printoptions(threshold=sys.maxsize, suppress=True)

new = 0.00025 #kinematic viscosity
u0 = 5 #in m/s
v0 = 0 #in m/s
imax = 41
jmax = 41
delta_x = 0.01 #in m
delta_y = 0.01 #in m
j1 = 20
j2 = 25
j3 = 5
j4 = 10
lgt = 0.4
hgt = 0.4
rexparam = 1.4 # relaxation parameter
si1 = 0.25 
err = 0.00001 #minimum error value

print('The results are for the values: (in SI units)')
print('\n')
print('kinematic viscosity: ',new)
print('u0: ',u0)
print('v0: ',v0)
print('imax: ',imax)
print('jmax: ',jmax)
print('delta_x: ',delta_x)
print('delta_y: ',delta_y)
print('j1: ',j1)
print('j2: ',j2)
print('j3: ',j3)
print('j4: ',j4)
print('length in x direction: ',lgt)
print('length in y direction: ',hgt)
print('over-relaxation parameter: ',rexparam)
print('minimum error: ',err)
print('\n')

u = np.full(shape=imax*jmax , fill_value=u0, dtype=np.float64).reshape(imax,jmax)
v = np.zeros((imax,jmax), dtype=np.float64)
si = np.zeros((imax,jmax),dtype=np.float64)
w = np.zeros((imax,jmax), dtype=np.float64) #vorticity

''' initial si on interior points (i sweep) '''
for i in range(1,imax-1):
    for j in range(1,jmax-1):
        si[-(i+1)][j] = i*delta_y*u[-(i+1)][j]
        
        
'''            applying BCs for face AB           '''
for j in range(1,jmax+1):
    if(j<j1):
        u[-j][0] = 0
        v[-j][0] = 0
        si[-j][0] = 0
        w[-j][0] = -2*(si[-j][1] - si[-j][0])/(delta_x**2)
    ''' inflow part j1->j2'''
    if(j>=j1 and j<=j2):
        u[-j][0] = u0
        v[-j][0] = 0
        w[-j][0] = 0
        si[-j][0] = (j-j1)*delta_y*u[-j][0]
    if(j>j2):
        u[-j][0] = 0
        v[-j][0] = 0
        si[-j][0] = si1
        w[-j][0] = -2*(si[-j][1] - si[-j][0])/(delta_x**2)
        
        
'''            applying BCs for face BC           '''
u[-1] = 0
v[-1] = 0
si[-1] = 0
for i in range(1,imax):
    w[-1][i-1] = -2*(si[-2][i-1] - si[-1][i-1])/(delta_y**2)
    

'''            applying BCs for face CD           '''
for j in range(1,jmax+1):
    if(j<=j3):
        u[:,-1][-j] = 0
        v[:,-1][-j] = 0
        si[:,-1][-j] = 0
        w[:,-1][-j] = -2*(si[:,-2][-j] - si[:,-1][-j])/(delta_x**2)
    ''' outflow part j3->j4'''
    if(j>j3 and j<=j4):
        v[:,-1][-j] = 0
        si[:,-1][-j] = 2*si[:,-2][-j] - si[:,-3][-j]
        w[:,-1][-j] = w[:,-2][-j]
        u[:,-1][-j] = 0
    if(j>j4):
        u[:,-1][-j] = 0
        v[:,-1][-j] = 0
        si[:,-1][-j] = si1
        w[:,-1][-j] = -2*(si[:,-2][-j] - si[:,-1][-j])/(delta_x**2)
    

'''            applying BCs for face DA           '''
u[0] = 0
v[0] = 0
si[0] = si1
for i in range(1,imax):
    w[0][i-1] = -2*(si[1][i-1] - si[0][i-1])/(delta_y**2)
w[0][0] = w[0][1]
w[-1][0] = w[-1][1]
w[:,-1][-1] = w[:,-2][-1]
w[:,-1][0] = w[:,-2][0]
    
si_int = copy(si)

'''Iteration begins....'''
n=1
while (n>=1):
    """         calculating si at (k+1)'        """
    for j in range(1,jmax-1):
        for i in range(1,imax-1):
            si[-(i+1)][j] = 0.25*(si[-(i+1)][j-1] + si[-i][j] + si[-(i+1)][j+1] + si[-(i+2)][j] + w[-(i+1)][j]*delta_x**2)
            ''' using over-relaxation method '''
            si[-(i+1)][j] = si_int[-(i+1)][j] + rexparam*(si[-(i+1)][j] - si_int[-(i+1)][j])    # using over-relaxation method
            
    
    '''         calculating u and v          '''
    for j in range(1,jmax-1):
        for i in range(1,imax-1):
            u[-(i+1)][j] = (si[-(i+2)][j] - si[-i][j])/(2*delta_y)
            v[-(i+1)][j] = -(si[-(i+1)][j+1] - si[-(i+1)][j-1])/(2*delta_x)
    
    '''         calculating w          '''
    w2 = copy(w)
    
    '''        by applying 1st-order upwinding for u and v         '''
    
    for j in range(1,jmax-1):
        for i in range(1,imax-1):
            
            newx = new/delta_x
            ul = newx - u[-(i+1)][j]
            ur = newx + u[-(i+1)][j]
            vl = newx - v[-(i+1)][j]
            vr = newx + v[-(i+1)][j]           
            
            if(u[-(i+1)][j]>0 and v[-(i+1)][j]>0):
                w[-(i+1)][j] = (ur*w2[-(i+1)][j-1] + vr*w2[-i][j] + newx*w2[-(i+1)][j+1] + newx*w2[-(i+2)][j])/(u[-(i+1)][j] + v[-(i+1)][j] + 4*newx)
            elif(u[-(i+1)][j]>0 and v[-(i+1)][j]<0):
                w[-(i+1)][j] = (ur*w2[-(i+1)][j-1] + newx*w2[-i][j] + newx*w2[-(i+1)][j+1] + vl*w2[-(i+2)][j])/(u[-(i+1)][j] - v[-(i+1)][j] + 4*newx)
            elif(u[-(i+1)][j]<0 and v[-(i+1)][j]>0):
                w[-(i+1)][j] = (newx*w2[-(i+1)][j-1] + vr*w2[-i][j] + ul*w2[-(i+1)][j+1] + newx*w2[-(i+2)][j])/(-u[-(i+1)][j] + v[-(i+1)][j] + 4*newx)
            elif(u[-(i+1)][j]<0 and v[-(i+1)][j]<0):
                w[-(i+1)][j] = (newx*w2[-(i+1)][j-1] + newx*w2[-i][j] + ul*w2[-(i+1)][j+1] + vl*w2[-(i+2)][j])/(-u[-(i+1)][j] - v[-(i+1)][j] + 4*newx)
                
    
    '''          applying BCs for w on all faces as si value is updated         '''       
    for j in range(1,jmax+1):
        if(j<j1):
            w[-j][0] = -2*(si[-j][1] - si[-j][0])/(delta_x**2)
        if(j>=j1 and j<=j2):
            w[-j][0] = 0
            si[-j][0] = (j-j1)*delta_y*u[-j][0]
        if(j>j2):
            w[-j][0] = -2*(si[-j][1] - si[-j][0])/(delta_x**2)
    for i in range(1,imax):
        w[-1][i-1] = -2*(si[-2][i-1] - si[-1][i-1])/(delta_y**2)
    for j in range(1,jmax+1):
        if(j<=j3):
            w[:,-1][-j] = -2*(si[:,-2][-j] - si[:,-1][-j])/(delta_x**2)
        if(j>j3 and j<=j4):
            w[:,-1][-j] = w[:,-2][-j]
        if(j>j4):
            w[:,-1][-j] = -2*(si[:,-2][-j] - si[:,-1][-j])/(delta_x**2)
    for i in range(1,imax):
        w[0][i-1] = -2*(si[1][i-1] - si[0][i-1])/(delta_y**2)
    w[0][0] = w[0][1]
    w[-1][0] = w[-1][1]
    w[:,-1][-1] = w[:,-2][-1]
    w[:,-1][0] = w[:,-2][0]
    
    
    '''         caculating the rms value of error and the number of iterations for it to fall below 0.00001        '''
    sum = 0
    for j in range(1,jmax+1):
        for i in range(1,imax+1):
            diff = si_int[i-1][j-1] - si[i-1][j-1]
            sum+=diff**2
    rms = math.sqrt(sum/(imax))
    print('rms: ',rms)
    
    if (rms <= err):
        break
    
    si_int = copy(si)
    n+=1

'''           printing the final results           '''   
print('n: ',n)
print('u: ',u)
print('v: ',v)
print('Final si: ',si)
print('rms: ',rms)
print('number of iterations to reach rms value of error < '+str(err)+': ',n)


'''          plotting the stream-lines in the solution domain         '''
x = np.linspace(0,0.4,imax)
y = np.linspace(0.4,0,jmax)
X, Y = np.meshgrid(x, y)
plt.contour(X,Y,si,levels=np.linspace(-0.2,0.45,260))
plt.title("Stream-lines in the solution domain")
plt.colorbar()
plt.xlabel('X ->')
plt.ylabel('Y ->')
plt.show()

'''          plotting the velocity field in the solution domain         '''
plt.quiver(x,y,u,v,color='Blue')
plt.title("velocity field in the solution domain")
plt.show()


