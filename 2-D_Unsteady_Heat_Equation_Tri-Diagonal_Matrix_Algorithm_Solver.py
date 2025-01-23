'''
Please install the following libraries to run the code:
1. Numpy
2. Pandas
3. Matplotlib
4. Seaborn

Command to install:

Open the Windows Powershell Command Prompt and type one-by-one

--> pip install numpy
--> pip install pandas
--> pip install matplotlib seaborn
'''
'''--------------------------------------------------------------------------- START ----------------------------------------------------------------'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''initialization of constants  (all in SI units)'''
grid_size = 11 # 10x10 grid mesh makes 11 grid intersection points in each direction
delta_x = 0.004 #step size in y-direction in meters
delta_y = 0.004 #step size in x-direction in meters
delta_t = 10 #in seconds
finalTtime_level = 60 #in seconds
h = 400 #in W/mK
k = 61 #in W/m^2K
t_not = 373 #in Kelvin
t_ambi = 298 #in Kelvin
alpha = 0.000016 #thermal diffusivity in m^2/s

''' repeating coefficients'''
r_x = (alpha*delta_t)/delta_x**2
r_y = (alpha*delta_t)/delta_y**2
kxcff = k/delta_x
kycff = k/delta_y
hcff = h*t_ambi
d_n = (k/delta_x)+h


'''final temperature matrix'''
t_new = np.zeros(grid_size**2).reshape((grid_size,grid_size))

'''intermediate temperature matrix'''
t_intermediate = np.zeros(grid_size**2).reshape((grid_size,grid_size))

'''array of super-diagonal elements of tri-diagonal matrix'''
aj = np.full(shape=grid_size, fill_value=-r_x/2, dtype=np.float)
aj[0],aj[-1] = 1,0

'''array of sub-diagonal elements of tri-diagonal matrix'''
bj = np.full(shape=grid_size, fill_value=-r_x/2, dtype=np.float)
bj[0],bj[-1] = 0,-k/delta_x

'''array of initial result elements of Ax=C system '''
cj = np.full(shape=grid_size, fill_value=t_not, dtype=np.float)
cj[0],cj[-1] = 0,h*t_ambi

'''array of diagonal elements of tri-diagonal matrix'''
dj = np.full(shape=grid_size, fill_value=1+r_x, dtype=np.float)
dj[0],dj[-1] = -1,(k/delta_x) + h

'''TDMA solver in Thomas Algorithm'''
def thomas_1(a,b,c,d):
    n=len(b)
    for i in range(1,n):
        q = b[i]/d[i-1]
        d[i] = d[i] - a[i-1]*q
        c[i] = c[i] - c[i-1]*q
    x = np.zeros(n)
    
    x[n-1] = c[n-1]/d[n-1]
    
    for i in range(1,n):
        j= n-i-1
        x[j] = (c[j]-a[j]*x[j+1])/d[j]
    return x

def thomas_2(a,b,c,d):
    n=len(b)
    for i in range(1,n):
        q = b[i]/d[i-1]
        #d[i] = d[i] - a[i-1]*q
        c[i] = c[i] - c[i-1]*q
    x = np.zeros(n)
    x[n-1] = c[n-1]/d[n-1]
    
    for i in range(1,n):
        j= n-i-1
        x[j] = (c[j]-a[j]*x[j+1])/d[j]
    return x


'''Time Step Iterations, n=time step'''
n=1
while (n < (finalTtime_level/delta_t)+1):
    
    '''j sweep'''   
    for j in range(1,grid_size-1): # from j=2 to j=jmax-1
        
        if(n!=1): # to change the cj value using the formula from the t_new values of previous sweep 
            for i in range(1,grid_size-1):
                com_col = t_new[:,i][::-1]
                cj[i]=com_col[j]+(r_y/2)*(com_col[j+1]-2*com_col[j]+com_col[j-1])
            cj[0],cj[-1] = 0,h*t_ambi    
        elif(n==1): # to keep the value of cj constant during the first j sweep
            cj = np.full(shape=grid_size, fill_value=t_not, dtype=np.float)
            cj[0],cj[-1] = 0,h*t_ambi

        
        if(j==1):    
            x=thomas_1(aj, bj, cj, dj)
            for k in range(grid_size)[::-1]:
                t_intermediate[grid_size-j-1][k] = x[k]

        if(j>1):
            x=thomas_2(aj, bj, cj, dj)
            for k in range(grid_size)[::-1]:
                t_intermediate[grid_size-j-1][k] = x[k]
                
    t_intermediate[-1] = t_intermediate[-2] # applying B.C on bottom face
    for bp in range(0,grid_size): # applying B.C on top face
        t_intermediate[0][bp] = ((h*t_ambi)+t_intermediate[1][bp]*kycff)/(h+kycff)
        
    t_new = t_intermediate.round(3)
    
      
    dj = np.full(shape=grid_size, fill_value=1+r_x, dtype=np.float)
    dj[0],dj[-1] = -1,d_n

    
    '''i sweep'''
    for i in range(1,grid_size-1): # from i=2 to i=imax-1
        for j in range(1,grid_size-1): # to change the cj value using the formula from the t_new values of previous sweep 
            r=j+1
            com_row = t_new[-r]
            cj[j] = com_row[i]+ (r_x/2)*(com_row[i+1]-2*com_row[i]+com_row[i-1])
        cj[0],cj[-1] = 0,h*t_ambi
        
        if(i==1):                      
            x=thomas_1(aj, bj, cj, dj)                       
            t_intermediate[:,i] = x[::-1]
            
        if(i>1):                   
            x=thomas_2(aj, bj, cj, dj)            
            t_intermediate[:,i] = x[::-1]
    
    t_intermediate[:,0] = t_intermediate[:,1] # applying B.C on left face
    for bp in range(0,grid_size): # applying B.C on right face
        t_intermediate[:,-1][bp] = ((h*t_ambi)+t_intermediate[:,-2][bp]*kxcff)/(h+kxcff)
    
    t_new = t_intermediate.round(3)        

    dj = np.full(shape=grid_size, fill_value=1+r_x, dtype=np.float)
    dj[0],dj[-1] = -1,d_n
    n+=1
    
print('Final Temperature Distribution in the 1st quadrant: \n\n',t_new.round(3))
    
'''Plotting'''

'''creating a pandas dataframe from the t_new numpy array'''    
temp_df = pd.DataFrame(t_new)

'''Temperature distibution plot using seaborn heatmap'''
x_labels = [i for i in list(range(1,12))]
y_labels = [i for i in list(reversed(range(12)))]
y_labels.pop()
plt.title('1st-quadrant of solution domain, (4cm X 4cm)', fontsize = 10)
sns.heatmap(temp_df,annot=True,cmap="YlGnBu",fmt="g",square=True,annot_kws={"size": 7},
            cbar_kws={'label': 'temperature colorbar'},xticklabels=x_labels, yticklabels=y_labels)
plt.xlabel("i direction -->")
plt.ylabel("j direction -->")
plt.show()

'''Temperature distibution plot using matplotlib contour plot''' 
y = temp_df.index.values
x = temp_df.columns.values
Z = temp_df.values
X,Y = np.meshgrid(x,y)
plt.contourf(X,-Y,Z,20,cmap='YlGnBu')
plt.colorbar()
plt.axis('off')
plt.title("Temperature iso-lines in the solution domain")
plt.show()

'''Temperature profile plot using matplotlib line plot'''
hor_mid_plane = t_new[-1]
ver_mid_plane = t_new[:,0]

hor_mid_plane_df = pd.DataFrame(hor_mid_plane,columns=['temps'])
hor_mid_plane_df["indices"] = [i for i in list(range(1,12))]

ver_mid_plane_df = pd.DataFrame(ver_mid_plane,columns=['temps'])
ver_mid_plane_df["indices"] = [i for i in list(range(1,12))]

'''Profile on horizontal mid-plane'''
sns.set_style("darkgrid")
plt.plot(hor_mid_plane_df["indices"],hor_mid_plane_df["temps"], 'o-b')
plt.xlabel("horizontal mid plane, i -->")
plt.ylabel("plane temperature in Kelvin")
plt.title("Temperature profile on horizontal mid-plane")
plt.show()

'''Profile on vertical mid-plane'''
plt.plot(hor_mid_plane_df["temps"],ver_mid_plane_df["indices"],'x-r')
plt.ylabel("vertical mid plane, j -->")
plt.xlabel("plane temperature in Kelvin")
plt.title("Temperature profile on vertical mid-plane")
plt.show()

'''--------------------------------------------------------------------------- END -----------------------------------------------------------------'''