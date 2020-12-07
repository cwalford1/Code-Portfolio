import numpy as np 
import matplotlib.pyplot as plt 

"""About this task:
When a sound wave transitions from one material to another some of the wave is reflected and some is transmitted.
This is a simple visual representation of the relative tranmission/reflection that occurs when sound transitions to various materials.
There's not a ton going on here, but the plot is visually appealing and conveys the information effectively.
"""

#functions to calculate reflection and transmission coefficients
def R(vs,vsp):
    return np.square((vsp-vs)/(vsp+vs))

def T(vs,vsp):
    return 1-R(vs,vsp)

#potentials of various gasses (and steel)
othervs=np.array([1007,221,178,1270,201,1482,5960,345,446])
gasses=np.array(['He','Kr','Xe','H','SO2','H20','Steel','Air','CH4'])
#initial potential and range of new potentials
vs=345
vsps=np.linspace(1,6000,201)

#create plots
fig,ax=plt.subplots()
ax.plot(vsps/vs,R(vs,vsps),label='R')
ax.plot(vsps/vs,T(vs,vsps),label='T')
ax.legend(fontsize='x-large')
ax.scatter(othervs/vs,R(vs,othervs))
ax.scatter(othervs/vs,T(vs,othervs))
#label points
for gas,v in zip(gasses,othervs):
    ax.annotate(gas,(v/vs,R(vs,v)),fontsize='large')
    ax.annotate(gas,(v/vs,T(vs,v)),fontsize='large')
#plot labels & formatting
ax.set_xlabel('v\'/v',fontsize='xx-large')
ax.set_ylabel('R or T',fontsize='xx-large')
ax.set_title('Reflection and Transmission of Sound at a Step',fontsize='xx-large')
plt.grid()

plt.show()
