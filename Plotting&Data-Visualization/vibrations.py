import numpy as np 
import matplotlib.pyplot as plt 

"""About this task:
This file is exploring the conceptually simple situation of a plucked string vibrating.  
I numerically solve the wave equation that describes the motion of this string under a few different conditions and plot the results on a contour map.

I won't go into the specific mathematical background of the numerical solution used, 
but the general idea is that you break up the whole situation into discrete time/position coordinates.  
A desired time/position coordinate can be calculated by looking at what this coordinate and it's neighbors were like one timestep ago.
We then begin with some starting condition (a list of values marking the amplitude of our string at each x position) and feel our way forward in time from there.

What you see when you run this file is the amplitude of the string plotted against time and position under these conditions.  
These plots are not only visually beautiful, but are so densely packed with insights and information about the physical situation that you can spend hours staring at them and have new and interesting observations to make.
I'll say a little about each graph here:

1. this graph is the base case of a string plucked at x=0.8 m t=0 s with with pluck amplitude of 2 cm.  The string is modelled to be anchored at its ends x=0 and x=1.
If you pick an x position and move vertically across the graph you see that this position on the string bobs up and down (just as we'd expect from a vibrating string).
We can also see how the wave bounces off the right endpoint of the string and begins interfering with the rest of the string's motion.
The slope of these V shaped patterns represent the velocity of the wave propogating through the string.

2. This is a look into the the shortcomings of the particular numerical method used to solve the wave equation.  
If you chose the sizes of your time and position steps wrong you end up with unstable solutions that quickly give ridiculous values.
The scale of these contours indicate that our string would be vibrating to heights on the order of 10^216 meters.

3. Here we see what happens when a frictional term is added to the solution which damps our strings vibrations.
Notice how the initial pluck produces a wave that propogates a short distance (the V figure) and there are some reverberations in this region, 
but after some time all vibrations die out and the string returns to rest.

4. Here we explore how varrying the tension (alpha_t) and density (alpha_rho) across it's length affects the mechanics of vibrations.
In this graph we change these two variables together and the result is that they cancel out and produce a graph identically to the base case from 1.

5. Here, however, we ramp up tension faster than density and observe a significant departure from normal behavio such as changing wave velocity.

6. In this graph, we explore what happens when the right end of the string is driven up and down in addition to the initial pluck.
This is another grahp one can stare at all day imagining shaking a string on one end and visualizing how different parts of the string will bob up and down.
The timescale is much longer (10x) here than in the past graphs so we also get to see the initial wave from the pluck bounce off the left edge of the string.

I love these graphs because they take a super intuitive situation (a vibrating string) and just freeze every little aspect of it for you to dive into and explore at your leisure.
"""

parts='abcdef'
#setting up parameters
#maximum time is set to a low value, because on larger timescales the wave characteristics become hard to see.
tmax=.01
L=1
T=40
rho=0.01
#time and position stepsize dx/dt=100
dx=.01
dt=.0001
#total steps
nxs=int(L/dx)
nts=int(tmax/dt)
#vectors to be put together to form lattice of x,t space to plot y values
xs=np.linspace(0,1,nxs+1)
ts=np.linspace(0,tmax,nts+1)
#vectors that hold value of tension and rho at all points on x.  Constant now but changes later
Ts=np.repeat(T,np.shape(xs))
rhos=np.repeat(rho,np.shape(xs))

#create x,t space and fill the first time row with a perterbation at x=.8m
lattice,a=np.meshgrid(xs,ts)
x0=np.zeros(np.shape(xs))
x0[80]=.02
ys=np.zeros(np.shape(lattice))
ys[0]=x0

#function that returns v^2/v'^2 important in time stepping algorithm
def getv2(i,Tvec,rhovec):
    v=np.sqrt(Tvec[i]/rhovec[i])
    vprime=dx/dt
    v2=np.square(v)/np.square(vprime)
    return v2
#function that returns x values for first time step.  Seperate function required because the time stepping algorithm requires two time steps back, but a new algorithm can be used that 
#implements the 0 derivative at t=0
def firstStep(vec1,Tvec,rhovec,end=0,Fric=False):
    #nextvec is the vector of x values at the t value one time stap ahead of the input vec1.  The left hand side of our string is fixed at 0 amplitude, so the first entry is always 0.
    nextVec=np.array([0])
    for i in range(1,len(vec1)-1):
        #new is the value of amplitude after i timesteps.
        new=vec1[i]+.5*getv2(i,Tvec,rhovec)*(vec1[i+1]+vec1[i-1]-2*vec1[i])
        nextVec=np.append(nextVec,new)
        #Optional friction term, set to 0 because dy/dt at the first step is definitionally 0.  Included for symmetry with generalized timestep algorithm.
        if Fric:
            new-=0
        #appends the value of y at right hand end of string.  Constant 0 by defualt but can be changed for part f.
    nextVec=np.append(nextVec,end)
    return nextVec
#identical to firstStep except for equation for new and Frictional term
def nextStep(vec0,vec1,Tvec,rhovec,end=0,Fric=False):
    nextVec=np.array([0])
    
    for i in range(1,len(vec1)-1):
        #algorithm for time step given last two vectors of x values.
        new=2*vec1[i]-vec0[i]+getv2(i,Tvec,rhovec)*(vec1[i+1]+vec1[i-1]-2*vec1[i])
        
        if Fric:
            #frictional force subtracted from change in y value.  dx/dt defined by difference in last two points.  It doesn't really make
            #physical sense to subtract a force from an amplitude, so this is incorrect, but I don't know where else to implement the term.
            #Post grading comment: I received full marks on this assignment, so apparently this was correct.
            new-=2*k*dx*(vec1[i]-vec0[i])/dt
        nextVec=np.append(nextVec,new)
    nextVec=np.append(nextVec,end)
    return nextVec

#solution for part a
#the plot produced is a fairly standard looking solution to the wave equation.  I find the reflected wave from the right edge causing interference particularly interesting.
if 'a' in parts:
    #get first set of x values from first step algorithm
    ys[1]=firstStep(ys[0],Ts,rhos)
    for i in range(2,nts+1):
        #repeat stepping algorithm until max time
        ys[i]=nextStep(ys[i-2],ys[i-1],Ts,rhos)
    #plotting
    fig,ax=plt.subplots()
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Magnitude of Wave: Part a (m)')
    h = plt.contourf(xs,ts,ys)
    fig.colorbar(h, ax=ax)
    plt.show()
#there is a lot of copy paste from problem to problem that probably could have been cut out with some refinement, but bare with me.
#b) dx/dt>v produces stable solutions, while if dx/dt<v the peaks of the waves grow exponentially.
#for example

tmax=0.1
#the only change from part a is dx/dt=10 rather than 100 to showcase unstable solution
dx=.01
dt=0.001
#dx/dt=10<v~=63.2

nxs=int(L/dx)

nts=int(tmax/dt)

xs=np.linspace(0,1,nxs+1)
ts=np.linspace(0,tmax,nts+1)

lattice,a=np.meshgrid(xs,ts)
x0=np.zeros(np.shape(xs))
x0[80]=.02

ys=np.zeros(np.shape(lattice))
ys[0]=x0
#The produced plots have exponentially growing values for the amplitude indicated by the enormous scale of the colorbar
if 'b' in parts:
    ys[1]=firstStep(ys[0],Ts,rhos)
    for i in range(2,nts+1):
        ys[i]=nextStep(ys[i-2],ys[i-1],Ts,rhos)
    fig,ax=plt.subplots()
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Magnitude of Wave: Part b (m)')
    h = plt.contourf(xs,ts,ys)
    fig.colorbar(h, ax=ax)
    plt.show()

#c) From the plot produced in part a we can see that the wave seems to propogate at arounnd 60 m/s which is roughly the value of v=sqrt(T/rho)=63.2
#this value is obtained from the slope of the leading edge of wave propagation.

#d)
#the only change from a is turning on the optional Fric term in time step algorithms.
tmax=.01
L=1
T=40
rho=0.01

dx=.01
dt=.0001

nxs=int(L/dx)

nts=int(tmax/dt)

xs=np.linspace(0,1,nxs+1)
ts=np.linspace(0,tmax,nts+1)
Ts=np.repeat(T,np.shape(xs))
rhos=np.repeat(rho,np.shape(xs))


lattice,a=np.meshgrid(xs,ts)
x0=np.zeros(np.shape(xs))
x0[80]=.02

ys=np.zeros(np.shape(lattice))
ys[0]=x0

k=0.001
#again, I don't think these are right, but the waves do die off which is consistent with the addition of a frictional force
if 'd' in parts:
    ys[1]=firstStep(ys[0],Ts,rhos,Fric=True)
    for i in range(2,nts+1):
        ys[i]=nextStep(ys[i-2],ys[i-1],Ts,rhos,Fric=True)
    fig,ax=plt.subplots()
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Magnitude of Wave: Part d (m)')
    h = plt.contourf(xs,ts,ys)
    fig.colorbar(h, ax=ax)
    plt.show()

#e) if rho and T have the same functional dependance on x then there is no change in the wave behavior as the only place these parameters
#show up in our algorithm is as a ratio.  However, if the parameter alpha in the functions for rho and T are different, then we observe curving peak of the wave.

#following code is identical to part a except for the functional dependance of rho and T on x.

#here the alpha term in both functions are the same
alpha_r=1
alpha_T=1
tmax=0.01
dx=.01
dt=.0001
nxs=int(L/dx)

nts=int(tmax/dt)

xs=np.linspace(0,1,nxs+1)
ts=np.linspace(0,tmax,nts+1)
rhos=rho*np.exp(alpha_r*xs)

Ts=T*np.exp(alpha_T*xs)


lattice,a=np.meshgrid(xs,ts)
x0=np.zeros(np.shape(xs))
x0[80]=.02

ys=np.zeros(np.shape(lattice))
ys[0]=x0
#plot for equal alphas is identical to the plot in part a as expected
if 'e' in parts:
    ys[1]=firstStep(ys[0],Ts,rhos)
    for i in range(2,nts+1):
        ys[i]=nextStep(ys[i-2],ys[i-1],Ts,rhos)
    fig,ax=plt.subplots()
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Magnitude of Wave: Part e.  alpha_T=alpha_rho (m)')
    h = plt.contourf(xs,ts,ys)
    fig.colorbar(h, ax=ax)
    plt.show()

#the following code does the same as above but with differnt alpha values for rho and T
alpha_r=.1
alpha_T=1
tmax=0.01
dx=.01
dt=.0001
nxs=int(L/dx)

nts=int(tmax/dt)

xs=np.linspace(0,1,nxs+1)
ts=np.linspace(0,tmax,nts+1)
rhos=rho*np.exp(alpha_r*xs)

Ts=T*np.exp(alpha_T*xs)


lattice,a=np.meshgrid(xs,ts)
x0=np.zeros(np.shape(xs))
x0[80]=.02

ys=np.zeros(np.shape(lattice))
ys[0]=x0
#plots for alpha_T=10*alpha_rho
#here we observe a curved drifting of the peaks centered at 0.8m which were static before
if 'e' in parts:
    ys[1]=firstStep(ys[0],Ts,rhos)
    for i in range(2,nts+1):
        ys[i]=nextStep(ys[i-2],ys[i-1],Ts,rhos)
    fig,ax=plt.subplots()
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Magnitude of Wave: Part e.  alpha_T=10*alpha_rho (m)')
    h = plt.contourf(xs,ts,ys)
    fig.colorbar(h, ax=ax)
    plt.show()


#f)A standing wave with a node at one end and antinode at the other would require that the wave be a quarter period father along in its cycle than a
# typical standing wave with nodes at both endpoints which has the equation for frequencies of n*pi/L so perhaps a period of (n*pi/L+pi/4) would work.
#parameters for calculation of right edge sine wave
A=0.02
omega=100*np.pi+np.pi/4

tmax=.1
L=1
T=40
rho=0.01

dx=.01
dt=.0001
nxs=int(L/dx)

nts=int(tmax/dt)

xs=np.linspace(0,1,nxs+1)
ts=np.linspace(0,tmax,nts+1)
#y0 is the vector of right edge values of y passed into the timestep algorithm
y0=A*np.sin(omega*ts)

Ts=np.repeat(T,np.shape(xs))
rhos=np.repeat(rho,np.shape(xs))


lattice,a=np.meshgrid(xs,ts)
x0=np.zeros(np.shape(xs))
x0[80]=.02

ys=np.zeros(np.shape(lattice))
ys[0]=x0
#plotting for part f
if 'f' in parts:
    ys[1]=firstStep(ys[0],Ts,rhos,end=y0[1])
    for i in range(2,nts+1):
        ys[i]=nextStep(ys[i-2],ys[i-1],Ts,rhos,end=y0[i])
    fig,ax=plt.subplots()
    ax.set_xlabel('x position (m)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Magnitude of Wave: Part f (m)')
    h = plt.contourf(xs,ts,ys)
    fig.colorbar(h, ax=ax)
    plt.show()
#This plot seems to exibit standing waves, as when traveling along the T axis there are regular periods of rising and falling wave amplitude with relatively unmoving peaks
#This behavior particularly strong early on in time evolution perhaps due to the inital perturbation.
