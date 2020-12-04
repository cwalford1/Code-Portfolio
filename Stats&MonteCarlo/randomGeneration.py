import numpy as np 
import matplotlib.pyplot as plt

"""About this task:
In this file I'm attempting to generate random numbers with a probability desnity function (pdf) of (1/3)(1+4x).
This is non-trivial as modules like numpy allow for generation of random numbers on a few set distributions, but not any arbitrary distribution.
However, we can use random numbers on a (0,1) uniform distribution to generate the distribution we want.
This pdf is well enough behaved that we can directly calculate the inverse cdf.  
With this in hand we need only pass in numbers generated on the uniform distribution and the output will be numbers on our pdf.
A graphical comparison of the pdf and a histogram of our generated data is shown at the end of calculation to check how closely our data matches the pdf.
"""

#generate random numbers
Ns=np.random.uniform(0,1,10000)
#pass through inverse cdf 
xs=np.sqrt(1+24*Ns)-1
xs*=0.25
#plot histogram of generated xs
fig=plt.figure()
ax1=fig.add_subplot(111)
ax1.hist(xs,bins=10)
#generate new xs array that will be points for line
xs=np.linspace(0,1,100)
#pass into pdf (scaled by 1000 because initial pdf was normalized but histogram has area of Npoints*binwidth=10000*0.1=1000)
ys=(1+4*xs)*(1/3)*1000
ax1.plot(xs,ys,label='f(x)=(1/3)(1+4x)')
ax1.legend()
ax1.set_title('Comparison of pdf and generated data')
plt.show()
