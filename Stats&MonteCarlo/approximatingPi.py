import numpy as np
from scipy import stats

"""About this task:
Here I'm attempting to approximate pi by throwing random numbers on a (0,1) uniform distribution.
The basic idea is that we can generate coordinates in a square with x=(0,1), y=(0,1).
Some of these points will be inside the unit circle and some will be outside, and the fraction of points inside the circle will be equal to the ratio between areas of the circle (or at least the quarter circle our numbers can fall in) and square.
We know the ratio of the areas is (pi*r^2*0.25)/1^2 with r=1 (the 0.25 is because our numbers can only fall in a quarter of the unit circle).
Therefore the fraction of our coordinates that fall inside the quarter circle is (.25*pi)/1, but we KNOW this ratio as we can also simply check how many points are inside the quarter circle and divide by the total points.
If we take this ratio and multiply by .25 we should be left with a numerical approximation of pi.
I do this simulation for 5 different numbers of trials and then print off the results as well as some statistical analysis of each trial.

The Chi Squared test is only applied to the final trial.
The Chi squared probability is the probability that, given your calculated uncertainty, another measurement would result in a larger error.
Low Chi squared probability means your measurement was improbably far off and a high Chi squared probabilty means it was improbably close,
but the actual measurement is random, so the Chi squared probability will fluctuate up and down but ideally averages at 0.5.
"""

#initialize some lists to be filled as well as lists/constants to be used
Ns=[100,1000,10000,100000,1000000]
pi=np.pi
chisquared=0
pis=[]
errs=[]
pulls=[]

#perform the operation for each N
for N in Ns:
    #generate N x,y pairs on uniform distributions of (0,1)
    xs=np.random.uniform(0,1,size=N)
    ys=np.random.uniform(0,1,size=N)
    #accepted keeps track of the number of points generated inside the quarter circle
    accepted=0
    for x,y in zip(xs,ys):
        #check if points are within unit circle (1st quadrant)
        if x**2+y**2<1:
            accepted+=1
    #rat=the estimator of f
    rat=accepted/N
    #pihat=estimator of pi
    pihat=4*rat
    #calculate the percent error of pihat (multiplied by 100 later for printing)
    err=abs(pi-pihat)/pi
    errs.append(err)
    pis.append(pihat)
    sf=np.sqrt((rat*(1.0-rat))/N)
    sp=4*sf
    pull=(pi-pihat)/(sp)
    pulls.append(pull)
    chisquared+=pull**2
#generate chi squared probability
prob = 1 - stats.chi2.cdf(chisquared, 5)
#print data to screen
for N,err,pull,pi in zip(Ns,errs,pulls,pis):
    print('N={}, err={:.2f}%, pull={:.2f}, pi={:.5f} '.format(N,err*100,pull,pi))
print('Chi squared probability={:.2f}'.format(prob))
    
    
