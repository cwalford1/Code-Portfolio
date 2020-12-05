import numpy as np 
from timeit import timeit
"""About this task:
This is a timing experiment to compare simple Monte-Carlo integration to that using an antithetic variable of integration.
The motivation here is that the uncertainty MC integration is inversely proportional to the square root of the number of trials (σI=σf/√N),
so you get diminishing returns on uncertainty reduction by calculating more trials.
The method of antithetic variables is a way to adjust reduce σf directly.

This file calculates the integral I=∫1/(1+x)dx from 0 to 1 with both MC integration methods and compares their speed/accuracy.
Antithetic MC integration turns out to take slightly longer for the same number of trials, but saves massively on time if you increase the number simple MC integration trials in order to achieve the same uncertainty.
"""

#a) exact value of integral is ln(2)~0.69315
true=np.log(2)
#b)
N=1500

xs=np.random.uniform(0,1,2*N)
#function to be integrated
def f(x):
  return 1.0/(1.0+x)
#pass random numbers on uniform distribution to function and calculate average values (i.e.) perform MC integration
answer1=np.mean(f(xs))
#error is stanrdard deviation of the mean of trials
uncertainty1=np.std(f(xs))/np.sqrt(2*N)

#c)
#produce new set of 1500 random numbers on uniform distribution
xs=np.random.uniform(0,1,N)
#produce antithetic variable set
axs=1-xs
#perform MC integration with both sets
results=np.mean(f(xs))
aresults=np.mean(f(axs))
#find average of both integrations
answer2=(np.mean(results)+np.mean(aresults))/2
#calculate variance of estimator
cov=(np.cov(f(xs),f(axs)))
#variance of combined estimator is (Var(results)+Var(aresults)+2Cov(results,aresults))/4 which is merely the sum of all elements of the covariance matrix divided by 4
var=np.sum(cov)/4
#sd=sqrt(var)
#again SDM=SD/sqrt(N)
uncertainty2=np.sqrt(var)/np.sqrt(2*N)
#print out results of 3 integration methods
print('True value of integral = {}'.format(true))
print('Simple MC integral = {} with error of {} and discrepency of {}.'.format(answer1,uncertainty1,abs(answer1-true)))
print('Antithetic MC integral = {} with error of {} and discrepency of {}.'.format(answer2,uncertainty2,abs(answer2-true)))

#timing of two methods
t1=timeit('''import numpy as np 
N=1500
xs=np.random.uniform(0,1,2*N)
def f(x):
  return 1.0/(1.0+x)
answer1=np.mean(f(xs))
''',number=100000)

t2=timeit('''import numpy as np
N=1500
def f(x):
  return 1.0/(1.0+x)
xs=np.random.uniform(0,1,N)
axs=1-xs
results=np.mean(f(xs))
aresults=np.mean(f(axs))
answer2=(np.mean(results)+np.mean(aresults))/2
''',number=100000)

diff=t2-t1
percent=diff/t1
print('\nAntithetic MC integration {:.2f}% slower than simple MC integration...'.format(percent*100))
print('but yielded an uncertainty ~{:.1f} times smaller than that of the simple MC integration, which would take {:.2f} times as many samples.'.format(uncertainty1/uncertainty2,np.square(uncertainty1/uncertainty2)))
