# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:06:36 2019

@author: botond

# limitations:
-only one subject

# questions:
%part1:
-take absolute value of correlations?
-substitute for loops if scipy pearson correlation can take in np arrays
-pvals are not saved

%part2:
-how do we calculate sigma1 and sigma2? i calcualted the total deviation of all x and y values, but i was thinking what if sigma is the deviation of individual points?
-pre exp factor not necessary as it is a constant (if prev statement is true)
-took the log otherwise reaches zero very fast

"""

import numpy as np
import scipy.io
import scipy.stats
import matplotlib.pyplot as plt
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

plt.close("all")

oxy_sub1=scipy.io.loadmat('C:/Users/boton/Documents/Documents/Stony Brook University/2. Semester/BME 502 - Advanced Numerical & Computation Analysis/Final Project\Github repo/fMRI-analysis/oxytocinRSdata/subject1.mat')

llp = oxy_sub1['mts'][0][0] # left lateral parietal cortex
mpfc = oxy_sub1['mts'][1][0] # medial prefrontal cortex
pcc = oxy_sub1['mts'][2][0] # posterior cingulate cortex
rlp = oxy_sub1['mts'][3][0] # right lateral parietal cortex

# timeshfiting mechanism:

max_shift = 10 # 1 increment is 0.8 seconds

shifts = np.arange(-max_shift,max_shift+1,1)

# array for indexes, so this converts one parameters (Tau) into timeshift
timeshifts = np.zeros((2*max_shift+1,4))
timeshifts[:,0:2] = [0,-max_shift]
timeshifts[:,2] = np.abs(shifts)
timeshifts[:,3] = np.array(-(10-abs(shifts)))
timeshifts[max_shift+1:,[0,1,2,3]] = timeshifts[max_shift+1:,[2,3,0,1]] 

timeshifts = timeshifts.astype(int)

#%%############# PART 1: Calculate Pearson correlation at different time delays #############

combs = list(combinations([0, 1, 2, 3], 2))
titles = ["llp","mpfc","pcc","rlp"]

pcorrs = np.zeros((len(combs),len(shifts))) # 2D array for correlation values
real_timeshifts = np.linspace(-max_shift*0.8,max_shift*0.8,len(shifts),endpoint=True) # x values (in seconds) for graphs
    
fig, axs = plt.subplots(2, 3) # open figure with subplots
    
for i,comb in enumerate(combs): # for all combinations
    ind1 = comb[0] # index of A
    ind2 = comb[1] # index of B
    
    title = titles[ind1] + " vs " + titles[ind2]  # title for graph
    
    x = oxy_sub1['mts'][ind1][0] # timeseries A
    y = oxy_sub1['mts'][ind2][0] # timeseries B
    
    for j,ts in enumerate(range(len(shifts))): # for each timeshift
        xvals = x[timeshifts[ts,0]:(timeshifts[ts,1] or None)]
        yvals = y[timeshifts[ts,2]:(timeshifts[ts,3] or None)]
        pcorrs[i][j] = scipy.stats.pearsonr(xvals,yvals)[0] # calculate correlation for each timeshift

    # display
    a = int(i/3) # first coordinate of subplot
    b = i%3 # second coordinate of subplot
   
    axs[a,b].scatter(real_timeshifts, pcorrs[i], color='b') # plot
    axs[a,b].set_title(title)
    axs[a,b].set_xlabel("timeshift (s)")
    axs[a,b].set_ylabel("pearson correlation")
    axs[a,b].grid()
    axs[a,b].set_ylim([-0.8,0.8])
    axs[a,b].plot([-8,8],[0,0],linewidth=2, color='k')
    
print("negative delay means y happens sooner than x") ###

#%%############## PART 2: rho at zero time delay #############
    
# function for calculation P(r)  
def Posterior(xvals,yvals,r,mx,my,sigx,sigy):
    preexp = 1/(2*np.pi*sigx*sigy*np.sqrt((1-r**2)))
    P_rho = 0  # for log
    for x,y in zip(xvals,yvals):
        z = (x-mx)**2/sigx**2 + (y-my)**2/sigy**2 - 2*r*(x-mx)*(y-my)/(sigx*sigy)
        newP = np.log(preexp * np.exp(-z/(2*(1-r**2))))
        P_rho = P_rho + newP
    return P_rho

r_vals = np.linspace(-0.8,0.8,101) # r values tested

P_r = np.zeros((len(combs),len(r_vals))) # empty array for storing P(r) values

fig, axs = plt.subplots(2, 3) # open figure with subplots
    
for i,comb in enumerate(combs): # for all combinations
    ind1 = comb[0] # index of A
    ind2 = comb[1] # index of B
    
    title = titles[ind1] + " vs " + titles[ind2]  # title for graph
    
    x = oxy_sub1['mts'][ind1][0] # timeseries A
    y = oxy_sub1['mts'][ind2][0] # timeseries B
    
    # inputs for function
    mx = np.mean(x)
    my = np.mean(y)
    sigx = np.std(x)
    sigy = np.std(y)
    
    # call function
    P_r[i] = Posterior(x,y,r_vals,mx,my,sigx,sigy)

    # display
    a = int(i/3) # first coordinate of subplot
    b = i%3 # second coordinate of subplot
   
    axs[a,b].plot(r_vals, P_r[i]) # plot
    axs[a,b].set_title(title)
    axs[a,b].set_xlabel("r")
    axs[a,b].set_ylabel("log(P(r))")
    axs[a,b].grid()
    
#%%############## PART 3: time delay & rho #############
r_vals = np.linspace(-0.8,0.8,11) # r values tested

R,S = np.meshgrid(r_vals, real_timeshifts) # meshgrid for displaying

P_r = np.zeros((len(combs),len(shifts),len(r_vals)))  # for storing P(r) values
maxrs = np.zeros((len(combs),len(shifts),3)) # for storing max P(r) values

#fig = plt.figure() # if we want everything to be on one figure

for i,comb in enumerate(combs): # for all combinations
    ind1 = comb[0] # index of A
    ind2 = comb[1] # index of B
    
    title = titles[ind1] + " vs " + titles[ind2]  # title for graph
    
    x0 = oxy_sub1['mts'][ind1][0] # timeseries A
    y0 = oxy_sub1['mts'][ind2][0] # timeseries B
        
    for j,ts in enumerate(range(len(shifts))): # for each timeshift
        x = x0[timeshifts[ts,0]:(timeshifts[ts,1] or None)]
        y = y0[timeshifts[ts,2]:(timeshifts[ts,3] or None)]
        
        # inputs for function
        mx = np.mean(x)
        my = np.mean(y)
        sigx = np.std(x)
        sigy = np.std(y)
        
        # call function
        P_r[i][j] = Posterior(x,y,r_vals,mx,my,sigx,sigy)
        
        # save the maximum
        maxrs[i][j] = [r_vals[P_r[i][j] == max(P_r[i][j])], real_timeshifts[j], max(P_r[i][j])]  # for each time delay, save the location of the maximum P(r) and its value

    # display
#    ax = fig.add_subplot(2, 3, i+1, projection='3d') # if we want everything to be on one figure      
    
    fig = plt.figure() # if we want a new figure every loop
    ax = fig.add_subplot(111,projection='3d') # if we want a new figure every loop
    
    ax.plot_surface(R,S,P_r[i], antialiased=True) # plot the surface in 3d
#    ax.plot(maxrs[i,:,0], maxrs[i,:,1], maxrs[i,:,2], color='r') # plot the maximum P(r) as a line
#    ax.plot(pcorrs[i], maxrs[i,:,1], maxrs[i,:,2], color='b') # plot the calculated pearson R from part 2 (at the height of log P(r,delay))
    ax.set_title(title)
    ax.set_xlabel("R value")
    ax.set_ylabel("time delay (s)")
    ax.set_zlabel("log P(r,delay)")
    
#    l1 = mpatches.Patch(color='r', label='maximum')
#    l2 = mpatches.Patch(color='b', label='pearson R from part2')

#    ax.legend(handles=[l1, l2], loc=2)
        
#%% pymc3 theanos sampling method TESTING
##
#import pymc3 as pm
#import scipy.io
#import scipy as sp
#import theano.tensor as th
#
#def precision(sigma, rho):
#    C = th.alloc(rho, 2, 2)
#    C = th.fill_diagonal(C, 1.)
#    S = th.diag(sigma)
#    return th.nlinalg.matrix_inverse(S.dot(C).dot(S))
#
## allocate a matrix
## fill the diagonal of ones
## combine this matrix and then invert it
#    
##the reason why we do it with theano is beacuse it's much faster (since it has to be called every loop for the samlpling)
#    
## define the main part of the main to make it universal for different data (the only thing left is to call it using trace)
#def Bayesian_Pearson(data):
#    with pm.Model() as model:
#        # priors might be adapted here to be less flat
#        mu = pm.Normal('mu', mu=0., tau=0.000001, shape=2, testval=np.mean(data, axis=1))
#        sigma = pm.Uniform('sigma', lower=1e-6, upper=1000., shape=2, testval=np.std(data, axis=1))
#        rho = pm.Uniform('r', lower=-1., upper=1., testval=0)
#
#        prec = pm.Deterministic('prec', precision(sigma, rho))
#        mult_n = pm.MvNormal('mult_n', mu=mu, tau=prec, observed=data.T)
#
#    return model
#
#rlp_llp = np.array([rlp,llp])
#
#model = Bayesian_Pearson(rlp_llp)
#with model:
#    trace = pm.sample(10000,njobs=1)
#pm.traceplot(trace, varnames=['mu', 'r', 'sigma'])

#%% PyMC3 TESTING section

import pymc3 as pm
import scipy.io
import scipy as sp ###
import theano ###
import theano.tensor as th

def precision(sigma, rho):
    C = th.alloc(rho, 2, 2)
    C = th.fill_diagonal(C, 1.)
    S = th.diag(sigma)
    return th.nlinalg.matrix_inverse(S.dot(C).dot(S))

"""
this is what I want:
"""

def Bayesian_Pearson(data):
    with pm.Model() as model:
        # priors might be adapted here to be less flat
        mu = pm.Normal('mu', mu=0., tau=0.000001, shape=2, testval=np.mean(data, axis=1))
        sigma = pm.Uniform('sigma', lower=1e-6, upper=1000., shape=2, testval=np.std(data, axis=1))
        rho = pm.Uniform('r', lower=-1., upper=1., testval=0)
        ts = pm.DiscreteUniform('ts', lower=0, upper=2*max_shift, testval=11) # if max_shift is 10 then we go from 0 to 21 values: [-10,-9,-8,....0,....7,8,9,10]

        tsi = pm.Deterministic('test', ts) # this needs additional work
        sdata = np.array([data[0,timeshifts[tsi,0]:(timeshifts[tsi,1] or None)], data[1,timeshifts[tsi,2]:(timeshifts[tsi,3] or None)]]) # shifted time data

        prec = pm.Deterministic('prec', precision(sigma, rho))
        mult_n = pm.MvNormal('mult_n', mu=mu, tau=prec, observed=sdata.T)

    return model

"""
this is the closest I got:
"""


@theano.compile.ops.as_op(itypes=[th.lscalar],otypes=[th.lscalar])
def tsconv(ts):
    return ts

@theano.compile.ops.as_op(itypes=[th.lscalar],otypes=[th.lscalar])
def Bayesian_Pearson(data):
    with pm.Model() as model:
        # priors might be adapted here to be less flat
        mu = pm.Normal('mu', mu=0., tau=0.000001, shape=2, testval=np.mean(data, axis=1))
        sigma = pm.Uniform('sigma', lower=1e-6, upper=1000., shape=2, testval=np.std(data, axis=1))
        rho = pm.Uniform('r', lower=-1., upper=1., testval=0)
        ts = pm.DiscreteUniform('ts', lower=0, upper=2*max_shift, testval=11) # this needs to be discrete!!! is max inclusive?? # if max_shift is 10 then we go from 0 to 21 values: [-10,-9,-8,....0,....7,8,9,10]

        tsi = tsconv(ts)

        sdata = np.array([data[0,timeshifts[tsi,0]:(timeshifts[tsi,1] or None)], data[1,timeshifts[tsi,2]:(timeshifts[tsi,3] or None)]]) # shifted time data
   
        prec = pm.Deterministic('prec', precision(sigma, rho))
        mult_n = pm.MvNormal('mult_n', mu=mu, tau=prec, observed=sdata.T)

    return model
#
rlp_llp = np.array([rlp,llp])

model = Bayesian_Pearson(rlp_llp)
with model:
    trace = pm.sample(1000,njobs=1)
pm.traceplot(trace, varnames=['mu', 'r', 'sigma', 'ts'])


"""

things to find out:
can i use discrete parameters for the distribution
is the max boundary inclusive or exclusive
possibilities: we will have to interpolate the data for every loop, but that would cost enormous amount of calculations in my opinion
                or we can just round the continuous variable to integer (which might have issues in terms of the gradient)??

if it's working put 500 for tuning

+ 
# fit parabola for correlation (step 1)
"""

#%% now I'll write my own sampler
# setup:
x = llp
y = rlp

n = int(1e5) # number of draws

# function
def p_posterior(mux,muy,sigx,sigy,rho,ts):
    global p
    xvals = x[timeshifts[ts,0]:(timeshifts[ts,1] or None)]
    yvals = y[timeshifts[ts,2]:(timeshifts[ts,3] or None)]
    z = (xvals-mux)**2/sigx**2 + (yvals-muy)**2/sigy**2 - 2*rho*(xvals-mux)*(yvals-muy)/(sigx*sigy)
        # should we calculate rho from x and y or should it be a parameter? (i'm guessing the latter since mu and sigma are all parameters)
#    p = 1/(2*np.pi*sigx*sigy*np.sqrt(1-rho**2))*np.exp(-z/(2*(1-rho**2)))
#    return np.prod(p)
    p = -z/(2*(1-rho**2))
    return np.sum(p)

# prior distributions:
#mux = np.random.normal(np.mean(x),1e-8,n)
#muy = np.random.normal(np.mean(y),1e-8,n)
#    # what should the sigma be for mean
#sigx = np.random.normal(np.std(x),1,n)
#sigy = np.random.normal(np.std(y),1,n)
#    # what should the sigma be for sigma

rho = np.random.uniform(-1.0,1.0,n)
ts = np.random.randint(0,2*max_shift+1,n)

# if mu and sigma are considered to be known
mux = np.mean(x)
muy = np.mean(y)
sigx = np.std(x)
sigy = np.std(y)

mux_pr = np.mean(x)
muy_pr = np.mean(y)
sigx_pr = np.std(x)
sigy_pr = np.std(y)    

p_mc = p_posterior(mux,muy,sigx,sigy,rho[0],ts[0])

# initiate chains
#mux_ch = np.full(n,np.nan)
#muy_ch = np.full(n,np.nan)
#sigx_ch = np.full(n,np.nan)
#sigy_ch = np.full(n,np.nan)
rho_ch = np.full(n,np.nan)
ts_ch = np.full(n,np.nan)

#mux_ch[0] = mux[0]
#muy_ch[0] = muy[0]
#sigx_ch[0] = sigx[0]
#sigy_ch[0] = sigy[0]
rho_ch[0] = rho[0]
ts_ch[0] = ts[0]


# initial proposal p
#p_mc = p_posterior(mux[0],muy[0],sigx[0],sigy[0],rho[0],ts[0])

# execute MCMC sampling
for i in range(1,n):
    # proposals
#    mux_pr = mux[i]
#    muy_pr = muy[i]
#    sigx_pr = sigx[i]
#    sigy_pr = sigy[i]
    rho_pr = rho[i]
    ts_pr = ts[i]
    
    # calculate proposal p
    p_proposal = p_posterior(mux_pr,muy_pr,sigx_pr,sigy_pr,rho_pr,ts_pr)
    
#    print("p_proposal:", float(format(p_proposal,'.2f')), 
#          "p_mc:", float(format(p_mc,'.2f')), 
#          "ratio:", float(format((p_proposal/p_mc),'.2f')))
    
    # decide
    if p_proposal>=p_mc or 0.97<p_mc/p_proposal:
        # update
        p_mc = p_proposal
        
        # append   
#        mux_ch[i] = mux[i]
#        muy_ch[i] = muy[i]
#        sigx_ch[i] = sigx[i]
#        sigy_ch[i] = sigy[i]
        rho_ch[i] = rho[i]
        ts_ch[i] = ts[i]
                
"""
PROBLEM 1: scaling problem (product is essentially 0)
    p_proposal if flipped now because the returned p-s are around -7000 since i took the log.
    if i don't take the log their product becomes very small (practically 0)

PROBLEM 2: bad combinations also get accepted (relation in if statement is to permissive)
    it is too permissive, try tweaking the proposal distributions (narrowing them down)
    AND increasing the barrier for accepting cases (maybe set it above 0.8 instead of np.random.random?)
    
    ALSO: I removed 4 parameters (mux, muy, sigx, sigy), but their values are for the whole timeseries.
        they are not updated with every timeshift
"""

# clean up chains from nans
def clearNaNs(chain):
    return chain[~np.isnan(chain)]

#mux_ch = clearNaNs(mux_ch)
#muy_ch = clearNaNs(muy_ch)
#sigx_ch = clearNaNs(sigx_ch)
#sigy_ch = clearNaNs(sigy_ch)
rho_ch = clearNaNs(rho_ch)
ts_ch = clearNaNs(ts_ch)

# generate traces
def trace(var):
    pairs =  np.array([[var[i-1],var[i]] for i in range(1,len(var))])
    plt.figure()
    plt.plot(pairs[:,0],pairs[:,1],linewidth=0.1)
    plt.xlabel('$x_n$')
    plt.ylabel('$x_{n+1}$')
    return

trace(ts_ch)

ts_ch = 0.8*(ts_ch+-10)
plt.figure()
plt.hist(ts_ch,density=True,bins=len(shifts),color='r')
plt.xlabel("timeshift")
plt.ylabel("P(timeshift)")

plt.figure()
plt.hist(rho_ch,density=True)
plt.xlabel("rho")
plt.ylabel("P(rho)")

print("acceptance ratio: ",float(format(len(ts_ch)/len(ts)*100,'.2f')),"%",sep="")