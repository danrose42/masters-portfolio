# -*- coding: utf-8 -*-
""" Created on Wed Apr 3 2019 @author: Daniel """
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt, rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12, 6)
rcParams['errorbar.capsize'] = 2
from sys import stdout
from datetime import datetime as time

burn_val = 10000

def SVmodel(mu, phi, sigmaeta_squared, time=6000, disregard=1000):
              #|phi|<1
    generated_ht = np.zeros(time+1)
    for i in range(1, time+1):
        etat = np.random.normal(scale = np.sqrt(sigmaeta_squared))
        generated_ht[i] = mu + phi*(generated_ht[i-1] - mu) + etat
    generated_ht = generated_ht[disregard:]
    
    epsilont = np.random.normal(size = len(generated_ht))
    yt = np.exp(generated_ht/2) * epsilont
    return yt

def sigmaeta2_rand(ht, mu, phi):
    shape = (len(ht)-1)/2
    A = ((1-phi**2)*(ht[0]-mu)**2 + np.sum((ht[1:]-mu-phi*(ht[:-1]-mu))**2))/2
    return stats.invgamma.rvs(shape, scale=A)

def mu_rand(ht, phi, sigmaeta_squared):
    B = (1-phi**2) + (len(ht)-2)*(1-phi)**2
    C = (1-phi**2)*ht[0] + (1-phi)*np.sum(ht[1:]-phi*ht[:-1])
    sigma = np.sqrt(sigmaeta_squared/B)
    return np.random.normal(C/B, sigma)

def phi_rand(ht, mu, sigmaeta_squared, phi_last):
    D = -(ht[0]-mu)**2 + np.sum((ht[1:]-mu)**2)
    E = np.sum((ht[1:]-mu)*(ht[:-1]-mu))
    sigma = np.sqrt(sigmaeta_squared/D)
    P2_new = np.random.normal(E/D, sigma)
    if np.abs(P2_new) >= 1:
        return phi_last
    else:
        Pmetro = min(1, np.sqrt((1-P2_new**2)/(1-phi_last**2)))
        if Pmetro >= np.random.uniform(0, 1):
            return P2_new
        else:
            return phi_last

def dp_dt(ht, y, mu, sigmaeta_squared, phi):
    #sum of [t-1] and [t+1] elements
    ht_sum = np.concatenate([[0], ht[2:] + ht[:-2], [0]])

    dpdt = y**2 * np.exp(-ht)/2 - 1/2\
         - (ht*(1+phi**2) - phi*ht_sum - mu*(phi-1)**2)/sigmaeta_squared
    dpdt[0] = y[0]**2 * np.exp(-ht[0])/2 - 1/2\
            - (ht[0] - mu - phi*(ht[1]-mu))/sigmaeta_squared
    dpdt[-1] = y[-1]**2 * np.exp(-ht[-1])/2 - 1/2\
             - (ht[-1] - mu - phi*(ht[-2]-mu))/sigmaeta_squared
    return dpdt

def leapfrog(x0, p0, n, y, mu, sigmaeta_squared, phi):
    dt = 1/n #trajectory length = 1
    p = p0
    x = x0 + p*dt/2 #x initial half step
    for i in range(n-1):
        p = p + dp_dt(x, y, mu, sigmaeta_squared, phi)*dt
        x = x + p*dt
    p = p + dp_dt(x, y, mu, sigmaeta_squared, phi)*dt #p final step
    x = x + p*dt/2 #x final half step
    return x, p

def Hamiltonian(ht, pt, y, mu, sigmaeta_squared, phi):
    a = np.sum(pt**2 + ht + y**2*np.exp(-ht)) /2
    b = (ht[0] - mu)**2 * (1 - phi**2) /(2*sigmaeta_squared)
    c = np.sum((ht[1:] - mu - phi*(ht[:-1] - mu))**2) /(2*sigmaeta_squared)
    return a + b + c

def HMC(time_series, length_used, iterations=60000, lf_steps=50, burn=burn_val, beta=1, LA=0):
    t0 = time.now(); yt = time_series[:length_used] #LA = look ahead steps
    
    plt.plot(yt, 'k'); plt.xlim(0, length_used)
    plt.xlabel('$t$'); plt.ylabel('$y_t$'); plt.show()
    
    htA = np.zeros((iterations+1, len(yt))); muA = np.zeros(iterations+1)
    phiA = np.zeros(iterations+1); sigeta2A = np.zeros(iterations+1)
    htA[0] = np.full(len(yt), -0.6) #start ht as -0.6 for all values
    muA[0] = 0.0 #cannot also be -0.6 if all ht values are -0.6 (phi sums fail)
    phiA[0] = np.random.uniform(-1, 1)
    sigeta2A[0] = sigmaeta2_rand(htA[0], muA[0], phiA[0])
    pt = np.random.normal(size = len(yt))
    
    #MCMC
    i = 0; acceptance = 0; completions = -1
    k_rejects = 0; k_vals = np.array([]) #k = look ahead steps per iteration
    while np.sum(k_vals) < iterations:
        i = i + 1
        muA[i] = mu_rand(htA[i-1], phiA[i-1], sigeta2A[i-1])
        phiA[i] = phi_rand(htA[i-1], muA[i-1], sigeta2A[i-1], phiA[i-1])
        sigeta2A[i] = sigmaeta2_rand(htA[i-1], muA[i-1], phiA[i-1])
    
        pt = pt*np.sqrt(1-beta) + np.random.normal(size = len(yt))*np.sqrt(beta)
        Hold = Hamiltonian(htA[i-1], pt, yt, muA[i], sigeta2A[i], phiA[i])
        
        if LA == 0:
            HMCtype = 'HMC'; acceptanceType = '\u0394H'
            ht_new, pt_new = leapfrog(htA[i-1], pt, lf_steps, yt, muA[i], sigeta2A[i], phiA[i])
            #leapfrog steps (n) set to 50 for ~85% deltaH acceptance
            Hnew = Hamiltonian(ht_new, pt_new, yt, muA[i], sigeta2A[i], phiA[i])
        
            deltaH = Hnew - Hold
            Pmetro = min(1, np.exp(-deltaH))
            if Pmetro >= np.random.uniform(0, 1):
                htA[i] = ht_new
                pt = pt_new
                acceptance = acceptance+1
            else:
                htA[i] = htA[i-1]
            k_vals = np.append(k_vals, 1)
        
        else:
            HMCtype = 'LA('+str(LA)+')HMC'; acceptanceType = 'Look ahead'
            H_LA = np.zeros(LA+1); H_LA[0] = Hold
            ht_LA = np.zeros((LA+1, len(yt))); ht_LA[0] = htA[i-1]
            pt_LA = np.zeros((LA+1, len(yt))); pt_LA[0] = pt
            
            backward = np.zeros((LA+1, LA+1))*np.nan
            forward = np.zeros((LA+1, LA+1))*np.nan
            
            for k in range(1, LA+1):
                ht_LA[k],pt_LA[k]=leapfrog(ht_LA[k-1],pt_LA[k-1],lf_steps,yt,muA[i],sigeta2A[i],phiA[i])
                H_LA[k] = Hamiltonian(ht_LA[k], pt_LA[k], yt, muA[i], sigeta2A[i], phiA[i])
                
                for a in range(1, k+1):
                    for b in range(a, k+1):
                        backward[a, b] = np.min([1.0 - np.sum(backward[1:a, b]),\
                                np.exp(-H_LA[b-a] + H_LA[b])*(1.0 - np.sum(forward[1:a, b-a]))])
                    for b in range(0, k+1-a):
                        forward[a, b] = np.min([1.0 - np.sum(forward[1:a, b]),\
                               np.exp(-H_LA[b+a] + H_LA[b])*(1.0 - np.sum(backward[1:a, b+a]))])

                if np.sum(forward[1:k+1, 0]) >= np.random.uniform(0, 1):
                    htA[i] = ht_LA[k]
                    pt = pt_LA[k]
                    acceptance = acceptance + 1
                    k_vals = np.append(k_vals, k)
                    break

            if np.sum(htA[i]) == 0:
                htA[i] = htA[i-1]
                k_vals = np.append(k_vals, LA)
                k_rejects = k_rejects + 1
            
        if int(np.round(np.sum(k_vals)*100/iterations)) > completions:
            completions = int(np.round(np.sum(k_vals)*100/iterations))
            stdout.write('\r'); stdout.write('T={} B={} {} {}% complete, runtime = {}'\
            .format(length_used, beta, HMCtype, completions, time.now() - t0)); stdout.flush()
    print(); print('{} acceptance = {:.1f}%'.format(acceptanceType, acceptance*100/(acceptance+k_rejects)))
    k = len(k_vals+1)
    print('<mu> = {:.3f} +/- {:.3f}'.format(np.mean(muA[burn:k]), np.std(muA[burn:k])))
    print('<phi> = {:.3f} +/- {:.3f}'.format(np.mean(phiA[burn:k]), np.std(phiA[burn:k])))
    print('<sigmaeta2> = {:.3f} +/- {:.3f}'.format(np.mean(sigeta2A[burn:k]), np.std(sigeta2A[burn:k])))
    return htA[:k+1], muA[:k+1], phiA[:k+1], sigeta2A[:k+1], (k_vals, k_rejects)

def thermalisation_plot(Tstart, Tend, A, Alabel, burn=burn_val, beta=None):
    """ Does not plot correct MC-time if using Look Ahead HMC """
    if Alabel[0] == 'h':
        plt.plot(np.arange(Tstart, Tend), A[Tstart+burn:Tend+burn], 'k')
    else:
        plt.plot(A, 'k', label='<${}$> = {:.3f} \u00B1 {:.3f}'\
                 .format(Alabel, np.mean(A[burn:]), np.std(A[burn:])))
        plt.legend(handlelength=0)
    if type(beta) == int or type(beta) == float:
        plt.xlabel('Monte Carlo history ($\u03B2={}$)'.format(beta))
    else:
        plt.xlabel('Monte Carlo history')
    plt.ylabel('${}$'.format(Alabel))
    plt.xlim(Tstart, Tend); plt.show()
    return np.mean(A[burn:]), np.std(A[burn:])
    
def ACF(t, X, LA):
    if np.all(LA == 1.0):
        Xj = X[t:]; Xjt = X[:-t]; len_bins = len(Xj)
        bins = (Xj - np.mean(X))*(Xjt - np.mean(X))
        gamma = np.mean(bins)
        jackknife_bins = (np.sum(bins) - bins)/(len_bins-1)
        jackknife_error_squared = (len_bins-1)*np.mean((jackknife_bins - gamma)**2)

    else:
        X_arr = X; X_avg = np.mean(X); LA = np.insert(LA, 0, 1.0)

        k_vals = np.unique(LA)
        for k in k_vals:
            k_locs = np.array([], dtype=int)
            for i in range(int(k)-1):
                k_locs = np.append(k_locs, np.where(LA == k)[0])
            X_arr = np.insert(X_arr, k_locs, np.nan)
            LA = np.insert(LA, k_locs, np.nan)

        Xj = X_arr[t:]; Xjt = X_arr[:-t]
        bins = (Xj - X_avg)*(Xjt - X_avg)

        val_bins = bins[~np.isnan(bins)]

        gamma = np.mean(val_bins)

        jackknife_bins = (np.sum(val_bins) - val_bins)/(len(val_bins)-1)
        jackknife_error_squared = (len(val_bins)-1)*np.mean((jackknife_bins - gamma)**2)
    
    return gamma/np.var(X), np.sqrt(jackknife_error_squared)/np.var(X)
    
def ACFplot(tmax, variables, labels, k_LAs, minY=False, burn=burn_val, singleParam=False):
    step = int(np.ceil(tmax/200)) #USE MULTIPLES OF 200 FOR tmax FOR MAXIMUM DENSITY GRAPHS
    MCT = np.arange(0, tmax, step)
    
    if type(singleParam) == str:
        for i in range(len(variables)):
            varray = np.ones(len(MCT)); jackknife_errors = np.zeros(len(MCT))
            for j, t in enumerate(MCT[1:]):
                varray[j+1], jackknife_errors[j+1] =\
                ACF(t, variables[i][burn:], k_LAs[i][burn:])
            plt.errorbar(MCT, np.abs(varray), jackknife_errors,\
                         fmt='.', elinewidth=1, label='${}$'.format(labels[i]))
        plt.ylabel('${}$ $ACF(t)$'.format(singleParam))
        
    else:
        for i, v in enumerate(variables):
            varray = np.ones(len(MCT)); jackknife_errors = np.zeros(len(MCT))
            for j, t in enumerate(MCT[1:]):
                varray[j+1], jackknife_errors[j+1] = ACF(t, v[burn:], k_LAs[burn:])
            plt.errorbar(MCT, np.abs(varray), jackknife_errors,\
                         fmt='.', elinewidth=1, label='${}$'.format(labels[i]))
        plt.ylabel('$ACF(t)$')
    
    plt.yscale('log')
    if minY == False:
        plt.ylim(top = 1)
    else:
        plt.ylim(minY, 1)
    plt.xlim(0, tmax); plt.xlabel('$t$'); plt.legend(); plt.show()

def ACT(x, xname, k_LA, burn=burn_val):
    i = 0; ACFsum = 0; errsum = 0
    while True:
        i = i + 1
        val, error = ACF(i, x[burn:], k_LA[burn:])
        if val - error > 0:
            ACFsum = ACFsum + val
            errsum = errsum + error
        else:
            break
    print('2ACT for {} = {:.1f} ({:.1f})'.format(xname, 2*(0.5 + ACFsum), 2*(0.5 + errsum)))
    if xname[0] != 'h':
        print('<{}> = {:.3f} +/- {:.3f} (real error)'\
              .format(xname, np.mean(x[burn:]), np.sqrt(np.var(x[burn:])*2*(0.5 + ACFsum))))
    return 2*(0.5 + ACFsum), 2*(0.5 + errsum)