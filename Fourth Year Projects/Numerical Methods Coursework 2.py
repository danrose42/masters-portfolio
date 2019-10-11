# -*- coding: utf-8 -*-
"""
The method I have chosen to solve this boundary value problem is shooting, as 
it is relatively simple to implement and usually more accurate than the finite 
differences method. Had my shooting method failed to obtain the desired 
solution, I would have then opted to use either finite differences or a more 
complicated function basis method. However for the task given shooting has 
worked to an acceptable degree of accuracy, and the required results are 
displayed in 3 figures when the code is run. 

Shooting works by guessing the initial y’ value y’(a) = z, and then solving the 
related initial value problem. The resulting ordinary differential equation 
from this guess is then integrated to obtain the approximate solution (for 
this I chose to use the black-box function scipy.integrate.odeint). If for the 
approximate solution y(b) = B, then the boundary value problem is solved. 
However as there are many solutions with the initial condition y(a) = A, it is 
unlikely that the desired solution is obtained immediately. By varying the 
initial guess for z and then comparing the solution y(b) to B, the desired 
result is obtained via a root find using scipy.optimize.brentq (the recommended 
solver for a scalar case).

Finally I would suggest that for the second case alpha = 7/4 and beta = 5, the 
mathematical problem has not been posed correctly to solve the original 
problem. The obtained solution has the charge y(t) initially rising slightly 
above 1 before dropping, and as this function corresponds to the real world 
efficiency then this is not physically possible. Realistically the company has 
been told to reduce prices do to their monopoly, and will therefore almost 
certainly not be allowed to increase prices even temporarily. 
"""

from __future__ import division
import numpy as np
from scipy import integrate, optimize
from matplotlib import pyplot, rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)

def Lagrangian(t, y, ydot, a, b):
    """
    Function L(t, y(t), y'(t)) as given in equation (2) of notes.

    Parameters:
    t: time coordinate (float)
    y: y(t) coordinate (float)
    ydot: y'(t) coordinate (float)
    a: alpha value (float)
    b: beta value (float)
    
    Returns:
    P - y from equation (1) where P is the penalty: (float) 
    P(t, y') = alpha*(y')^2 + beta*(t^2 -1)*(y')^3 
    """
    
    assert(np.isreal(t) and np.isscalar(t) and np.isfinite(t)),\
    't values must be real scalars, finite and not NaN'
    assert(np.isreal(y) and np.isscalar(y) and np.isfinite(y)),\
    'y values must be real scalars, finite and not NaN'
    assert(np.isreal(ydot) and np.isscalar(ydot) and np.isfinite(ydot)),\
    'ydot values must be real scalars, finite and not NaN'
    assert(np.isreal(a) and np.isscalar(a) and np.isfinite(a)),\
    'a must be a real scalar, finite and not NaN'
    assert(np.isreal(b) and np.isscalar(b) and np.isfinite(b)),\
    'b must be a real scalar, finite and not NaN'
    
    return a*ydot**2 + b*(t**2 -1)*ydot**3 - y

def ODE1st(z, t, a, b, h=0.01, L=Lagrangian):
    """
    Defines the first order form of the ordinary differential equation from 
    Euler-Lagrange equation (4), rearranged for y''(t).
    
    Parameters:
    z: the state vector z = [y, y'] (2D vector of floats)
    t: time coordinate (float)
    a: alpha value (float)
    b: beta value (float)
    h: step size for numerical derivative functions (float)
    L: function defining Lagrangian in equation (2) (function returning a float)
        
    Returns:
    dzdt: derivative of z with respect to t (2D vector of floats)
    """

    assert(np.all(np.isreal(z)) and len(z) == 2 and np.all(np.isfinite(z))),\
    'z must be a real 2D vector, finite and not NaN'
    # t, a and b inputs already checked by Lagrangian function
    assert(np.isreal(h) and h > 0 and np.isscalar(h) and np.isfinite(h)),\
    'h must be a real, positvive, non-zero scalar, finite and not NaN'
    assert(hasattr(L, '__call__')), 'L must be a callable function'
    
    # Numerical derivative functions for individual terms in equation (4)
    def dL_dy(t, y, ydot):
        return (L(t, y+h, ydot, a, b) - L(t, y-h, ydot, a, b))/(2*h)
    def dL_dydot(t, y, ydot):
        return (L(t, y, ydot+h, a, b) - L(t, y, ydot-h, a, b))/(2*h)
    def d2L_dtdydot(t, y, ydot):
        return (dL_dydot(t+h, y, ydot) - dL_dydot(t-h, y, ydot))/(2*h)
    def d2L_dydydot(t, y, ydot):
        return (dL_dydot(t, y+h, ydot) - dL_dydot(t, y-h, ydot))/(2*h)
    def d2L_dydot2(t, y, ydot):
        return (dL_dydot(t, y, ydot+h) - dL_dydot(t, y, ydot-h))/(2*h)

    dzdt = np.zeros_like(z)
    dzdt[0] = z[1]
    dzdt[1] = (dL_dy(t, z[0], z[1]) - d2L_dtdydot(t, z[0], z[1])\
            - z[1]*d2L_dydydot(t, z[0], z[1]))/d2L_dydot2(t, z[0], z[1])
    return dzdt

def ODE1stEXACT(z, t, a, b):
    """
    Defines the first order form of the ordinary differential equation from 
    Euler-Lagrange equation (3), with the derivate calculated analytically by
    hand, rearranged for y''(t). Used for the final task as the 'accurate'
    solution, towards which the algorithm using numerical derivatives converges 
    as its accuracy is varied.
    
    Parameters z, t, a and b and return dzdt of same nature as ODE1st function.
    """
    
    assert(np.all(np.isreal(z)) and len(z) == 2 and np.all(np.isfinite(z))),\
    'z must be a real 2D vector, finite and not NaN'
    # t, a and b inputs already checked by Lagrangian function

    dzdt = np.zeros_like(z)
    dzdt[0] = z[1]
    dzdt[1] = - (6*b*t*z[1]**2 + 1)/(2*a + 6*b*(t**2-1)*z[1])
    return dzdt

def shooting_BVP(f, t, args, A=1.0, B=0.9):
    """
    Solve the boundary value problem z' = f(z, t), where z = [y, y'],
    subject to boundary conditions y(a) = A and y(b) = B.
    
    Parameters:
    f: function defining the ODE (function returning 2D vector)
    t: time interval array (array of floats)
    args: arguments of f (tuple)
    A: initial condition on y(t) at t = a (float)
    B: final condition on y(t) at t = b (float)
        
    Returns:
    Correct solution of the boundary value problem.
    """
    
    assert(hasattr(f, '__call__')), 'f must be a callable function'
    assert(np.all(np.isreal(t)) and np.all(np.isfinite(t))),\
    't array must be real, finite and not NaN'
    assert(np.all(np.isreal(args)) and np.all(np.isfinite(args))),\
    'args must be real, finite and not NaN'
    assert(np.isreal(A) and np.isscalar(A) and np.isfinite(A)),\
    'A must be a real scalar, finite and not NaN'
    assert(np.isreal(B) and np.isscalar(B) and np.isfinite(B)),\
    'A must be B real scalar, finite and not NaN'

    # Internal function to solve the IVP with initial guess y'(a) = z, and then 
    # compute the error (difference from B) in the boundary condition at t = b
    def shooting_IVP(guess):
        z = [A, guess] # initial conditions z from y(a) and guess
        y = integrate.odeint(f, z, t, args) # solves the IVP
        return y[-1, 0] - B # computes the error at the final point

    # Root-find to obtain correct guess value for y'(a) = z
    correct_guess = optimize.brentq(shooting_IVP, -10.0, 10.0)

    z = [A, correct_guess] # initial condition using the 'correct' guess value
    return integrate.odeint(f, z, t, args) # Solve the BVP (IVP with correct z)

if __name__ == "__main__":
    time = np.linspace(0, 1, 500) # time array (with 500 interval steps)
    
    # TASK 2.a (alpha = 5, beta = 5)
    y_sol = shooting_BVP(ODE1st, time, (5, 5)) # BVP solution y(t)
    pyplot.plot(time, y_sol[:, 0], 'b-') # plots solution
    pyplot.xlim(0.0, 1.0); pyplot.xlabel('t'); pyplot.ylabel('y(t)')
    pyplot.title('Solution for \u03B1 = 5 = \u03B2'); pyplot.show()
    
    # TASK 2.b (alpha = 7/4, beta = 5)
    y_sol = shooting_BVP(ODE1st, time, (7/4, 5)) # BVP solution y(t)
    pyplot.plot(time, y_sol[:, 0], 'b-') # plots solution
    pyplot.xlim(0.0, 1.0); pyplot.xlabel('t'); pyplot.ylabel('y(t)')
    pyplot.title('Solution for \u03B1 = 7/4, \u03B2 = 5'); pyplot.show()
     
    # CONVERGENCE TASK
    
    # 'Exact' solution to compare algorithm (with numerical derivatives) against
    y_solE = shooting_BVP(ODE1stEXACT, time, (7/4, 5))

    length = 8
    errors = np.zeros(length); hs = np.zeros(length)
    for i in range(length):
        h = 0.1/2**i # calculates step size for numerical derivates based on i
        hs[i] = h
        y_sol = shooting_BVP(ODE1st, time, (7/4, 5, h)) # BVP solution y(t)
        
        # Calculates total error by summing errors at each t value, times h (step size)
        errors[i] = h*np.sum(np.abs(y_sol[:, 0] - y_solE[:, 0]))
    
    pyplot.loglog(hs, errors, 'kx') # plots error points on loglog graph
    pfit = np.polyfit(np.log(hs), np.log(errors), 1) # fits curve to data
    error_fit = errors[0]*(hs/hs[0])**pfit[0] # straight line on loglog scale
    pyplot.loglog(hs, error_fit, 'b-') # #plots fit line on loglog graph
    pyplot.xlabel('h (step size for numerical derivatives)')
    pyplot.ylabel('|error|')
    pyplot.title('Algorithm convergence rate = {:.3f} (\u03B1 = 7/4, \u03B2 = 5)'\
                 .format(pfit[0])); pyplot.show()