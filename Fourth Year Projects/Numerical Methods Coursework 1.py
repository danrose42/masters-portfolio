# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy import optimize
from matplotlib import pyplot
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16
rcParams['figure.figsize'] = (12,6)


def f(t, q, options):
    """
    Function f(t, q) as given in equation (2) of notes.

    Parameters:
    t: time coordinate (float)
    q: x(t) and y(t) coordinates at time t (2D vector of floats)
    options: gamma, omega and epsilon constants (array of floats)
    
    Returns:
    Right hand side of equation (2) in notes (2D vector of floats)
    """
    
    assert(np.all(np.isreal(t)) and np.all(np.isfinite(t))),\
    't must be real, finite and not NaN'
    #cannot check q is real, finite and not NaN because of RK3 stiff case
    assert(np.all(np.isreal(options)) and np.all(np.isfinite(options))),\
    'gamma, omega and epsilon must be real, finite and not NaN'
    
    x = q[0,0]; y = q[1,0]
    matrix1 = np.array([[options[0], options[2]], [options[2], -1]])
    matrix2 = np.array([[(-1.0 + x**2.0 - np.cos(t))/(2.0*x)],\
                        [(-2.0 + y**2.0 - np.cos(options[1]*t))/(2.0*y)]])
    matrix3 = np.array([[np.sin(t)/(2.0*x)], [options[1]*np.sin(options[1]*t)/(2.0*y)]])
    return np.matmul(matrix1, matrix2) - matrix3


def MyRK3_step(f, t, qn, dt, options):
    """
    Explicit RK3 algorithm for a single step of ODE problem, equation (4) of notes.

    Parameters:
    f: function defining the ODE, equation (2) of notes (function returning 2D vector)
    t: time coordinate (float)
    qn: vector coordinates of approximate solution at time t (2D vector of floats)
    dt: time step interval (float)
    options: gamma, omega and epsilon constants (array of floats)

    Returns:
    New coordinates qn+1 as given in equation (4d) of notes (2D vector of floats)
    """
    
    assert(hasattr(f, '__call__')), 'f must be a callable function'
    #t and options inputs already checked by function f
    #cannot check qn is real, finite and not NaN because of RK3 stiff case
    assert(np.isreal(dt) and dt > 0 and np.isscalar(dt) and np.isfinite(dt)),\
    'dt must be a real positvive scalar, finite and not NaN'
    
    k1 = f(t, qn, options)
    k2 = f(t + dt/2.0, qn + dt/2.0 * k1, options)
    k3 = f(t + dt, qn + dt*(-k1 + 2.0*k2), options)
    return qn + dt/6.0 * (k1 + 4.0*k2 + k3)


def MyGRRK3_step(f, t, qn, dt, options):
    """
    Explicit GRRK3 algorithm for a single step of ODE problem, equation (5) of notes.

    Parameters:
    f: function defining the ODE, equation (2) of notes (function returning 2D vector)
    t: time coordinate (float)
    qn: vector coordinates of approximate solution at time t (2D vector of floats)
    dt: time step interval (float)
    options: gamma, omega and epsilon constants (array of floats)

    Returns:
    New coordinates qn+1 as given in equation (5c) of notes (2D vector of floats)
    """
    
    assert(hasattr(f, '__call__')), 'f must be a callable function'
    assert(np.all(np.isreal(qn)) and np.all(np.isfinite(qn))),\
    'qn must be real, finite and not NaN'
    assert(np.isreal(dt) and dt > 0 and np.isscalar(dt) and np.isfinite(dt)),\
    'dt must be a real, positvive, non-zero scalar, finite and not NaN'

    k1 = np.zeros((2,1)); k2 = np.zeros((2,1))
    def F(K):
        k1[0,0], k1[1,0], k2[0,0], k2[1,0] = K
        return k1[0,0] - f(t + dt/3.0, qn + dt/12.0 *(5.0*k1 - k2), options)[0,0],\
               k1[1,0] - f(t + dt/3.0, qn + dt/12.0 *(5.0*k1 - k2), options)[1,0],\
               k2[0,0] - f(t + dt, qn + dt/4.0 *(3.0*k1 + k2), options)[0,0],\
               k2[1,0] - f(t + dt, qn + dt/4.0 *(3.0*k1 + k2), options)[1,0]
               
    k1_initial = f(t + dt/3.0, qn, options); k2_initial = f(t + dt, qn, options)
    Kinitial = np.array([k1_initial[0,0], k1_initial[1,0], k2_initial[0,0], k2_initial[1,0]])
    
    k1[0,0], k1[1,0], k2[0,0], k2[1,0] = optimize.fsolve(F, Kinitial)
    return qn + dt/4.0 * (3.0*k1 + k2)


def exact_solution(t, options):
    """
    Exact solution of function f(t, q) as given in equation (3) of notes.

    Parameters:
    t: time coordinate (float)
    options: gamma, omega and epsilon constants (array of floats)
    
    Returns:    
    x: exact x(t) coordinates of system (array of floats)
    y: exact y(t) coordinates of system (array of floats)
    """
    
    assert(np.all(np.isreal(t)) and np.all(np.isfinite(t))),\
    't must be real, finite and not NaN'
    assert(np.all(np.isreal(options)) and np.all(np.isfinite(options))),\
    'gamma, omega and epsilon must be real, finite and not NaN'

    x = np.sqrt(1.0 + np.cos(t))
    y = np.sqrt(2.0 + np.cos(options[1]*t))
    return x, y


def plot_subplots(sol_name, t, array, formatting):
    """
    Plots approximate solutions to the ODE from a solver.

    Parameters:
    sol_name: name of solver being used (string)
    t: time coordinate (float)
    array: approximate coordinates calculated (array of floats)
    formatting: plot formatting to use (string)
    
    Returns: (none)
    Plots results on two predefined subplots
    """
    
    assert(type(sol_name) == str), 'solver name (sol_name) must be a string'
    assert(np.all(np.isreal(t)) and np.all(np.isfinite(t))),\
    't must be real, finite and not NaN'
    assert(len(t) == len(array[0])), 'Size of t must be the same as coordinate array'
    assert(type(formatting) == str), 'formatting must be set as a string'

    ax1.plot(t, array[0], formatting, label='{} solution'.format(sol_name))
    ax2.plot(t, array[1], formatting, label='{} solution'.format(sol_name))


def plot_labels_layout():
    """
    Adds labels and adjusts layout of plots (improves clarity of code).
    Parameters: (none), Returns: (none)
    """
    ax1.set_xlabel('$t$'); ax1.set_ylabel('$x(t)$'); ax1.legend()
    ax2.set_xlabel('$t$'); ax2.set_ylabel('$y(t)$'); ax2.legend()
    fig.tight_layout(); fig.subplots_adjust(top=0.91)


def convergence_plot(sol_name, solver, dt_numerator, q0, formatting, options):
    """
    Measures 1-norm y(t) errors for decreasing dt, then calculates and plots the convergence.

    Parameters:
    sol_name: name of solver being used (string)
    solver: chosen Runge-Kutta step function (function returning 2D vector)
    dt_numerator: numerator of dt as defined in tasks 4 and 7 (float)
    q0: initial q(0) coordinates for starting algorithm (2D vector of floats)
    formatting: plot formatting to use (string)
    options: gamma, omega and epsilon constants (array of floats)
    
    Returns: (none)
    Plots convergence of algorithm on predefined subplot
    """
    
    assert(type(sol_name) == str), 'solver name (sol_name) must be a string'
    assert(hasattr(solver, '__call__')), 'solver must be a callable function'
    assert(np.isreal(dt_numerator) and dt_numerator > 0 and np.isscalar(dt_numerator)\
           and np.isfinite(dt_numerator)),\
           'dt_numerator must be a real, positvive, non-zero scalar, finite and not NaN'
    assert(np.all(np.isreal(q0)) and len(q0) == 2 and np.all(np.isfinite(q0))),\
    't must be real 2D vector, finite and not NaN'
    assert(type(formatting) == str), 'formatting must be set as a string'
    assert(np.all(np.isreal(options)) and np.all(np.isfinite(options))),\
    'gamma, omega and epsilon must be real, finite and not NaN'
    
    j_range = 8
    norm_error = np.zeros((j_range,)); deltat = np.zeros((j_range,))
    for j in range(j_range):
        dtj = dt_numerator/2**j #calculates time step interval based on j
        deltat[j] = dtj
        timej = np.arange(0.0, 1.0+dtj, dtj) #generates time coordinates
        sol_y = np.zeros_like(timej)
        sol_y[0] = q0[1,0]; qj = q0
        for i in range(1, len(timej)): #uses chosen algorithm to approximate solutions
            qj = solver(f, timej[i-1], qj, dtj, options)
            sol_y[i] = qj[1,0]
        exact_y = exact_solution(timej, options)[1] #exact solution to calculate errors
        norm_error[j] = dtj * np.sum(np.abs(sol_y - exact_y)) #equation (8) from notes
    ax.loglog(deltat, norm_error, 'kx') #plots points on loglog graph
    pfit = np.polyfit(np.log(deltat), np.log(norm_error), 1) #fits curve to data
    error_fit = norm_error[0]*(deltat/deltat[0])**pfit[0] #straight line on loglog scale
    ax.loglog(deltat, error_fit, formatting,\
              label='{} convergence rate = {:.3f}'.format(sol_name, pfit[0]))
        

if __name__ == "__main__":
    q0 = np.array([[np.sqrt(2)], [np.sqrt(3)]]) #initial q(0) coordinates
    
    # NON-STIFF PROBLEM
    non_stiff_options = np.array([-2.0, 5.0, 0.05]) #gamma, omega, epsilon
    dt = 0.05 #time step interval for task 3
    time = np.arange(0.0, 1.0+dt, dt) #generates time coordinates
    fig, (ax1, ax2) = pyplot.subplots(1, 2) #defines subplots
    
    exact_curve = exact_solution(time, non_stiff_options)
    plot_subplots('Exact', time, exact_curve, 'r-') #plots exact solution
    
    RK3array = np.zeros((2,len(time)))
    RK3array[:,0] = q0[:,0]; qnew = q0
    for i in range(1, len(time)): #approximates solution using RK3 algorithm
        qnew = MyRK3_step(f, time[i-1], qnew, dt, non_stiff_options)
        RK3array[:,i] = qnew[:,0]
    plot_subplots('RK3', time, RK3array, 'gx') #plots RK3 solution

    GRRK3array = np.zeros((2,len(time)))
    GRRK3array[:,0] = q0[:,0]; qnew = q0
    for i in range(1, len(time)): #approximates solution using GRRK3 algorithm
        qnew = MyGRRK3_step(f, time[i-1], qnew, dt, non_stiff_options)
        GRRK3array[:,i] = qnew[:,0]
    plot_subplots('GRRK3', time, GRRK3array, 'bx') #plots GRRK3 solution

    ax1.set_xlim([0.0, 1.0]); ax2.set_xlim([0.0, 1.0]) #standard time limits
    fig.suptitle('Exact, RK3 and GRRK3 (overlapping) solutions to non-stiff problem, '\
                 '$\u0394t$ = %g' %dt)
    plot_labels_layout(); pyplot.show()    
    
    # NON-STIFF CONVERGENCE
    fig = pyplot.figure(); ax = fig.add_subplot(111)
    convergence_plot('RK3', MyRK3_step, 0.1, q0, 'g-', non_stiff_options)
    convergence_plot('GRRK3', MyGRRK3_step, 0.1, q0, 'b-', non_stiff_options)
    ax.set_xlabel('$\u0394t$'); ax.set_ylabel('$||Error||_1$'); ax.legend()
    ax.set_title('RK3 and GRRK3 convergence rates for non-stiff system')
    fig.tight_layout(); pyplot.show()

    
    # RK3 STIFF PROBLEM
    stiff_options = np.array([-2e5, 20.0, 0.5]) #gamma, omega, epsilon
    dt = 0.001 #time step interval for task 5
    time = np.arange(0.0, 1.0+dt, dt) #generates time coordinates
    fig, (ax1, ax2) = pyplot.subplots(1, 2) #defines subplots
    
    exact_curve = exact_solution(time, stiff_options)
    plot_subplots('Exact', time, exact_curve, 'r-') #plots exact solution
    
    RK3array = np.zeros((2,len(time)))
    RK3array[:,0] = q0[:,0]; qnew = q0
    for i in range(1, len(time)): #approximates solution using RK3 algorithm
        qnew = MyRK3_step(f, time[i-1], qnew, dt, stiff_options)
        RK3array[:,i] = qnew[:,0]
    plot_subplots('RK3', time, RK3array, 'g-') #plots RK3 solution
    
    ax1.set_xlim([0.0, 0.1]); ax1.set_ylim([1.2, 1.6]) #plot limits restricted to better
    ax2.set_xlim([0.0, 0.1]); ax2.set_ylim([1.5, 1.9]) #demonstrate algorithm instability
    fig.suptitle('Exact and RK3 (unstable) solutions to stiff problem, $\u0394t$ = %g' %dt)
    plot_labels_layout(); pyplot.show()

    # GRRK3 STIFF PROBLEM
    dt = 0.005 #time step interval for task 6
    time = np.arange(0.0, 1.0+dt, dt) #generates time coordinates
    fig, (ax1, ax2) = pyplot.subplots(1, 2) #defines subplots
    
    exact_curve = exact_solution(time, stiff_options)
    plot_subplots('Exact', time, exact_curve, 'r-') #plots exact solution
    
    GRRK3array = np.zeros((2,len(time)))
    GRRK3array[:,0] = q0[:,0]; qnew = q0
    for i in range(1, len(time)): #approximates solution using GRRK3 algorithm
        qnew = MyGRRK3_step(f, time[i-1], qnew, dt, stiff_options)
        GRRK3array[:,i] = qnew[:,0]
    plot_subplots('GRRK3', time, GRRK3array, 'bx') #plots GRRK3 solution

    ax1.set_xlim([0.0, 1.0]); ax2.set_xlim([0.0, 1.0]) #standard time limits
    fig.suptitle('Exact and GRRK3 (stable) solutions to stiff problem, $\u0394t$ = %g' %dt)
    plot_labels_layout(); pyplot.show()
    
    # GRRK3 STIFF CONVERGENCE
    fig = pyplot.figure(); ax = fig.add_subplot(111)
    convergence_plot('GRRK3', MyGRRK3_step, 0.05, q0, 'b-', stiff_options)
    ax.set_xlabel('$\u0394t$'); ax.set_ylabel('$||Error||_1$'); ax.legend()
    ax.set_title('GRRK3 convergence rate for stiff system')
    fig.tight_layout(); pyplot.show()
