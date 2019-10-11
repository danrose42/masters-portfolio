from numpy import random, pi, sin, cos, log10, unique, poly1d, polyfit, sum
from matplotlib import pyplot
from datetime import datetime
start_time = datetime.now()

radii = []; masses = []

for repeats in range(10): #loops whole fractal build for end graph
    size = 1601 #matrix size
    empty_space = 0
    matrix = [[empty_space]*size for i in range(size)] #creates empty matrix
    mid = int(size/2) #max fractal size
    maxR = int(mid/2) #max fractal radius
    matrix[mid][mid] = 1 #makes fractal origin occupied
    radius = 10 #initial diffusion start radius
    
    while True: #loops for multiple diffusion particles
        theta = random.uniform(0, 2*pi) #chooses random angle for diffusion start
        x = int(round(radius*cos(theta))) #transfroms polar to cartesian
        y = int(round(radius*sin(theta)))
        n = mid + x; m = mid + y #diffusion start in cartesian
        matrix[n][m] = 1 #diffusion particle start
       
        while True: #loops random particle movement
            matrix[n][m] = empty_space #sets previous particle location to empty
            direction = random.randint(4) #random diffusion direction
            if direction == 0:
                n = n-1
            if direction == 1:
                m = m+1
            if direction == 2:
                n = n+1
            if direction == 3:
                m = m-1
            matrix[n][m] = 1 #new particle position

            xlength = n-mid; ylength = m-mid #x and y distances from origin
            rlength = int(round((xlength**2 + ylength**2)**0.5)) #particle radius vector
            
            if rlength == 2*radius: #abandons particle if it diffuses too far away
                matrix[n][m] = empty_space
                break
    
            if matrix[n-1][m] == 1 or matrix[n][m+1] == 1\
            or matrix[n+1][m] == 1 or matrix[n][m-1] == 1: #particle joins cluster
                if rlength > radius:
                    radius = rlength #diffusion start radius increases as cluster grows
                    radii.append(radius) #radius and mass values taken
                    masses.append(sum(matrix))
                break
        if radius == maxR: #cluster stops growing when it reaches predetermined size
            break
    print('Fractal', repeats+1, 'complete. Runtime =', datetime.now() - start_time)

logR = log10(radii); logM = log10(masses) #logs radii and masses for graph

pyplot.figure(figsize = (12,12))
pyplot.scatter(logR, logM, c='k', marker='.')
pyplot.plot(unique(logR), poly1d(polyfit(logR, logM, 1))(unique(logR)), linewidth=3, c='r')
pyplot.xlabel('logR', fontsize=20)
pyplot.ylabel('logM', fontsize=20)
pyplot.show() #plots data points with first order polynomial fit

pyplot.figure(figsize = (12,12))
pyplot.scatter(logR, logM, c='k', marker='.')
pyplot.plot(unique(logR), poly1d(polyfit(logR, logM, 1))(unique(logR)), linewidth=3, c='r')
pyplot.plot(unique(logR), poly1d(polyfit(logR, logM, 2))(unique(logR)), linewidth=3, c='c')
pyplot.xlabel('logR', fontsize=20)
pyplot.ylabel('logM', fontsize=20)
pyplot.show() #plots data points again with second order polynomial fit on top

gradient, grad_err_squared = polyfit(logR, logM, 1, cov=True) #calculates d and error
print('d = ', gradient[0], '+/-', (grad_err_squared[0][0])**0.5) #extracts matrix elements
print('Total runtime = ', datetime.now() - start_time) #program runtime