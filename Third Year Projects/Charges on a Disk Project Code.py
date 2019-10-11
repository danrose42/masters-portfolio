from numpy import random, exp, sin, cos, pi, linspace, ones
from matplotlib import pyplot
from datetime import datetime
start_time = datetime.now()
print('Program start:', datetime.now())

def energy(charges, Rc, Tc):
    """Calculates the total energy of the system by converting polar to cartesian
    coordinates, calculating the separations, inverting and summing/2."""
    Xc =[]; Yc = []; separations = []; inverse_seps = []
    for i in range(charges):
        Xc.append(Rc[i]*cos(Tc[i]*pi/180.0)); Yc.append(Rc[i]*sin(Tc[i]*pi/180.0))
    for i in range(charges):
        for j in range(charges):
            if j > i:
                separations.append(((Xc[i]-Xc[j])**2.0 + (Yc[i]-Yc[j])**2.0)**0.5)
    for i in range(len(separations)):
        if separations[i] == 0.0:
            separations[i] = 0.1
        inverse_seps.append(1.0/separations[i])
    W = sum(inverse_seps)/2.0
    return W

charges = 12 #number of charges on disk (N)
middle_final_charges = 1 #Number of charges not at max radius in final configuration

size = 101 #max radius of disk (delta=0.01)
M = size*charges*10 #number of repetitions per decrease in temperature T
end_config_number = 100 #number of same configurations in a row to determine final config
lowest_energy = float('inf') #starts energy at infinity
disk_number = 0

while True: #loops whole program if wrong configuration obtained
    disk_number = disk_number+1
    
    Rc = []; Tc = []; charge_counter = 0
    for charge in range(charges): #puts initial charges on disk
        Rstart = random.randint(size); Tstart = random.randint(360)
        Rc.append(Rstart); Tc.append(Tstart)
    old_energy = energy(charges, Rc, Tc)
    temperature = 0.01 #sets initial temperature
    repititions = 0
    configurations = []
    
    while True: #loops movement for 1 disk
        particle = random.randint(charges) #picks a random charge
        direction = random.randint(4) #moves chosen charge in a random direction
        if direction == 0 and Rc[particle] < size-1: #prevents movement past max radius
            Rc[particle] = Rc[particle]+1
        if direction == 1 and Rc[particle] > 0: #prevents movement below 0 radius
            Rc[particle] = Rc[particle]-1
        if direction == 2:
            Tc[particle] = Tc[particle]+1
        if direction == 3:
            Tc[particle] = Tc[particle]-1
        
        new_energy = energy(charges, Rc, Tc)
        if new_energy <= old_energy: #accepts movement if energy has decreased
            old_energy = new_energy
        elif random.random() < exp((old_energy-new_energy)/temperature): #Equation 2
            old_energy = new_energy
        else: #reverses movement if conditions above not met
            if direction == 0:
                Rc[particle] = Rc[particle]-1
            if direction == 1:
                Rc[particle] = Rc[particle]+1
            if direction == 2:
                Tc[particle] = Tc[particle]-1
            if direction == 3:
                Tc[particle] = Tc[particle]+1
        
        if Tc[particle] == 360: #keeps angles within 0 to 360 for simplicity
            Tc[particle] = 0
        if Tc[particle] == -1:
            Tc[particle] = 359
        
        configurations.append(Rc + Tc)
        repititions = repititions+1
        if repititions == M: #decreases T after M repetitions
            repititions = 0
            temperature = temperature*0.9
    
        break_count = 0
        if len(configurations) > end_config_number+1: #decides when final configuration
            for i in range(-(end_config_number+1), -1):
                if configurations[i] == configurations[i-1]:
                    break_count = break_count+1
        if break_count == end_config_number: #ends run for 1 disk
            break
    
    charges_at_radius = 0; TatR = []
    for i in range(charges):
        if Rc[i] == size-1:
            charges_at_radius = charges_at_radius+1
            TatR.append(Tc[i])
    if charges_at_radius == charges - middle_final_charges: #ends program if right configuration
        print('Disk', disk_number, 'complete, final configuration displayed:')
        break
    else: #starts a new disk if incorrect configuration
        print('Disk', disk_number, 'completed incorrectly, runtime:', datetime.now() - start_time)

TatR.sort(); TatR_diffs = []; angle_diff = []
for i in range(len(TatR)-1): #works out angle differnces for charges at radius
    TatR_diffs.append(TatR[i+1]-TatR[i])
angle_between_charges = 360.0/(charges - middle_final_charges)
for i in range(len(TatR_diffs)): #total error on all charge angles
    angle_diff.append(TatR_diffs[i] - angle_between_charges)
angle_error = abs(sum(angle_diff))

Tc_radians = []
for i in range(charges): #converts angles to radians for graph
    Tc_radians.append(Tc[i]*pi/180.0)
pyplot.figure(figsize=(6, 6))
graph = pyplot.subplot(111, projection='polar')
pyplot.polar(linspace(0, 2*pi, 100), ones(100)*(100), 'k-') #plots disk edge on graph
pyplot.polar(Tc_radians, Rc, 'ro', ms=10) #plots charges on graph
graph.set_rmin(0)
pyplot.axis('off')
pyplot.savefig(str(charges) + ' charges')
pyplot.show()
print('Charges:', charges); print('Radii:', Rc); print('Theta:', Tc); print()
print('Ordered angles for radii charges:', TatR)
print('Total angle error for all charges on radius (should be ~ 0): +/-', angle_error)
print('Runtime:', datetime.now() - start_time)
