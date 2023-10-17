# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
System: CO2 inside the COF-5 framework.

This script requires that YAFF is installed, which can be found here:
https://github.com/molmod/yaff
'''

import numpy as np
import os
import re

from yaff import System, ForceField

from tools import dump_poscar, read_vasp_adsorption_energy
from henry_importance import *

def removeLeadingZeros(str):
#remove the leading sample character
    char = str.replace('./sample', '')
# Regex to remove leading
# zeros from a string
    regex = "^0+(?!$)"

    # Replaces the matched
    # value with given string
    char = re.sub(regex, "", char)
    char = int(char)

    return char

class MM3Interaction(object):
    def __init__(self, host, guest):

        
        ####### The host system and force field
       
        self.system0 = System.from_file('COF-5.chk')
        
        self.ff0=ForceField.generate(self.system0, 'pars_COF-5_mm3.txt')
        self.indexes0 = np.arange(self.system0.natom)
        self.energy0 = self.ff0.compute()
        self.parts0 = {}
        for part in self.ff0.parts:
           self.parts0[part.name] = part.energy
        self.host_mass = np.sum(self.system0.masses)
        self.number_overlap = 0
        
        ####### The guest system and force field
        self.system1 = System.from_file('COF-5.chk',rvecs=self.system0.cell.rvecs.copy())
        
        self.ff1= ForceField.generate(self.system1,'pars_co2_mm3.txt')
        self.indexes1 = np.arange(self.system0.natom,self.system0.natom+self.system1.natom)
        self.energy1 = self.ff1.compute()
        self.parts1 = {}
        for part in self.ff1.parts:
           self.parts1[part.name] = part.energy
        
        ####### The host+guest system and force field
        self.system = self.system0.merge(self.system1)
        self.ff = ForceField.generate(self.system, 'pars_COF-5_co2_mm3.txt')
    
    def compute(self, guestpos, energy_overlap=10000*kjmol, thresshold_overlap=0.2*angstrom):
        print(guestpos/angstrom)
        self.ff.system.pos[self.indexes1] = guestpos.copy()
        new_complex_pos = self.ff.system.pos
        self.ff.update_pos(new_complex_pos)
        self.ff1.update_pos(guestpos.copy())
        if self.overlap(thresshold=thresshold_overlap):
            self.number_overlap += 1
            return energy_overlap
        else:
            energytot = self.ff.compute()
            energy1 = self.ff1.compute()
            for part in self.ff1.parts:
                self.parts1[part.name] = part.energy
            energy = energytot - energy1 - self.energy0
            for part in self.ff.parts:
                print('%s: %f kJ/mol' %(part.name, (part.energy-self.parts0[part.name]-self.parts1[part.name])/kjmol))
            print('total: %f kJ/mol' %(energy/kjmol))
            return energy

    def overlap(self, thresshold=0.1*angstrom):
        mind = np.inf
        for ih in self.indexes0:
            for ig in self.indexes1:
                r = self.ff.system.pos[ig] - self.ff.system.pos[ih]
                self.ff.system.cell.mic(r)
                d = np.linalg.norm(r)
                if d<mind: mind = d
                if d<thresshold:
                    print('atoms %i and %i overlap (distance=%f)' %(ih,ig,d/angstrom))
                    return True
        print('no overlap, smallest distance = %f' %(mind/angstrom))
        return False

if __name__=='__main__':
    host = 'COF-5'
    guest = 'co2'
    print("Running demonstration for the adsorption of co2 in COF-5...")
    T = 298.0*kelvin
    nsamples_widom = 1000000
    nsamples_importance = nsamples_widom/1000

    ff = MM3Interaction(host, guest)
    rho = ff.host_mass/np.linalg.det(ff.system0.cell.rvecs)


with open("out.txt") as f:
    sample_list = f.readlines()
    # remove new line characters
    sample_list = [x.strip() for x in sample_list]
    # Driver Code
    for i in range(len(sample_list)):
        sample_list[i]= removeLeadingZeros(sample_list[i])
    
    samples = np.array(sample_list)
    energies=np.load(os.path.join('energies_%s_%s.npy'%(host,guest)))
    configurations=np.load(os.path.join('configurations_%s_%s.npy'%(host,guest)))
    # When the ab initio calculations are finished, we read the adsorption
    # energies generated by VASP
    read_vasp_adsorption_energy.counter =0
    energies_ai = np.array([read_vasp_adsorption_energy(os.path.join('%s_%s'%(host,guest),'sample%09d'%sample))\
    for sample in samples])
    
    print("Energies:",energies_ai)
    if not np.all(np.isfinite(energies_ai)):
        print("ERROR: Some ab initio energy could not be read...")
        
    # Process these ab initio energies and apply the importance sampling
    # formulas.
    process_importance_samples(energies, configurations, samples, energies_ai, rho=rho, T=T)






