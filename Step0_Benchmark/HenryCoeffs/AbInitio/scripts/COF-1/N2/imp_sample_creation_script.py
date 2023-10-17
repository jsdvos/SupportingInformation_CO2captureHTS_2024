# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
System: n2 inside the COF-1 framework.

This script requires that YAFF is installed, which can be found here:
https://github.com/molmod/yaff
'''

import numpy as np
import os
import re

from yaff import System, ForceField

from tools import dump_poscar, read_vasp_adsorption_energy
from henry_importance import *


class MM3Interaction(object):
    def __init__(self, host, guest):

        
        ####### The host system and force field
       
        self.system0 = System.from_file('COF-1.chk')
        
        self.ff0=ForceField.generate(self.system0, 'pars_COF-1_mm3.txt')
        self.indexes0 = np.arange(self.system0.natom)
        self.energy0 = self.ff0.compute()
        self.parts0 = {}
        for part in self.ff0.parts:
           self.parts0[part.name] = part.energy
        self.host_mass = np.sum(self.system0.masses)
        self.number_overlap = 0
        
        ####### The guest system and force field
        self.system1 = System.from_file('n2.chk',rvecs=self.system0.cell.rvecs.copy())
        
        self.ff1= ForceField.generate(self.system1,'pars_n2_mm3.txt')
        self.indexes1 = np.arange(self.system0.natom,self.system0.natom+self.system1.natom)
        self.energy1 = self.ff1.compute()
        self.parts1 = {}
        for part in self.ff1.parts:
           self.parts1[part.name] = part.energy
        
        ####### The host+guest system and force field
        self.system = self.system0.merge(self.system1)
        self.ff = ForceField.generate(self.system, 'pars_COF-1_n2_mm3.txt')
    
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
            energy = energytot - energy1 - self.energy0
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
    host = 'COF-1'
    guest = 'n2'
    print("Running demonstration for the adsorption of n2 in COF-1...")
    T = 298.0*kelvin
    nsamples_widom = 1000000
    nsamples_importance = nsamples_widom/1000
    ####### Construct a MM3 for this host+guest
    ff = MM3Interaction(host, guest)

    rho = ff.host_mass/np.linalg.det(ff.system0.cell.rvecs)
    
    # Select some samples from the force-field Widom insertion simulation
    # Note that the number of samples can be substantially smaller that what
    # is required for the Widom insertion
   
    energies=np.load(os.path.join('energies_%s_%s.npy'%(host,guest)))
    configurations=np.load(os.path.join('configurations_%s_%s.npy'%(host,guest)))

    print("Applying the importance sampling methodology...")
    samples = get_importance_samples(energies, configurations, int(nsamples_importance), T=T)
    # Now ab initio calculation need to be performed for these samples
    # This is not actually done in these scripts, but using external software
    # In our case, we generate the VASP POSCAR files
    for sample in samples:
        ff.system.pos[ff.indexes1] = configurations[sample]
        ff.system1.pos[:] = configurations[sample]
        for fn, system in zip(['host','guest','complex'],[ff.system0,ff.system1,ff.system]):
            workdir = os.path.join('%s_%s'%(host,guest),'sample%09d'%sample,fn)
            if not os.path.isdir(workdir): os.makedirs(workdir)
            dump_poscar(os.path.join(workdir,'POSCAR'), system.cell.rvecs, system.numbers, system.pos)
