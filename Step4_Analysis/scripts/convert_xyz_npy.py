#!/usr/bin/env python

import sys
import numpy as np
import h5py
import os
from numpy.linalg import inv
from molmod.units import *
from molmod.periodic import periodic
from yaff import System, log
log.set_level(log.silent)
from RASPA_convert_movie import load_pdb

def construct_density(pdb_guests, N_guest_atoms, N_guest_atom0=0, N=160, init_step=0, max_step = None, unitcell = None):

    '''
        list_indices: atom indices used for density calculation
        N_guests: the number of guest molecules
        N_atoms_guest: the number of atoms of a guest molecule
        N: the number of pixels
        unitcell: unit cell to project density on if simulation performed on supercell
    '''

    data = np.zeros((N, N, N))
    if max_step is None:
        max_step = len(pdb_guests.pos)
    elif max_step < len(pdb_guests.pos):
        print('WARNING: max_step should be between {} and {}, got {}'.format(
            init_step, len(pdb_guests.pos), max_step))
        print('\t\tContinuing with max_step = {}'.format(len(pdb_guests.pos)))
        max_step = len(pdb_guests.pos)
    
    for i in range(init_step, max_step):
        if len(pdb_guests.pos[i]) > 0:
            if type(N_guest_atom0) == int:
                pos = pdb_guests.pos[i][N_guest_atom0::N_guest_atoms]
            else:
                count = 0
                pos = []
                while count < len(pdb_guests.pos[i]):
                    positions = np.array([pdb_guests.pos[i][count + j] for j in N_guest_atom0])
                    pos.append(sum(positions)/len(positions))
                    count += N_guest_atoms
                pos = np.array(pos)
            if unitcell is None:
                unitcell = pdb_guests.cell[i]
            reduced_coordinates = np.dot(pos, np.linalg.inv(unitcell)) % 1
            data += coordinates2grid(reduced_coordinates, N)

    return data/(len(pdb_guests.pos) - init_step)

def coordinates2grid(coordinates, N):

    grid = np.zeros((N, N, N))

    for i in range(coordinates.shape[0]):
        grid[int(np.floor(coordinates[i,0]*N))%N, int(np.floor(coordinates[i,1]*N))%N, int(np.floor(coordinates[i,2]*N))%N] += 1

    return grid

if __name__=='__main__':
    for framework in os.listdir('../input'):
        print(framework)
        folder = os.path.join('../output', framework)

        for i, component in enumerate(['co2', 'n2']):
            if component == 'n2': continue
            if component == 'co2':
                N_grid = 100 # Number of grid points in each direction
                init_step = 10000 # First initialization steps are removed
                N_guest_atoms = 3 # Number of atoms in a guest molecule
                N_guest_atom0 = 0 # Index of the targeted atom within the molecule
            elif component == 'n2':
                N_grid = 100 # Number of grid points in each direction
                init_step = 10000 # First initialization steps are removed
                N_guest_atoms = 2 # Number of atoms in a guest molecule
                N_guest_atom0 = [0, 1] # Index of the targeted atom within the molecule
    
            for fn in os.listdir(folder):
                if fn.startswith('Movie_') and fn.endswith('_component_{}_{}.pdb'.format(component, i)):
                    fn_gcmc = os.path.join(folder, fn)

            pdb_guests = load_pdb(fn_gcmc)
    
            sys_yaff = System.from_file('../input/{}/{}.chk'.format(framework, framework))
            cell = sys_yaff.cell.rvecs

            data = construct_density(pdb_guests, N_guest_atoms, N_guest_atom0, N=N_grid, init_step=init_step, max_step = 100000, unitcell = cell)
            np.save(os.path.join(folder, 'density_{}.npy'.format(component)), data)
