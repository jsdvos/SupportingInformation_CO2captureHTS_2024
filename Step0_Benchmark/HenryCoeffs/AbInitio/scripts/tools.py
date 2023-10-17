# -*- coding: utf-8 -*-
'''Auxiliary methods'''

import numpy as np
import os
import subprocess
from subprocess import check_output

def rand_rotation_matrix(deflection=1.0, randnums=None):
    '''
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    '''
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def dump_poscar(filename, rvecs, numbers, coordinates, selection=[]):
    '''Write a file in VASP's POSCAR format

       **Arguments:**

       filename
            The name of the file to be written. This is usually POSCAR.

        rvecs
            A (3,3) NumPy array whose rows are the unit cell vectors in bohr.

        numbers
            A (natom,) NumPy array containing atomic numbers.

        coordinates
            A (natom,3) NumPy array containing atomic coordinates in bohr.
    '''
    from molmod import angstrom, deg
    from molmod.periodic import periodic
    # Reciprocal cell vectors, used to convert cartesian to fractional coordinates
    gvecs = np.zeros((3,3))
    for i in range(3):
        gvecs[i] = np.cross(rvecs[(i+1)%3],rvecs[(i+2)%3])/np.linalg.det(rvecs)

    with open(filename, 'w') as f:
        #print >> f, 'Title'
        #print('Title', f=f)
        f.write('Title\n')
        #print >> f, '   1.00000000000000'
        #print('   1.00000000000000', f=f)
        f.write('   1.00000000000000\n')
        # Write cell vectors, each row is one vector in angstrom:
        for rvec in rvecs:
            #print >> f, '  % 21.16f % 21.16f % 21.16f' % tuple(rvec/angstrom)
             #print('  % 21.16f % 21.16f % 21.16f' % tuple(rvec/angstrom), f=f)
             f.write('  % 21.16f % 21.16f % 21.16f\n' % tuple(rvec/angstrom))
        # Construct list of elements to make sure the coordinates get written
        # in this order. Heaviest elements are put first.
        unumbers = sorted(np.unique(numbers))[::-1]
        #print >> f, ' '.join('%5s' % periodic[unumber].symbol for unumber in unumbers)
        #print(' '.join('%5s' % periodic[unumber].symbol for unumber in unumbers), f=f)
        f.write(' '.join('%5s  ' % periodic[unumber].symbol for unumber in unumbers))
        f.write('\n')
        #print >> f, ' '.join('%5i' % (numbers == unumber).sum() for unumber in unumbers)
        #print(' '.join('%5i' % (numbers == unumber).sum() for unumber in unumbers), f=f)
        f.write(' '.join('%5i  ' % (numbers == unumber).sum() for unumber in unumbers))
        f.write('\n')
        #print >> f, 'Selective dynamics'
        #print('Selective dynamics', f=f)
        f.write('Selective dynamics\n')
        #print >> f, 'Direct'
        #print('Direct', f=f)
        f.write('Direct\n')

        # Write the coordinates
        for unumber in unumbers:
            indexes = (numbers == unumber).nonzero()[0]
            for index in indexes:
                row = np.einsum( 'ij,j', gvecs, coordinates[index])
                if index in selection: suffix = "   T   T   T"
                else: suffix = "   F   F   F"
                #print >> f, '  % 21.16f % 21.16f % 21.16f %s' % (row[0],row[1],row[2],suffix)
                #print('  % 21.16f % 21.16f % 21.16f %s' % (row[0],row[1],row[2],suffix), f=f)
                f.write('  % 21.16f % 21.16f % 21.16f %s\n' % (row[0],row[1],row[2],suffix))

def read_vasp_adsorption_energy(workdir):
    '''
    Read the adsorption energy (in Hartree) from VASP calculations.
    Workdir should contain three subdirectories, called complex, host and guest,
    containing VASP output (at least the vasprun.xml file). The adsorption
    energy is calculated as E_complex - E_host - E_guest

    **Arguments**
        workdir
            Directory containing VASP output files

    **Returns**
        energy
            The adsorption energy in atomic units (Hartree)
    '''
    electronvolt = 0.036749325919680595
    def read_vasprun(fn, first=False):
        energy = np.nan
        proc = subprocess.Popen('''grep '<i name="e_fr_energy">' %s'''%(fn), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = proc.communicate()[0].split(b"\n")
        if len(output)<3: return energy
        else:
            energies = []
            for out in output[:-1]:
                #if out.startswith('   <i'):
                if out.startswith(b'   <i'):
                    energy = float(out.split()[-2])*electronvolt
                    energies.append(energy)
                    #print("Energies_array:",energies)
                    #print("Length of energies array:",len(energies))
            if first: energy = energies[0]
            else: energy = energies[-1]
        return energy
    e = read_vasprun(os.path.join(workdir,'complex','vasprun.xml'))
    e -= read_vasprun(os.path.join(workdir,'host','vasprun.xml'))
    e -= read_vasprun(os.path.join(workdir,'guest','vasprun.xml'))
    return e
