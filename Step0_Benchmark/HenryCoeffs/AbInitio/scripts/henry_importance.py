# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
Methods that can be used to perform the importance sampling described in the
manuscript. The following three methods are normally used in sequence

    widom_insertion
        Conventional Widom insertion using a force field, where all energies
        and configurations are retained. Requires a Python interface that
        allows computing force-field energies
    get_importance_samples
        Select samples from the force-field Widom insertion based on a Boltzmann
        distribution of force-field energies
    process_importance_samples
        After ab initio energies are calculated for the samples from the previous
        step (typically done externally), this method calculates the ab initio
        Henry coefficient and adsorption enthalpy
'''

import numpy as np

from tools import rand_rotation_matrix

'''
    Internally all computations are performed in atomic units.
    Below, conversion factors are defined for some more convenient units.
    Naming conventions in this module: unit is the value of one external unit
    in internal - i.e. atomic - units. e.g. If you want to have a distance of
    five angstrom in internal units: ``5*angstrom``. If you want to convert a
    length of 5 internal units to angstrom: ``5/angstrom``. It is recommended to
    perform this kind of conversions, only when data is read from the input and
    data is written to the output.
'''
kelvin = 1.0
angstrom = 1.889726133921252
boltzmann = 3.1668154051341965e-06
avogadro = 6.0221415e23
kjmol = 0.0003808799176039228
kilogram = 1.0977693252662276e+30
amu = 1822.8886273532887
pascal = 3.3989315827578425e-14
bar = 3.3989315827578425e-09


def widom_insertion(guest_init_configuration, ff, rvecs, nsamples, rho=1.0, T=300.0*kelvin):
    '''
    Compute the Henry coefficient and adsorption enthalpy for a guest molecule
    in a rigid framework using Widom insertion.

    **Arguments**
        guest_init_configuration
            NumPy array (natom,3) with the geometry of the guest molecule,
            which is rigid. The guest molecule consists of ``natom'' atoms.

        ff
            Class that has compute method that takes as argument a guest molecule configuration
            (as an (natom,3) NumPy array) and returns the corresponding
            adsorption energy.

        rvecs
            A (3,3) NumPy array whose rows are the unit cell vectors.

        nsamples
            The number of Widom insertions (integer)

    **Optional arguments**
        rho
            The mass density of the framework, only used for outputting the
            Henry coefficient

        T
            Temperature in Kelvin at which the Widom insertion is performed
            (default 300K)

    **Returns**
        energies
            An (nsample,) NumPy array containing all adsorption energies.

        configurations
            An (nsample,natom,3) NumPy array containing all guest configurations   
    '''
    beta = 1.0/boltzmann/T
    # Prepare the output arrays
    energies = np.zeros((nsamples,))
    configurations = np.zeros((nsamples,guest_init_configuration.shape[0],3))
    # Loop over the requested number of samples; each sample considers a
    # completely random configuration for the guest molecule inside the framework
    for isample in range(nsamples):
        # Generate a random translation vector in fractional coordinates
        frac = np.random.uniform(low=-0.5,high=0.5,size=3)
        # Convert the translation vector to Cartesian coordinates
        cart = np.einsum('ji,j',rvecs,frac)
        # Construct a random rotation matrix
        M = rand_rotation_matrix()
        # Apply rotation and translation to initial guest geometry
        pos = np.einsum('ab,ib',M,guest_init_configuration-guest_init_configuration.mean(axis=0)).T
        pos += cart
        print('coords inserted guest=,', pos/angstrom)
        # Compute the corresponding adsorption energy
        e = ff.compute(pos)
        # Store the results
        energies[isample] = e
        configurations[isample] = pos
        if 100*isample%nsamples==0:
            print('Percentage complete = %i' %int(100*isample/nsamples))
    print(energies/kjmol)
    print(np.exp(-beta*energies))
    print('Widom insertion at %4.1f K' %(T))
    print('  number of insertions = %i' %nsamples)
    print('  number of configurations withoverlap = %i' %(ff.number_overlap))
    exp_avg = (np.exp(-beta*energies)).mean()
    exp_std = (np.exp(-beta*energies)).std()
    vexp_avg = (energies*np.exp(-beta*energies)).mean()
    vexp_std = (energies*np.exp(-beta*energies)).std()
    print('  <exp(-E/kT)>   = %.3f +- %.3f' %( exp_avg, exp_std/np.sqrt(nsamples)))
    print('  <E*exp(-E/kT)> = %.3f +- %.3f kJ/mol' %(vexp_avg/kjmol,vexp_std/np.sqrt(nsamples)/kjmol))
    # Compute the Henry coefficient, see Equation 4 of the manuscript
    henry = beta/rho*exp_avg
    henry_error = beta/rho*exp_std/np.sqrt(nsamples)
    # Compute the adsorption enthalpy
    old_settings = np.seterr(all='warn', divide='raise')
    enthalpy = vexp_avg/exp_avg - 1.0/beta
    enthalpy_error = np.abs(enthalpy)/np.sqrt(nsamples)*np.sqrt((vexp_std/vexp_avg)**2+(exp_std/exp_avg)**2)
    # Print these quantities to the screen
    print("Widom insertion     at %4.1f K for %8d samples: henry coefficient = %8.3f +- %8.3f mol/kg/bar | adsorption enthalpy = %8.2f +- %8.2f kJ/mol" % 
            (T,nsamples,henry/avogadro*kilogram*bar,henry_error/avogadro*kilogram*bar,enthalpy/kjmol,enthalpy_error/kjmol))
    return energies, configurations


def get_importance_samples(energies_ff, configurations_ff, nsamples, T=300.0*kelvin):
    '''
    Select configurations from a Widom insertion simulation that can be used
    for importance sampling

    **Arguments**
        energies_ff
            An (N,) NumPy array containing adsorption energies from a Widom
            insertion simulation, typically done with a force field

        configurations_ff
            An (N,natom,3) NumPy array containing the configurations corresponding
            to the energies from the Widom insertion simulations

        nsamples
            Number of samples to be considered for the importance sampling

    **Optional arguments**
        T
            Temperature in Kelvin (default 300K)

    **Returns***
        samples
            An (nsample,) array containing indexes of the samples that will be
            considered in the importance sampling.
    '''
    beta = 1.0/boltzmann/T
    # Construct Boltzmann distribution of force-field energies
    probabilities = np.exp(-beta*energies_ff)
    probabilities /= np.sum(probabilities)
    # Randomly select samples based on this probability distribution function
    samples = np.random.choice(energies_ff.shape[0], size=nsamples, replace=True, p=probabilities)
    return samples


def process_importance_samples(energies_ff, configurations_ff, samples, energies_ai, rho=1.0, T=300.0*kelvin):
    '''
    Compute the Henry coefficient and adsorption enthalpy for a guest molecule
    in a rigid framework using importance sampling

    **Arguments**
        energies_ff
            An (N,) NumPy array containing adsorption energies from a Widom
            insertion simulation, typically done with a force field

        configurations_ff
            An (N,natom,3) NumPy array containing the configurations corresponding
            to the energies from the Widom insertion simulations

        samples
            An (nsample,) array containing indexes of the samples that are
            considered in the importance sampling.

        energies_ai
            An (nsample,) array containing adsorption energies for the
            configurations listed in ``samples'', computed typically with an
            ab initio level of theory.

    **Optional arguments**
        rho
            The mass density of the framework, only used for outputting the
            Henry coefficient

        T
            Temperature in Kelvin at which the Widom insertion is performed
            (default 300K)
    '''
    nsamples = samples.shape[0]
    beta = 1.0/boltzmann/T
    # Compute the Henry coefficient from the force-field Widom insetions
    henry_ff = beta/rho*np.mean(np.exp(-beta*energies_ff))
    
    exp_ff_std = (np.exp(-beta*energies_ff)).std()
    henry_ff_error = beta/rho*exp_ff_std/np.sqrt(nsamples)
    
    # Compute Equation 11 from the manuscript
    ratio = np.mean(np.exp(-beta*(energies_ai-energies_ff[samples])))

    rat_exp_ff_std = (np.exp(-beta*(energies_ai-energies_ff[samples]))).std()
    ratio_error = rat_exp_ff_std/np.sqrt(nsamples)
    # Compute Equation 10 from the manuscript, which provides the ab initio
    # Henry coefficient
    henry = ratio*henry_ff
    henry_error = ((ratio_error/ratio) + (henry_ff_error/henry_ff))*henry
    # Compute Equation 15 from the manuscript, which provides the ab initio
    # adsorption enthalpy (apart from a term k_B*T)
    vexp_num_avg = (energies_ai*np.exp(-beta*energies_ai-energies_ff[samples])).mean()
    vexp_num_std = (energies_ai*np.exp(-beta*energies_ai-energies_ff[samples])).std()
    vexp_den_avg = (np.exp(-beta*energies_ai-energies_ff[samples])).mean()
    vexp_den_std = (np.exp(-beta*energies_ai-energies_ff[samples])).std()

    enthalpy = np.sum(energies_ai*np.exp(-beta*(energies_ai-energies_ff[samples])))/np.sum(np.exp(-beta*(energies_ai-energies_ff[samples]))) -1.0/beta
    enthalpy_error = np.abs(enthalpy)/np.sqrt(nsamples)*np.sqrt((vexp_num_std/vexp_num_avg)**2+(vexp_den_std/vexp_den_avg)**2)
    # Print quantities to screen
    #print("Importance sampling at %4.1f K for %8d samples: henry coefficient = %8.3f mol/kg/bar | adsorption enthalpy = %8.2f kJ/mol | ratio = %8.5f [-]" % 
           # (T,nsamples,henry/avogadro*kilogram*bar,enthalpy/kjmol,ratio))

    print("Importance sampling at %4.1f K for %8d samples: henry coefficient = %8.3f +- %8.3f mol/kg/bar | adsorption enthalpy = %8.2f +- %8.2f kJ/mol" % 
            (T,nsamples,henry/avogadro*kilogram*bar,henry_error/avogadro*kilogram*bar,enthalpy/kjmol,enthalpy_error/kjmol))
