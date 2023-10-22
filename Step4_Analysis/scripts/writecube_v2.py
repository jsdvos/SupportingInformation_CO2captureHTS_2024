#!/usr/bin/env python

import numpy as np
import os

from iodata import IOData, dump_one, load_one
from iodata.utils import Cube
from RASPA_convert_movie import load_pdb
from yaff import System, log

log.set_level(log.silent)

def write_cube(fn_density, fn_framework, fn_cube):

    density = np.load(fn_density)

    framework = System.from_file(fn_framework)
    pos, cell, numbers = framework.pos, framework.cell.rvecs, framework.numbers

    origin = np.zeros((3))
    rvecs = np.zeros((3,3))
    for i in range(3):
        rvecs[i] = cell[i]/density.shape[i]

    cube = Cube(origin, rvecs, density)
    iodata = IOData(cube=cube, atcoords=pos, atnums=numbers)
    dump_one(iodata, fn_cube)


if __name__=='__main__':
    for framework in os.listdir('../input'):
        print(framework)
        folder = os.path.join('../output', framework)
        for i, component in enumerate(['co2', 'n2']):
            if component == 'n2': continue
            fn_density = os.path.join(folder, 'density_{}.npy'.format(component))
            fn_framework = os.path.join('../input', '{}/{}.chk'.format(framework, framework))
            fn_cube = os.path.join(folder, 'density_{}.cube'.format(component))
            write_cube(fn_density, fn_framework, fn_cube)
