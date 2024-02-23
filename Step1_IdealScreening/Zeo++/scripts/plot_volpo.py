import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

for guest in ['co2', 'n2']:
    count = 0
    count_poav0 = 0
    count_PONAV = 0
    poav = []
    ponav = []
    for fn in os.listdir('volpos'):
        if not fn.endswith('_{}_mm3.volpo'.format(guest)): continue
        count += 1
        with open('volpos/' + fn, 'r') as f:
            line = f.readline().strip().split()
            if not line[11] == '0':
                poav.append(float(line[11]))
            if not line[17] == '0':
                ponav.append(float(line[17]))
            assert line[10] == 'POAV_cm^3/g:'
            assert line[16] == 'PONAV_cm^3/g:'
    assert count == 15000
    print('{} ({}) - POAV: min = {}, mean = {}, median = {}, max = {}'.format(guest, len(poav), min(poav), np.mean(poav), np.median(poav), max(poav)))
    print('{} ({}) - PONAV: min = {}, mean = {}, median = {}, max = {}'.format(guest, len(ponav), min(ponav), np.mean(ponav), np.median(ponav), max(ponav)))

    plt.figure()
    for i, data in enumerate([poav, ponav]):
        data = np.log10(np.array(data))
        density = gaussian_kde(data)
        x = np.linspace(min(data), max(data), 100)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        label = ['POAV', 'PONAV'][i]
        plt.plot(10**x, density(x)/max(density(x)), label = label)
    plt.xscale('log')
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    plt.title({'co2': r'$\mathrm{CO_2}$', 'n2': r'$\mathrm{N_2}$'}[guest])
    plt.xlabel(r'(Non-)accessible probe-occupiable volume [$\mathrm{cm^3/g}$]')
    plt.yticks([])
    plt.ylabel('Normalized frequency')
    plt.legend(loc = 'lower left')
    plt.savefig('../PONAV{}.pdf'.format(guest), bbox_inches = 'tight')

