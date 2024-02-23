#!/bin/sh
#
#PBS -N _block
#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=1
#PBS -m bae

if [[ ! $(hostname) == gligar* ]]; then
        cd $PBS_O_WORKDIR
fi

radius_co2=1.65
radius_n2=1.82

for fn_cif in ../input/*; do
	struct=$(basename $fn_cif)
	struct=${struct/_optimized.cif/}
	network -r ../data/mm3.rad -volpo $radius_co2 $radius_co2 3000 ../output/${struct}_co2.volpo $fn_cif
	network -r ../data/mm3.rad -volpo $radius_n2 $radius_n2 3000 ../output/${struct}_n2.volpo $fn_cif
done
