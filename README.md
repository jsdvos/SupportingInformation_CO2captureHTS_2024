This file relates to the research done in the following paper:

J. S. De Vos, XXX, V. Van Speybroeck, _High-throughput screening of covalent organic frameworks for carbon capture using machine learning_ (2023)

This file is part of the midterm storage (publically available under the [CC BY-SA license](https://creativecommons.org/licenses/by-sa/4.0/)) of the input files relevant in this work. In the remainder of this file, we outline in detail the workflow used in this work including references to the relevant input and output files.

# Software

The following software packages are used to perform all relevant calculations.

- RASPA (version 2.0.41)
- Yaff (version 1.6.0)
- VASP (version 6.2)
- Scikit-learn (version 1.1.2)
- SHAP (version 0.41.0)

# Workflow

The SBU nomenclature within this data archive is somewhat different than the nomenclature used in the manuscript and the ReDD-COFFEE database. More specific, the following adaptions have to be made to the core linkers used in this data archive (data archive core linker id -> manuscript core linker id): 28 -> 26, 29 -> 27, 30 -> 28, 31 -> 29, 32 -> 30, 33 -> 31, 34 -> 32, 35 -> 34, 36 -> 33. Furthermore, the imide terminations are also given a separate id (data archive termination id -> manuscript termination id): 03-12 -> 03-07, 12-12 -> 09-07.

Some tar-files are splitted in multiple smaller files. In these cases, the original file can be restored by running:

`cat file.tar.xz.* | tar xvzf -`

## Step 0 - Benchmark

### Step 0a - Computation of the adsorption isotherms

The adsorption isotherms are calculated with grand-canonical Monte Carlo (GCMC) simulations adopting various force fields using RASPA (version 2.0.41). The isotherms in four different regimes of COF-1, COF-5, COF-102, and COF-103 are calculated. For each pressure point in each isotherm, a separate calculation is performed, for which the input files are stored in a separate folder. The folder structure is defined as follows:

`isotherm_type/material/level_of_theory/pressure`

where `material` determines the studied material, `level_of_theory` specifies which force fields are adopted for the van der Waals interactions (_e.g._ `mm3_trappe` uses the MM3 force field for the host-guest interactions and the TraPPE force field for the guest-guest interactions). `pressure` specifies the applied pressure and `isotherm_type` determines which isotherm is calculated and can be one of the following:

- `co2_298K_HP`: high-pressure isotherm of CO<sub>2</sub> at 298 K.
- `co2_273K_HP`: high-pressure isotherm of CO<sub>2</sub> at 273 K.
- `co2_273K_LP`: low-pressure isotherm of CO<sub>2</sub> at 273 K.
- `n2_77K_LP`: low-pressure isotherm of N<sub>2</sub> at 77 K.

**input**

 - `COF.cif`: the cif-file of the structure with the pseudoatom labels.
 - `guest.def`: the definition of the guest molecule (either CO<sub>2</sub> or N<sub>2</sub>), including its geometry and pseudoatom labels.
 - `pseudo_atoms.def`: the definition of all pseudoatoms including their charge.
 - `force_field_mixing_rules.def`: definition of the host-guest interactions, which are determined with mixing rules. 
 - `force_field.def`: definition of the guest-guest interactions. The interactions in this file overwrite the ones given in force_field_mixing_rules.def.
 - `simulation.input`: RASPA input file defining the simulation parameters.
 - `restart.input`: RASPA input file in case the job has to be restarted
 
**command line**

`simulate -i simulation.input > raspa.log 2> raspa.err`
( or `simulate -i restart.input > raspa.log 2> raspa.err` if the job had to be restarted)

**output**

RASPA produces a variety of output folders, containing a lot of output files

- `raspa.log` and `raspa.err`: RASPA log files
- `CrashRestart` and `Restart`: contain files to enable restarting the simulation in case of an unexpected crash due to, _e.g._, insufficient wall time.
- `Movies` and `VTK`: contain files for the visualization of the trajectory.
- `Output`: contains the most interesting RASPA output file with the number of guest molecules at all timesteps and their energy.

All input folders with their respective input files are stored in the `input.tar.gz` file, whereas all RASPA output files are provided in the `output.tar.gz` file. Furthermore, the resulting isotherms are provided in the `data` folder, together with the experimental reference data, in the [AIF (adsorption information format)](https://adsorptioninformationformat.com/) format.

### Step 0b - Computation of the Henry coefficients

#### i) Widom insertion to calculate the force field Henry coefficients

The Henry coefficients are calculated with Widom insertion adopting various force fields using RASPA (version 2.0.41). For both COF-1 and COF-5, the Henry coefficients of CO<sub>2</sub> and N<sub>2</sub> are calculated using different force fields. The folder structure is defined as follows:

`guest/material/level_of_theory`

**input**

`COF.cif`, `guest.def`, `pseudo_atoms.def`, `force_field_mixing_rules.def`, `force_field.def`, `simulation.input`, `restart.input`

**command line**

`simulate -i simulation.input > raspa.log 2> raspa.err`
( or `simulate -i restart.input > raspa.log 2> raspa.err` if the job had to be restarted)

**output**

`raspa.log`, `raspa.err`, `CrashRestart`, `Restart`, `Movies`, `VTK`, `Output`

All input folders with the respective input files are stored in the `input.tar.gz` file, whereas all RASPA output files are provided in the `output.tar.gz` file. The resulting Henry coefficients and the adsorption enthalpy are stored in the `data` folder using the [AIF (adsorption information format)](https://adsorptioninformationformat.com/) format.


#### ii) Importance sampling to calculate the _ab initio_ Henry coefficients

The workflow to apply the Importance Sampling methodology consists of the four following steps:

**1)** The Henry coefficients are calculated with **Widom insertion** adopting the MM3_CAP force field as the biasing potential using Yaff (version 1.6.0). All energies and configurations are stored as binary NumPy files in`energies.npy` and `configurations.npy`, respectively. Specifically, the `widom_insertion()` method from `henry_importance.py` is used which calls on Yaff to perform the simulations. The Widom insertion simulations are performed with the respective `widom_insertion_script_host_guest.py` master script.

**input**

`COF.chk`, `guest.chk`, `pars_COF_guest_mm3.txt`, `pars_COF_mm3.txt`, `pars_guest_mm3.txt`

**command line**

`python widom_insertion_script_host_guest.py`

**output**

`configurations_COF_guest.npy`, `energies_COF_guest.npy`, `FF_HC_AE.txt`

**2)** The samples from the Widom insertion simulation are selected according to a Boltzmann distribution of the force-field energies. The `imp_sample_creation_script.py` master script, which calls the `get_importance_samples()` method from `henry_importance.py`, performs the **sample selection** task.

**input**

`configurations_COF_guest.npy`, `energies_COF_guest.npy`

**command line**

`python imp_sample_creation_script.py`

**output**

`POSCAR`

3) The **_ab initio_ adsorption energy** for the selected samples (as obtained in Step 2) is computed using VASP (version 6.2).

**input**

`POSCAR`,`INCAR`,`POTCAR`,`KPOINTS`

**command line**

`vasp_std`

**output**

`OUTCAR`, `vasprun.xml`

The VASP input and output files are found within `VASP_input.tar.gz` and `VASP_output.tar.gz`, respectively. When untarring these files, the input and output files are grouped into subdirectories according to sample numbers considered in the initial Widom Insertion simulation. These subdirectories have further subdirectories named complex, guest and host to differentiate the configurations considered.

4) The _ab initio_ results are processed and used to **compute the Henry coefficient and adsorption enthalpy** as done in the `process_importance_samples()` method from `henry_importance.py`. The `ab_initio_HC_AE_comp.py` master script processes the _ab initio_ results to compute the _ab initio_ Henry coefficient and adsorption enthalpy. This master script requires the list of _ab initio_ samples which is provided with the `out.txt` file.  

**input**

`out.txt`

**command line**

`python ab_initio_HC_AE.py`

**output**

`ab_initio_HC_AE.txt`

## Step 1 - Ideal screening

In the first step of the high-throughput screening, the working capacity and the ideal selectivity is calaculated for a diverse set of 15 000 COFs from the ReDD-COFFEE database. For the selection of the diverse database, we refer to the `extract_subset.sh` script provided in the [ReDD-COFFEE landing page](https://doi.org/10.24435/materialscloud:nw-3j). For each of these materials, three GCMC calculations have to be performed:

- `co2_ads`: Uptake of CO<sub>2</sub> at 298 K and 10 bar
- `co2_des`: Uptake of CO<sub>2</sub> at 298 K and 0.1 bar
- `n2_ads`: Uptake of N<sub>2</sub> at 298 K and 10 bar

**input**

`COF.cif`, `guest.def`, `pseudo_atoms.def`, `force_field_mixing_rules.def`, `force_field.def`, `simulation.input`, `restart.input`

**command line**

`simulate -i simulation.input > raspa.log 2> raspa.err`
( or `simulate -i restart.input > raspa.log 2> raspa.err` if the job had to be restarted)

**output**

`raspa.log`, `raspa.err`, `CrashRestart`, `Restart`, `Movies`, `VTK`, `Output`

All input folders with the respective input files are stored in the `input.tar.gz` file, whereas all RASPA output files are provided in the `output.tar.gz` file. The resulting uptakes and enthalpies are stored in the `data` folder using the [AIF (adsorption information format)](https://adsorptioninformationformat.com/) format.

## Step 2 - Machine learning

In the second step of the high-throughput screening, the working capacity and ideal selectivity are predicted for all 268 678 COFs in the ReDD-COFFEE database using machine learning (ML).  All ML calculations are performed in Python using the scikit-learn (version 1.1.2) package, whereas the SHAP (version 0.41.0) package is adopted for the SHAP analysis. The features of each material are provided in the `features.csv` file. For the calculation of these features, we refer to the [ReDD-COFFEE paper](https://doi.org/10.1039/D3TA00470H). The resulting uptakes calculated in the previous step of this high-throughput screening are summarized in the `results.csv` file. The materials that are included in the train and test sets are listed in the `structs_train.txt` and `structs_test.txt` files, respectively. Three steps can be recognized in the ML workflow:

### Step 2a: ML algorithm training

For each of the 25 ML models, the combination of best-performing hyperparameters is selected with 10-fold cross-validation. After running the `run.py` script with the `train` argument, the cross-validation results are stored in the `train_MODELTARGET.csv` files, where `MODEL` is the adopted ML algorithm, and `TARGET` determines the predicted adsorption property:

- `TARGET` = `0`: Uptake of CO<sub>2</sub> at 298 K and 10 bar
- `TARGET` = `1`: Uptake of CO<sub>2</sub> at 298 K and 0.1 bar
- `TARGET` = `2`: Uptake of N<sub>2</sub> at 298 K and 10 bar
- `TARGET` = `3`: CO<sub>2</sub> working capacity
- `TARGET` = `4`: Ideal CO<sub>2</sub>/N<sub>2</sub> selectivity

**input**

`features.csv`, `results.csv`, `structs_train.txt`, `structs_test.txt`

**command line**

`python run.py train`

**output**

`train_MODELTARGET.csv`

### Step 2b: ML algorithm predictions

Once the ideal hyperparameters are selected, the ML algorithms can be adopted to predict the different adsorption properties. This can be executed by running the `run.py` script with the `predict` argument. The resulting predictions for each model and target are stored in the `test_MODELTARGET.csv` and `exp_MODELTARGET.csv` files. The files with the prefix `test_` contain the ML predictions on the 4997 COFs in the test set, together with their actual GCMC values, whereas the files with the prefix `exp_` contain the ML predictions for all 268 678 COFs in the ReDD-COFFEE database.

**input**

`features.csv`, `results.csv`, `structs_train.txt`, `structs_test.txt`

**command line**

`python run.py predict`

**output**

`test_MODELTARGET.csv`, `exp_MODELTARGET.csv`

### Step 2c: SHAP analysis

To gain insight in the importance of each feature in the ML prediction, a SHAP analysis is performed on the test set. For each material, this analysis assigns to each feature a SHAP value that determines how much that specific feature contributes to the predicted outcome. Adding all SHAP values to the mean expected outcome provides the actual prediction of the ML algorithm. 

**input**

`features.csv`, `results.csv`, `structs_train.txt`, `structs_test.txt`

**command line**

`python run.py shap`

**output**

`shap_MODELTARGET.csv`

All SHAP values for all 4997 materials in the test set are provided in the `shap_MODELTARGET.csv` files. The overall importance of each feature in the target prediction is obtained by taking the mean of the absolute SHAP values for that feature over all material instances.  The figures that are provided in Section S3.4 can be generated by running `python plot_shap.py`.

## Step 3 - Mixture screening

In the third step of the high-throughput screening, the working capacity and mixture selectivity are calculated for a promising set of 3305 COFs. To be able to compare the ideal and mixture selectivities, a total of 4 GCMC calculations have to be performed. For the materials that were already present in the diverse screening set of Step 1, some of these calculations have already been executed.

- `co2_ads`: Uptake of CO<sub>2</sub> at 298 K and 10 bar
- `co2_des`: Uptake of CO<sub>2</sub> at 298 K and 0.1 bar
- `n2_ads`: Uptake of N<sub>2</sub> at 298 K and 10 bar
- `mix`: Uptake of a binary mixture of CO<sub>2</sub>/N<sub>2</sub> (composition 15:85) at 298 K and 10 bar

**input**

`COF.cif`, `co2.def`, `n2.def`,  `pseudo_atoms.def`, `force_field_mixing_rules.def`, `force_field.def`, `simulation.input`, `restart.input`

**command line**

`simulate -i simulation.input > raspa.log 2> raspa.err`
( or `simulate -i restart.input > raspa.log 2> raspa.err` if the job had to be restarted)

**output**

`raspa.log`, `raspa.err`, `CrashRestart`, `Restart`, `Movies`, `VTK`, `Output`

All input folders with the respective input files are stored in the `input.tar.gz` file, whereas all RASPA output files are provided in the `output.tar.gz` file. The resulting uptakes and enthalpies are stored in the `data` folder using the [AIF (adsorption information format)](https://adsorptioninformationformat.com/) format.

## Step 4 - Analysis

In the last step of the high-throughput screening, the CO<sub>2</sub> six best-performing 2D and 3D COFs are visualized. Therefore, a mixture calculation is performed for each material with a higher number of Monte Carlo steps (100 000).

**input**

`COF.cif`, `co2.def`, `n2.def`,  `pseudo_atoms.def`, `force_field_mixing_rules.def`, `force_field.def`, `simulation.input`, `restart.input`

**command line**

`simulate -i simulation.input > raspa.log 2> raspa.err`
( or `simulate -i restart.input > raspa.log 2> raspa.err` if the job had to be restarted)

**output**

`raspa.log`, `raspa.err`, `CrashRestart`, `Restart`, `Movies`, `VTK`, `Output`

All input folders with the respective input files are stored in the `input.tar.gz` file. In this case, the visualization files in the `Movies` folder are most interesting, since they contain the location of the CO<sub>2</sub> molecules at each step. These files have the title `Movie_cof_unitcells_298.000000_1000000.000000_component_co2_0.pdb`, where `cof` is the COF name and `unitcells` determines the number of unit cells in the simulation domain. 

The `Movie.pdb` file can be converted to a binary NumPy file, _i.e._ `density_co2.npy`, containing the CO<sub>2</sub> density using the `convert_xyz_npy.py` script. Afterwards, the density is stored in the `density_co2.cube` file, together with the COF atomic positions, which can be read by VMD or VESTA for visualization. This file can be generated with the `writecube_v2.py` script.

**command line**

`python convert_xyz_npy.py`
`python writecube_v2.py`

The relevant output files are stored in the `output.tar.gz` file.

