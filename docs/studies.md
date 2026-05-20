# Introduction

The purpose of this package is to centrally manage and execute common DFT studies performed with VASP. A _study_ in this context refers to a set of calculations, where each one corresponds to a different system or set of input parameters. Some examples include fitting an $E(V)$ equation of state, which requires calculating the energy of a reference system for different degrees of volume scaling, or evaluating the convergence of the basis set cutoff energy parameter `ENCUT`. 

Typically one would have to create a large amount of files and directories, tweak input parameters between calculations, and compile data for analysis or downstream use. This package aims to eliminate this tedious, and potentially error-prone, process while maximizing reproducibility via a single long-form input file. By nature of design, this file contains every detail related to how VASP was run for all calculations.

A study is broken into three levels of organization:
1. __Study__: dictionary-like parameters detailing the study (e.g., name, type, ...)
2. __Calculations__: specific systems or input parameter combinations (e.g., TiO2, TiO2 w/ O-vacancy, ...)
3. __Steps__: indivudual VASP jobs performed in sequence (e.g., non-spin-polarized preconvergence, then spin-polarized, then high-accuracy single point)

While every type of study follows the same template, they may deviate as necessary according to their nature. Currently, the following studies are available
- `Individual` = one system is treated in isolation
- `EosFit` = fit the Birch-Murnaghan equation of state
- `PointDefectFormation` = calculate formation energy of point defects like vacancies and substitutional impurities

For each study, the following template below is used. 

In the `calculations` section, all calculations must at least share a common `POSCAR` file, indicated by being a top-level key. Any other input file can be common and thus should be excluded from the individual calculation sections. This is mirrored by the `steps` section, only either all calculations share the same set of steps, or all calculations have their own specific steps. Within a given step, it contains either (1) modifications to be made to the input file from the previous step, or (2) it contains full input files that completely overwrite the version from the previous step. 

A special case is the first step, where it should only contain a name. This is because its input files are defined as the common files + calculation-specific files. The files specified in `calculations` thus describe the structure of the first step.

Finally, to maintain input file size and portability, `POTCAR` is defined as a list of directory names (e.g., `Ni_pv`, `O`, `W_sv`). This tells the code which files to load for either all calculations if its common (i.e., a top-level key), or which files to use for specific calculations. For example, you may want to use `Ti` in a perfect TiO2 lattice, but use `Ti_pv` in the same lattice with an O-vacancy to account for participating semi-core electrons. 

Examples are provided in the `examples.md` documentation file.

```yaml
study:
    name: <name of study>
    dir: <full path to parent directory which will contain the study>
    type: <class name of study>
    param1: <val>
    param2: <val>
    ...

calculations:
    POSCAR: |
        ...

    calc_1:
        INCAR: |
            ...
        KPOINTS: |
            ...
        POTCAR: |
            <directory name of species 1>
            <directory name of species 2>
            ...

    calc_2:
        INCAR: |
            ...
        KPOINTS: |
            ...
        POTCAR: |
            <directory name of species 1>
            <directory name of species 2>
            ...

    ...

steps:
    1:
        name: <description of step 1>
    2:
        name: <description of step 2>
        INCAR: |
            <full input file contents>
    ...

    # OR

    calc_1:
        1:
            name: <description of step 1>
            INCAR:
            - Add: <tag definition to add or overwrite>
            - Remove: <tag to remove>
        2:
            name: <description of step 2>
            INCAR: |
                ...
        ...

    calc_2:
        1:
            name: <description of step 1>
            INCAR:
            - Add: <tag definition to add or overwrite>
            - Remove: <tag to remove>
        2:
            name: <description of step 2>
            INCAR: |
                ...
        ...
    ...
```

# `Individual`

This is the simplest study and consists of running an individual calculation. As such, it contains no special study parameters and all files/steps should be common.

```yaml
study:
    name: <name of study>
    dir: <full path to parent directory which will contain the study>
    type: Individual

calculations:
    INCAR: |
        ...
    KPOINTS: |
        ...
    POSCAR: |
        ...
    POTCAR: |
        <directory name of species 1>
        <directory name of species 2>
        ...

steps:
    1:
        name: <description of step 1>
    2:
        name: <description of step 2>
    ...
```

# `EosFit`

This study fits the Birch-Murnaghan equation of state given a set of linear scaling factors for a given system. 

Since every system is effectively the same (besides volume), there should only be common files/steps like in `Individual`. This is also typically a time-consuming process, so the `dir` parameter doubly serves as a restart directory path if a previous job failed or otherwise did not finish. Whenever this study is run, progress is saved in a file called `eosfit.restart`, which will be searched for as a top-level file within in the directory given by `dir`. If such a file is found, the system and calculation parameters and the fitting process will be restarted from the last completed scaling factor. This file also contains the volume and final energy of each calculation.

```yaml
study:
    name: <name of study>
    dir: <full path to parent directory or the directory of partially finished EosFit study>
    type: EosFit
    scaling: <list of floats describing how many systems to calculate and their linear scaling factors>

calculations:
    INCAR: |
        ...
    KPOINTS: |
        ...
    POSCAR: |
        ...
    POTCAR: |
        <directory name of species 1>
        <directory name of species 2>
        ...

steps:
    1:
        name: <description of step 1>
    2:
        name: <description of step 2>
    ...
```

# `PointDefectFormation`

This study calculates the formation energy of a vacancy (`vac`) or substitutional impurity (`sub`).

The study begins by relaxing the perfect/pristine lattice (relative to the defective version) according to the input files specificed in `calculations/perfect`. A defect is then inserted into the `CONTCAR` file and that acts as the `POSCAR` for the defective lattice. Most likely, however, the perfect system has already been relaxed previously. In this case, the `perfect` study parameter provides the path to the directory where VASP was run, containing `CONTCAR` and `OUTCAR` files. To use the default behavior (relaxing the perfect system first), simply exclude the `perfect` parameter from the input YAML file. Note, to maintain reproducibility, the perfect system parameters should be present even if it has been relaxed separately and the `POSCAR` should still correspond to the unrelaxed perfect lattice.

A special case is made for spin-polarized calculations. Upon introducing the defect, an additional initial magnetic moment will be required for interstitials and impurities, while one less magnetic moment will be required for vacancies. This is based on the order of the coordinates and species in the `POSCAR` file. If the starting `INCAR` contains the `MAGMOM` tag, the code automatically adjusts this line for the first and subsequent steps. If, however, spin-polarization is turned on in the second step or beyond, it is your responsibility to specify the `MAGMOM` line correctly. For example, if an O-vacancy is introduced in a 2x2x3 supercell of rutile TiO2 (24 Ti, 48 O), the `MAGMOM` line will need to be `MAGMOM = 24*<Ti MAGMOM> 47*<O MAGMOM>`. The `magmom` study parameter defines the magnetic moment for impurity defects, but can be excluded for vacancies and self-interstitials.

As described previously, the `POTCAR` file can be specified as a list common to all calculations, or it can be defined specific to each calculation. For example, suppose the rutile TiO2 system is doped with a single `W` atom. If `POTCAR` is common to all calculations (i.e., a top-level key in the `calculations` section) like `Ti`, `O`, `W_sv`, both perfect and defective systems will use these `POTCAR` files. If instead each calculation has their own `POTCAR` list, where `calculations/perfect` could have `Ti`, `O`, and `calculations/defective` could have `Ti_pv`, `O`, `W_sv`, then the each calculation will use the `POTCAR` files specific to themselves.

Finally, the formation energy is defined as $$E^f[X^q] = E_\text{tot}[X^q] - E_\text{tot}[\text{bulk}] - \sum n_i\mu_i + qE_F + E_\text{corr}.$$ The only external parameters are the chemical potential $\mu_i$ and the finite-size correction term $E_\text{corr}$. The Fermi energy, defect charge, and electronic structure energy difference terms are found internally from the `OUTCAR` files, while the `chemical_pot` and `fs_correction` study parameters can be included to account for these terms. By default, the chemical potential is defined as the energy per atom in the perfect lattice $E_\text{tot}[X^q]/N$, which may only be valid for unary metal systems (e.g., Fe, Ni). The default value for the finite-size correction term is $E_\text{corr}=0$.

```yaml
study: 
    name: <name of study>
    dir: <full path to parent directory>
    type: PointDefectFormation
    perfect: <optional, path to already relaxed perfect system containing CONTCAR and OUTCAR>
    defect: vac, sub
    species: <optional, element of substitutional impurity>
    magmom: <optional, initial magnetic moment for the substitutional impurity if doing spin-polarization>
    position: <approximate location of site to insert defect at (e.g., 0.5 0.5 0.5)>
    chemical_pot: <optional, value of chemical potential (eV) for point defect>

calculations:
    POSCAR: |
        ...

    perfect:
        INCAR: |
            ...
        KPOINTS: |
            ...
        POTCAR: |
        ...

    defective:
        INCAR: |
            ...
        KPOINTS: |
            ...
        POTCAR: |
        ...
    
steps:
    perfect:
        1:
            name: ...
        2:
            name: ...
        ...
    
    defective:
        1:
            name: ...
        2:
            name: ...
        ...
```

# Special steps: density of states and band structure

Every calculation as the option to add some post-processing steps which calculate and plot the electron density of states (eDOS) and/or band structure. This is achieved by providing the `dos` and `band` steps, following the same structure described in previous sections. In either case, entire `INCAR` and `KPOINTS` files must be provided, since these steps can be very specific.

```yaml
study:


calculations:
    INCAR: |
        ...
    KPOINTS: |
        ...
    POSCAR: |
        ...
    POTCAR: |
        <directory name of species 1>
        <directory name of species 2>
        ...
    
    dos:
        INCAR: |
            ...
        KPOINTS: |
            ...
    
    band:
        INCAR: |
            ...
        KPOINTS: |
            ...


steps:
    1:
        name: <descriptive name of step 1>
    2:
        name: <descriptive name of step 1>

    dos:
        INCAR: |
            ...
        KPOINTS: |
            ...
    
    band:
        KPOINTS_OPT: |
            ...
```