# Introduction

The purpose of this package is to centrally manage and execute common DFT studies performed with VASP. A _study_ in this context refers to a set of calculations, where each one corresponds to a different system or set of input parameters. Some examples include fitting an $E(V)$ equation of state, which requires calculating the energy of a reference system for different degrees of volume scaling, or evaulating the convergence of the basis set cutoff energy parameter `ENCUT`. 

Typically one would have to create a large amount of files and directories, tweak input parameters between calculations, and compile produced data for analysis or downstream use. This package aims to eliminate this tedious, and potentially error-prone, process while maximizing reproducibility via a single long-form input file. By nature of design, this file contains every detail related to how VASP was run for all calculations.

A study is broken into three levels of organization:
1. __Study__: dictionary-like parameters detailing the study (e.g., name, type, ...)
2. __Calculations__: specific systems or input parameter combinations (e.g., TiO2, TiO2 w/ O-vacancy, ...)
3. __Steps__: indivudual VASP jobs performed in sequence (e.g., non-spin-polarized preconvergence, then spin-polarized, then high-accuracy single point)

While every type of study follows the same template, they may deviate as necessary according their nature. Currently, the following studies are available
- `Individual` = one system is treated in isolation
- `EosFit` = fit the Birch-Murnaghan equation of state
- `Benchmark` = compare convergence speed for different KPAR and NCORE settings
- `PointDefectFormation` = calculate formation energy of point defects like vacancies and substitutional impurities

For each study, the following template is used. It is important to note that `steps` section details modifications to the input `INCAR` between VASP runs. Also note that the `calculations` and `steps` sections may be further divided up by system type when different `INCAR` and `KPOINTS` are desired for the some specific studies like `PointDefectFormation`.

```yaml
study:
    name: <name of study>
    dir: <full path to parent directory>
    type: <class name of study>
    param1: <val>
    param2: <val>
    ...

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
        name: <descriptive name of step 1>
        INCAR:
        - Add: <tag definition to add or overwrite>
        - Remove: <tag to remove>
    2:
        name: <descriptive name of step 1>
        INCAR:
            ...
    ...
```

# `Individual`

This is the simplest study and consists of running a calculation for an individual system. As such, it contains no study parameters.

```yaml
study:
    name: <name of study>
    dir: <full path to parent directory>
    type: Individual
```

# `EosFit`

This study fits the Birch-Murnaghan equation of state given a set of linear scaling factors for a given system. 

Since this can be a time-consuming process, the `dir` parameter also serves as a restart path if a previous job failed or otherwise did not finish. Whenever this study is run, progress is saved in a file called `eosfit.restart`, which will be searched for within in the directory given by `dir`. If such a file is found, the system and calculation parameters will be checked for consistency (not yet implemented) and the fitting process will be restarted from the last completed scaling factor, otherwise the study will start from scratch in a new directory.

```yaml
study:
    name: <name of study>
    dir: <full path to parent directory>
    type: EosFit
    scaling: [0.96, 0.97, ..., 1.03, 1.04]
```

# `Benchmark`

```yaml
parameters: 
    type: Benchmark
    max_kpar: <int>
    cores: <int>
    account: <name of Roar Collab account to run jobs with>
    max_time: <hrs:min:sec>
```

# `PointDefectFormation`

This study calculates the formation energy of a vacancy (`vac`) or substitutional impurity (`sub`).

While the default behavior is to relax both the perfect and defective systems, most likely the perfect system has already been relaxed previously. In this case, the `perfect` parameter provides the path to the directory where VASP was run, containing the CONTCAR and OUTCAR files. Since the relaxed ion positions are already defined via the CONTCAR file, the `POSCAR` section is ignored and the defective system uses the found CONTCAR file. To use the default behavior (relaxing the perfect system first), simply exclude the `perfect` parameter from the input YAML file. Note, to maintain reproducibility, the perfect system parameters should be present even if it has been relaxed separately. Consistency between the input system and calculation parameters for the present input file and the imported directory are thus checked (not implemented yet).

For impurity defects, the element type of the impurity is provided by the `species` parameter. This is used to update the POSCAR species and to load a POTCAR for it. If a specific POTCAR is desired (e.g., X_sv), it can be included in the `POTCAR` section. Order does not matter. Additionally, if spin-polarization is turned on, the impurity initial magnetic moment can be provided via the `magmom` parameter.

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
    POTCAR: |
        ...

    perfect:
        INCAR: |
            ...
        KPOINTS: |
            ...

    defective:
        INCAR: |
            ...
        KPOINTS: |
            ...
    
steps:
    perfect:
        1:
            name: ...
            tags:
                ...
        2:
            name: ...
            tags:
                ...
        ...
    
    defective:
        1:
            name: ...
            tags:
                ...
        2:
            name: ...
            tags:
                ...
        ...
```


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
    POTCAR: |
        ...

    perfect:
        INCAR: |
            ...
        KPOINTS: |
            ...

    defective:
        INCAR: |
            ...
        KPOINTS: |
            ...
    
steps:
    1:
        name: <descriptive name of step 1>
        INCAR:
        - Add: <tag definition to add or overwrite>
        - Remove: <tag to remove>
    2:
        name: <descriptive name of step 1>
        INCAR:
            ...
    ...
```

# Calculating Density of States and Band Structure

Every study has the option to add some post-processing steps which calculate and plot the electron density of states and band structure for a given calculation. This is achieved by enabling the `dos` and `bands` study parameters, then including the corresponding input files that will be executed in the `calculations` section.

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
    
    bands:
        INCAR: |
            ...
        KPOINTS: |
            ...


steps:
    1:
        name: <descriptive name of step 1>
        INCAR:
        - Add: <tag definition to add or overwrite>
        - Remove: <tag to remove>
        KPOINTS: |
            ...
    2:
        name: <descriptive name of step 1>
        INCAR:
            ...
    ...
```



# Examples