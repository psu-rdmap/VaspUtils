# Types of Studies Available

A study consists of running calculations for different systems. For example, when fitting an $E(V)$ equation of state on a given system, calculations are performed for varying degrees of compression and expansion. 

For each calculation, it may be desirable to break it up into sequential steps, such as
1. Geometry relaxition without spin-polarization
2. Continue with spin-polarization turned on and magnetic moments initialized
3. Conclude with a single-point calculation and write density of states data

A VaspUtils input file therefore contains two sections for describing the study to perform and the calculation steps to repeat for each system.

### `study`

Contains information about the study including its name, the parent directory, type, study-specific parameters, the contents of the initial INCAR, POSCAR, KPOINTS files, and the directory names where the desired POTCAR files reside. The following sections provide details specific to each study. The syntax should follow the template below.

```yaml
study:
    name: <name of study>
    dir: <full path to parent directory>
    parameters:
        type: <class name of study>
        param1: <val>
        param2: <val>
        ...
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
```

### `calculation`

Contains information about the steps of a calculation including each step's name and modifications for a specific VASP input file. The syntax should follow the template below.

```yaml
calculation:
    1:
        name: <name of first step>
        INCAR:
        - Add: <tag definition to add or overwrite>
        - Remove: <tag to remove>
        POSCAR:
        - L<#>: <line number to overwrite>
    2:
        name: <name of second step>
        INCAR:
            ...
        POSCAR:
            ...
    ...
```

## `Individual`

```yaml
parameters: 
    type: Individual
```

## `EosFit`

```yaml
parameters: 
    type: EosFit
    scaling: [sf1, sf2, ...]
```

## `Benchmark`

```yaml
parameters: 
    type: Benchmark
    max_kpar: <int>
    cores: <int>
    account: <name of Roar Collab account to run jobs with>
    max_time: <hrs:min:sec>
```

## `PointDefectFormation`

This study calculates the formation energy of a vacancy or self-interstitial.

While the default behavior is to relax both the perfect and defective systems, most likely the perfect system has already been relaxed. In this case, the `perfect` parameter provides the path to the directory where VASP was run, containing the CONTCAR and OUTCAR files. Since the relaxed ion positions are already defined via the CONTCAR file, the `POSCAR` section is ignored and the defective system uses the found CONTCAR file. To use the default behavior, simply exclude the `perfect` parameter from the input YAML file.

```yaml
parameters: 
    type: PointDefectFormation
    perfect: <optional, path to already relaxed perfect system containing CONTCAR and OUTCAR>
    defect: vac
    position: <approximate location of site to insert defect at (e.g., 0.5 0.5 0.5)>
    chemical_pot: <optional, value of chemical potential (eV) for point defect>
```

## Examples