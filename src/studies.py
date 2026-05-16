from vasp_file import VaspFile, VaspIncar, VaspPoscar, VaspKPoints, VaspPotcar, VaspOutcar, VaspContcar
from utils import next_path, wipe_directory, strip_split
from pathlib import Path
import subprocess, time, logging
from ase.eos import EquationOfState
from ase.units import kJ
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import deepcopy

logger = logging.getLogger('VaspUtils')
logging.getLogger("matplotlib").setLevel(logging.FATAL)
logging.getLogger("ase").setLevel(logging.FATAL)

class Study:
    """Baseclass for a DFT study which consists of at least one set of VASP calculations."""
    def __init__(self, input_yml: dict[str, dict]):
        self.name_str = input_yml['study']['name']
        self.parent_dir_path = Path(input_yml['study']['dir'])
        self.dir_path = None
        self.params = input_yml['study']['parameters']
        self.calculation_params = input_yml['calculation']

        # get POTCAR names to define the file
        self.potcar_names = [n.strip() for n in input_yml['study']['POTCAR'].split('\n')]

        # load VASP input files from user input
        self.incar = VaspIncar(contents_str=input_yml['study']['INCAR'])
        self.poscar = VaspPoscar(contents_str=input_yml['study']['POSCAR'])
        self.kpoints = VaspKPoints(contents_str=input_yml['study']['KPOINTS'])
        self.potcar = VaspPotcar(self.potcar_names, self.poscar)        
        logger.debug('Initialized VASP input file objects from user input')

        # build directory and run vasp (implemented by subclasses)
        self.subdir_paths = {}

    def build_directory(self):
        """Build directory specific to each Study subclass."""
        pass

    def write_input_files(self, dir_path: Path, overwrite_path=False):
        """Write current lines of input files to a files in a given directory."""
        self.incar.write_to_file(dir_path / 'INCAR', overwrite_path=overwrite_path)
        self.poscar.write_to_file(dir_path / 'POSCAR', overwrite_path=overwrite_path)
        self.kpoints.write_to_file(dir_path / 'KPOINTS', overwrite_path=overwrite_path)
        self.potcar.write_to_file(dir_path / 'POTCAR', overwrite_path=overwrite_path)

    def load_input_files(self, dir_path: Path):
        """Load lines of input files in a given directory."""
        self.incar = VaspIncar(file_path = dir_path / 'INCAR')
        self.poscar = VaspPoscar(file_path=dir_path / 'POSCAR')
        self.kpoints = VaspKPoints(file_path=dir_path / 'KPOINTS')
        self.potcar = VaspPotcar(self.potcar_names, self.poscar)
    
    def run_vasp(self, run_path: Path):
        """Run VASP in the background and perform any necessary supporting operations."""
        num_steps = len(self.calculation_params.keys())
        for step_num, step_params in self.calculation_params.items():
            logger.debug(f"({step_num}/{num_steps}) Running calculation: {step_params['name']}")
            # update input files with calculation step parameters
            for key in step_params.keys():
                if key == 'name':
                    pass
                else:
                    logger.debug(f'Updating input file {key}')
                    self.update_input_file(key, step_params[key])

            # initialize steps directory
            steps_dir = run_path / 'steps'
            steps_dir.mkdir(exist_ok=True)

            # run vasp in the background
            vasp_out = open(run_path / 'vasp.out', 'a')
            vasp_cmd = ['srun', '--kill-on-bad-exit', '--cpu-bind=cores', 'vasp_std']
            vasp = subprocess.Popen(vasp_cmd, cwd=run_path, stdout=vasp_out, stderr=subprocess.STDOUT)
            logger.debug(f'VASP launched')

            # load CONTCAR after the first scf cycle
            contcar_loaded = False
            contcar_path = run_path / 'CONTCAR'
            while contcar_loaded is False:
                time.sleep(1)
                if not contcar_path.exists():
                    continue
                with open(contcar_path, 'r') as f:
                    lines = f.readlines()
                if len(lines):
                    contcar_loaded = True
            self.contcar = VaspContcar(file_path = run_path / 'CONTCAR')

            # continuously save CONTCAR as it updates every ionic step
            while vasp.poll() is None:
                time.sleep(1)
                if self.contcar.check_updated():
                    time.sleep(0.5) # wait a moment to prevent read-write race
                    self.contcar.write_to_file(next_path(steps_dir / 'CONTCAR'))
            vasp.wait()
            vasp_out.close()

            # update poscar file at the end
            self.contcar.write_to_file(run_path / 'POSCAR')
            self.poscar.load_from_file(run_path / 'POSCAR')

    def update_input_file(self, file_name: str, new_lines: list[dict]):
        """Given an input file name, update the corresponding instance lines with a list of new lines and the desired operation (add, remove, overwrite line number, ...)."""
        for line in new_lines:
            action = next(iter([k for k in line.keys()]))
            logger.debug(f'Action: {action}')
            logger.debug(f'Line: {line[action]}')

            # append the current line to the end
            if action == 'Add':
                if file_name == 'INCAR':
                    self.incar.append_line(str(line[action]))

            # overwrite the line with a given line number (e.g., 'L43' is Line 43 and the index would be 42)
            elif action[0] == 'L':
                if file_name == 'POSCAR':
                    line_number = int(action[1:]) - 1
                    self.poscar.overwrite_line(line_number, str(line[action]))

study_registry: dict[str, Study] = {}
def register_study(cls):
    """Registry enrollment so that Study subclasses can be instantiated by string name."""
    study_registry[cls.__name__] = cls
    return cls

@register_study
class Individual(Study):
    """Simplest study consisting of one set of VASP calculations in a single directory."""
    def build_directory(self):
        """Single directory with no subdirectories."""
        self.dir_path = next_path(self.parent_dir_path / 'individual')
        self.dir_path.mkdir()
        self.write_input_files(self.dir_path, overwrite_path=True)
    
    def run_vasp(self):
        super().run_vasp(self.dir_path)
 
@register_study
class EosFit(Study):
    """Calculate energy for scaled up/down supercells and fit a V(E) equation of state."""
    def __init__(self, input_yml):
        super().__init__(input_yml)
        # special keywords
        try:
            self.finished = self.params['finished']
            self.resume_dir = self.params['resume']
        except:
            self.finished = []
            self.resume_dir = None

    def build_directory(self):
        """Subdirectories for each scale factor."""
        if self.resume_dir:
            self.dir_path = self.parent_dir_path / self.resume_dir
            logger.debug(f'Resuming from {self.dir_path}')
        else:
            self.dir_path = next_path(self.parent_dir_path / 'eos')
        self.dir_path.mkdir(exist_ok=True)

        for sf in self.params['scaling']:
            # create subdirectory
            subdir_path = self.dir_path / str(sf)
            subdir_path.mkdir(exist_ok=True)
            # cleanup directory and input files if sf has not already been run
            if sf not in self.finished:
                wipe_directory(subdir_path)
                self.write_input_files(subdir_path)
            self.subdir_paths[sf] = subdir_path

        # equilibrium subdirectory
        subdir_path = self.dir_path / 'eq'
        subdir_path.mkdir(exist_ok=True)
        if 'eq' not in self.finished:
            wipe_directory(subdir_path)
            self.write_input_files(subdir_path)

    def run_vasp(self):
        # calculate energies for fitting
        energies, volumes = [], []
        for sf in self.params['scaling']:
            # load input files in subdirectory and update POSCAR with current scaling factor
            subdir_path = self.dir_path / str(sf)
            self.load_input_files(subdir_path)
            self.update_input_file('POSCAR', [{'L2': sf}])
            logger.debug(f"Loaded input files for scale factor {sf}")
            # run vasp if it has not already been run
            if sf not in self.finished:
                super().run_vasp(subdir_path)
            else:
                logger.debug(f"Skipping calculations since {sf} has already run")
            # get volume and energy
            volumes.append(self.poscar.volume)
            logger.debug(f"Calculated volume: {self.poscar.volume}")
            self.outcar = VaspOutcar(file_path = subdir_path / 'OUTCAR')
            energies.append(self.outcar.get_energy())
            logger.debug(f"Calculated energy: {self.outcar.get_energy()}")

        # fit EoS
        eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
        eq_vol, eq_energy, bulk_mod = eos.fit()
        eos.plot(self.dir_path / 'eos.png')
        logger.debug(f"Birch-Murnaghan EOS fitted")
        
        # set up equilibrium volume supercell directory
        subdir_path = self.dir_path / 'eq'
        self.load_input_files(subdir_path)
        eq_vol_factor = eq_vol / self.poscar.volume
        eq_sf = self.poscar.scale_factor*(eq_vol_factor)**(1/3)
        self.update_input_file('POSCAR', [{'L2': eq_sf}])
        logger.debug(f"Loaded input files for equilibrium scale factor {eq_sf}")

        # calculate equilibrium energy
        if sf not in self.finished:
            super().run_vasp(subdir_path)
        else:
            logger.debug(f"Skipping equilibrium calculations since it has already run")

        # print out data
        self.outcar.load_from_file(subdir_path / 'OUTCAR')
        with open(self.dir_path / 'data.out', 'w') as d:
            d.write(f'Volumes: {volumes}\n')
            d.write(f'Energies: {[float(e) for e in energies]}\n')
            d.write(f'Equilibrium volume = {self.poscar.volume} A3\n')
            d.write(f'Equilibrium energy = {self.outcar.get_energy()} eV\n')
            d.write(f"Equilibrium lattice constant = {self.poscar.lattice_parameters['a']}")
            d.write(f'Bulk modulus: {bulk_mod / kJ * 1.0e24}')
            d.write(f'Average magnetic moments:')
            magmoms: dict[str, float] = self.outcar.get_magmom()
            if magmoms:
                for species, magmom in magmoms.items():
                    d.write(f'\t{species} = {magmom}')
        logger.debug(f"Printed fit data to {self.dir_path / 'data.out'}")

@register_study
class Benchmark(Study):
    """Benchmark KPAR and NCORE for a given test system."""
    def __init__(self, input_yml):
        super().__init__(input_yml)
        # determine allowable KPAR and corresponding NCORE values
        self.cores = min(self.params['cores'], 48)
        self.kpar: dict[int, list[int]] = {}
        logger.debug(f"Determining valid KPAR, NCORE combinations...")
        for k in range(1, self.params['max_kpar']+1):
            # allowable kpoint -> ncore_per_kpoint must be integer
            ncore_per_kpoint = self.cores / k
            if ncore_per_kpoint % 1 != 0:
                continue
            # allowable ncore -> ncore_per_band must be integer
            ncore = []
            for n in range(1, self.cores+1):
                ncore_per_band = ncore_per_kpoint / n
                if ncore_per_band % 1 == 0:
                    ncore.append(n)
            self.kpar[k] = ncore
            logger.debug(f"KPAR={k} works for NCORE={ncore}")
        
        # define job files lines
        max_time = self.params['max_time']
        max_time_str = f"{int(max_time/3600):02d}:{int((max_time/60)%60):02d}:{int(max_time%60):02d}"
        self.job = [
            "#!/bin/bash\n",
            f"#SBATCH --account={self.params['account']}\n",
            f"#SBATCH --nodes=1\n",
            f"#SBATCH --ntasks={self.params['cores']}\n",
            f"#SBATCH --mem-per-cpu=8GB\n",
            f"#SBATCH --time={max_time_str}\n",
            f"SECONDS=0\n",
            f"cd PATH\n",
            f"srun --export=ALL --kill-on-bad-exit --cpu-bind=cores vasp_std > vasp.out\n",
            f"ELAPSED=$SECONDS\n",
            f"echo $ELAPSED > time.out\n",
        ]
            
    def build_directory(self):
        """Subdirectories for each KPAR and NCORE calculation."""
        # top level directory
        self.dir_path = next_path(self.parent_dir_path / 'benchmark')
        self.dir_path.mkdir(exist_ok=True)
        # individual KPAR=K and NCORE=N directories
        for k, n_list in self.kpar.items():
            n_subdir_paths: dict[int, str] = {}
            for n in n_list:
                subdir_path = self.dir_path / f"KPAR={k}" / f"NCORE={n}"
                subdir_path.mkdir(parents=True)
                self.update_input_file('INCAR', [{'Add': f'KPAR = {k}'}, {'Add': f'NCORE = {n}'}])
                self.write_input_files(subdir_path)
                # update path in job file and write it
                self.job[7] = f"cd '{subdir_path}'\n"
                with open(subdir_path / 'run', 'w') as r:
                    r.writelines(self.job)
                n_subdir_paths[n] = subdir_path
            self.subdir_paths[k] = n_subdir_paths

    def run_vasp(self):
        """Schedule all jobs, wait until they are done, and extract elapsed time."""
        # submit jobs and increment counter
        num_unfinished_jobs = 0
        times: dict[int, dict[int, int]] = {}
        for k, n_list in self.kpar.items():
            n_times: dict[int, int] = {}
            for n in n_list:
                run_path: Path = self.subdir_paths[k][n] / 'run'
                vasp = subprocess.Popen(f"sbatch run", cwd=run_path.parent, stderr=subprocess.STDOUT)
                logger.debug(f"Submitted VASP job in {run_path}")
                num_unfinished_jobs += 1
                n_times[n] = 0
            times[k] = n_times

        # wait until all jobs are finished
        times = {}
        while num_unfinished_jobs:
            for k, n_list in self.kpar.items():
                for n in n_list:
                    check_fp: Path = self.subdir_paths[k][n] / 'time.out'
                    # VASP finished -> time.out created and the elapsed wall time is available
                    if check_fp.exists():
                        num_unfinished_jobs -= 1
                        with open(check_fp, 'r') as t:
                            time_line = t.readline()
                        elapsed = strip_split(time_line, '=')[-1]
                        times[k][n] = elapsed
                        logger.debug(f"({num_unfinished_jobs}) Job at {check_fp} finished. Wall time = {elapsed}s")
                    else:
                        time.sleep(1)

        # write times to an output file
        times_df = pd.DataFrame(times)
        with open(self.parent_dir_path / 'times.dat', 'w') as d:
            print(times_df, file=d)
        logger.debug(f"Wrote simulation times at {self.parent_dir_path / 'times.dat'}")

        # plot NCORE on x-axis, time on y-axis, and KPAR as different series
        baseline = times[1][1]
        for k, n_times in times.items():
            x, y = [], []
            for n, t in n_times.items():
                x.append(n)
                y.append(t / baseline)
            plt.plot(x, y, label=f'KPAR={k}')
        plt.xlabel('NCORE')
        plt.ylabel('Calculation Time (rel.)')
        plt.savefig(self.parent_dir_path / 'times.png')
        plt.close()
        logger.debug(f"Plotted NCORE and KPAR values and saved figure at {self.parent_dir_path / 'times.png'}")

        # plot best NCORE and KPAR
        x, y = [], []
        for k in times.keys():
            x.append(k)
            y.append(times_df[k].min() / baseline)
        plt.plot(x, y, label='KPAR')
        x, y = [], []
        for n in times[1].keys():
            x.append(n)
            y.append(times_df.loc[k].min() / baseline)
        plt.plot(x, y, label='NCORE')
        plt.xlabel('Tag Value')
        plt.ylabel('Best Calculation Time (rel.)')
        plt.savefig(self.parent_dir_path / 'best_times.png')
        logger.debug(f"Plotted best NCORE and KPAR values and saved figure at {self.parent_dir_path / 'best_times.png'}")

@register_study
class PointDefectFormation(Study):
    """Calculate formation for a vacancy or self-interstitial."""
    def __init__(self, input_yml):
        # if the perfect keyword was included, load this CONTCAR
        try:
            self.perfect_path = Path(input_yml['study']['parameters']['perfect'])
            perfect_contcar = VaspContcar(file_path = self.perfect_path / 'CONTCAR')
            perfect_poscar_lines = ''
            for line in perfect_contcar.lines:
                perfect_poscar_lines += line + '\n'
            input_yml['study']['POSCAR'] = deepcopy(perfect_poscar_lines)
            logger.debug(f"Path to relaxed perfect system provided. POSCAR loaded from {self.perfect_path / 'CONTCAR'}")
        except:
            self.perfect_path = None
            logger.debug(f"Path to relaxed perfect system could not be found or was not provided")
        super().__init__(input_yml)

    def build_directory(self):
        """Subdirectories for the perfect and defective system."""
        self.dir_path = next_path(self.parent_dir_path / (self.params['defect'] + '_formation'))
        # perfect system
        self.perfect_subdir_path = self.dir_path / 'perfect'
        self.perfect_subdir_path.mkdir(parents=True)
        self.write_input_files(self.perfect_subdir_path)
        # copy in OUTCAR if perfect system has already been relaxed
        if self.perfect_path:
            perfect_outcar = VaspOutcar(file_path = self.perfect_path / 'OUTCAR')
            perfect_outcar.write_to_file(self.perfect_subdir_path / 'OUTCAR')
            logger.debug(f"Perfect system OUTCAR copied from {self.perfect_path / 'CONTCAR'}")
        # insert the defect
        defect_pos = np.array(self.params['position'])
        if self.params['defect'] == 'vac':
            self.poscar.remove_ion(defect_pos, self.incar)
            logger.debug(f"Inserted vacancy near {str(defect_pos)}")
        elif self.params['defect'] == 'sub':
            rm_ion_pos = self.poscar.remove_ion(defect_pos, self.incar)
            try:
                magmom = self.params['magmom']
            except:
                magmom = None
            self.poscar.add_ion(self.params['species'], rm_ion_pos, self.incar, magmom=magmom)
            logger.debug(f"Inserted substitutional impurity near {str(defect_pos)}")
        # adjust INCAR
        self.update_input_file('INCAR', [{'Add': f'ISYM = 0'}]) # disable symmetry
        # update POTCAR in case species were added
        self.potcar = VaspPotcar(self.potcar_names, self.poscar)
        # defective system
        self.defective_subdir_path = self.dir_path / 'defective'
        self.defective_subdir_path.mkdir()
        self.write_input_files(self.defective_subdir_path)
    
    def run_vasp(self):
        # relax perfect system (if necessary)
        if self.perfect_path is None:
            logger.debug(f"Relaxing perfect system...")
            super().run_vasp(self.perfect_subdir_path)
        else:
            logger.debug(f"Perfect system already relaxed. Skipping it.")
        # relax defective system
        logger.debug(f"Relaxing defective system...")
        super().run_vasp(self.defective_subdir_path)
        # extract energies
        perfect_outcar = VaspOutcar(file_path = self.perfect_subdir_path / 'OUTCAR')
        perfect_energy = perfect_outcar.get_energy()
        logger.debug(f"Calculated perfect system energy: {perfect_energy} eV")
        defective_outcar = VaspOutcar(file_path = self.defective_subdir_path / 'OUTCAR')
        defective_energy = defective_outcar.get_energy()
        logger.debug(f"Calculated defective system energy: {defective_energy} eV")
        # define chemical potential if it is not provided using E/N (bulk metals)
        try:
            chemical_pot = self.params['chemical_pot']
        except:
            perfect_poscar = VaspPoscar(file_path = self.perfect_subdir_path / 'POSCAR')
            num_species = len(perfect_poscar.get_species()[-1])
            chemical_pot = perfect_energy / num_species
        logger.debug(f"Defined chemical potential: {chemical_pot} eV")
        # calculate formation energy
        if self.params['defect'] == 'vac':
            formation_energy = defective_energy - perfect_energy + chemical_pot
        else:
            formation_energy = defective_energy - perfect_energy - chemical_pot
        logger.debug(f"Calculated defect formation energy: {formation_energy} eV")
        # write data to a file
        with open(self.dir_path / 'data.out', 'w') as d:
            d.write(f'Perfect system energy: {perfect_energy} eV\n')
            d.write(f'Defective system energy: {defective_energy} eV\n')
            d.write(f'Chemical potential: {chemical_pot} eV\n')
            d.write(f'Defect formation energy: {formation_energy} eV\n')