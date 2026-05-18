from vasp_file import vasp_file_registry, VaspFile, VaspIncar, VaspPoscar, VaspKPoints, VaspPotcar, VaspOutcar, VaspContcar, VaspDoscar
from utils import next_path, wipe_directory, strip_split
from pathlib import Path
import subprocess, time, logging, yaml
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
    """Baseclass for a DFT study which consists of at least one calculation."""
    def __init__(self, input_yml: dict[str, dict]):
        self.input_yml = input_yml
        self.params = input_yml['study']
        self.calc_params = input_yml['calculations']
        self.steps_params = input_yml['steps']
        self.parent_dir = Path(input_yml['study']['dir'])
        self.dir = None
        
        self.name = self.params['name']
        logger.debug(f'Starting study: {self.name}')

        # define state dictionary organizing calculations and steps
        self.state, self.calc_ids, self.skip_calc_ids = {}, [], []
        self.init_state()

    def update_input_file(self, file: VaspFile, ops: list[dict]):
        for op in ops:
            action = next(iter([k for k in op.keys()]))
            if action == 'Add':
                file.append_line(str(op[action]))
            elif action == 'Remove':
                file.remove_line(str(op[action]))

    def write_input_files(self, dict_w_files: dict[str, VaspFile], write_dir: Path):
        for key, val in dict_w_files.items():
            if isinstance(val, VaspFile):
                val.write_to_file(write_dir / key)

    def init_state(self):
        """Initialize a dictionary containing all references required to run each calculation and their steps."""
        # define calc_ids, skip_calc_ids in subclass
        self.state = dict.fromkeys(self.calc_ids, {})
        
        # read in files from `calculations` section
        common_files: dict[str, VaspFile] = {}
        for key, val in self.calc_params.items():
            if fn in ['INCAR', 'KPOINTS', 'POSCAR']:
                common_files[fn] = vasp_file_registry[fn](contents_str=contents)

        if 'POSCAR' not in common_files.keys():
            raise KeyError('No POSCAR found in the `calculations` section`. All calculations must share a common POSCAR file')

        if 'POTCAR' in common_files.keys():
            common_files['POTCAR'] = VaspPotcar(contents_str=self.calc_params['POTCAR'], poscar=common_files['POSCAR'])

        self.state['common'] = common_files
        logger.debug('Read in common VASP files from user input')

        # steps for each calculation
        for calc_id in self.calc_ids:
            calc_dict: dict = self.state[calc_id]
            steps_params: dict = deepcopy(calc_dict['step_params'])
            step_dict: dict = deepcopy(steps_params[step_id])

            # option to skip calculation
            if calc_id in self.skip_calc_ids:
                continue

            # add files shared by all calculations to first step (should be just a name at this point)
            for fn, file in self.state['common'].items():
                if fn in ['INCAR', 'POSCAR', 'KPOINTS', 'POTCAR']:
                    step_dict[fn] = deepcopy(file)

            # save step
            calc_dict[1] = step_dict
            logger.debug(f'Defined step 1 for {calc_id}')

            # read next integer-labelled steps (2, 3, ...)
            num_steps = len([k for k in steps_params.keys() if type(k) == int])
            for step_id in range(2, num_steps+1):
                step_dict: dict = deepcopy(steps_params[step_id])

                # replace file modifications with a VaspFile object
                for key, val in step_dict.items():
                    # only INCAR and KPOINTS should be overwritten between steps
                    if key in ['POSCAR', 'POTCAR']:
                        raise KeyError('Steps may only modify INCAR and KPOINTS files')

                    # user gives either modifications to file, or the entire file
                    if type(val) == list:
                        step_dict[key] = self.update_input_file(file=calc_dict[step_id-1][key], ops=val)
                    else:
                        step_dict[key] = vasp_file_registry[fn](contents_str=val)

                    # save step
                    calc_dict[step_id] = step_dict
                    logger.debug(f'Defined step {step_id} for {calc_id}')
            
            # optional dos or band post-processing steps
            try:
                dos_dict: dict = self.steps_params['dos']
                for fn, contents in dos_dict.items():
                    dos_dict[fn] = vasp_file_registry[fn](contents_str=contents)
                calc_dict['dos'] = dos_dict
            except:
                pass

            try:
                bands_dict: dict = self.steps_params['bands']
                for fn, contents in bands_dict.items():
                    bands_dict[fn] = vasp_file_registry[fn](contents_str=contents)
                calc_dict['bands'] = bands_dict
            except:
                pass

            # save calculation
            self.state[calc_id] = calc_dict
    
    def build_directory(self):
        """Build directory specific to each subclass. Assign a directory path to each calculation."""
        pass

    def run_vasp(self, calc_id):
        """Runs a calculation and its steps."""
        calc_dict: dict = self.state[calc_id]
        run_dir = calc_dict['dir']

        # write common files before running steps
        self.write_input_files(self.state['common'], run_dir)

        # run each step
        num_steps = len([k for k in calc_dict.keys() if type(k) == int])
        for step_id in range(1, num_steps+1):
            step_dict = calc_dict[step_id]
            
            # write non-common and modified input files
            self.write_input_files(step_dict, run_dir)

            # run vasp in the background
            vasp_out = open(run_dir / 'vasp.out', 'a')
            vasp_cmd = ['srun', '--kill-on-bad-exit', '--cpu-bind=cores', 'vasp_std']
            vasp = subprocess.Popen(vasp_cmd, cwd=run_dir, stdout=vasp_out, stderr=subprocess.STDOUT)
            logger.debug(f'VASP launched')

            # load CONTCAR after the first scf cycle
            contcar_loaded = False
            contcar_path = run_dir / 'CONTCAR'
            while contcar_loaded is False:
                time.sleep(1)
                if not contcar_path.exists():
                    continue
                with open(contcar_path, 'r') as f:
                    lines = f.readlines()
                if len(lines):
                    contcar_loaded = True
            contcar = VaspContcar(file_path = run_dir / 'CONTCAR')

            # continuously save CONTCAR as it updates every ionic step
            i = 1
            with open(run_dir / 'CONTCAR_steps', 'a') as f:
                f.write('0\n\n')
                f.writelines([l+'\n' for l in contcar.lines])
                f.write('\n')

                while vasp.poll() is None:
                    time.sleep(1)
                    if contcar.check_updated():
                        time.sleep(0.5) # wait a moment to prevent read-write race
                        f.write(f'{i}\n\n')
                        f.writelines([l+'\n' for l in contcar.lines])
                        f.write('\n')
                vasp.wait()
                vasp_out.close()

            # update poscar file at the end
            contcar.write_to_file(run_dir / 'POSCAR')

            # write out input YAML for reference
            with open(run_dir / 'input.yml', 'w') as f:
                yaml.dump(self.input_yml, f)
        
        # special steps at the end
        try:
            self.write_input_files(calc_id['dos'], run_dir)
            logger.debug(f'DOS input files provided. Doing calculation...')
            with open(run_dir / 'vasp.out', 'a') as vasp_out:
                vasp = subprocess.run(vasp_cmd, cwd=run_dir, stdout=vasp_out, stderr=subprocess.STDOUT)
            doscar = VaspDoscar(file_path=run_dir / 'DOSCAR')
            doscar.plot(run_dir / 'dos.png')
            logger.debug(f'DOS calculation done')
        except:
            logger.debug(f'Skipping DOS calculation')

        try:
            self.write_input_files(calc_id['bands'], run_dir)
            logger.debug(f'Band structure input files provided. Doing calculation...')
            with open(run_dir / 'vasp.out', 'a') as vasp_out:
                vasp = subprocess.run(vasp_cmd, cwd=run_dir, stdout=vasp_out, stderr=subprocess.STDOUT)
            logger.debug(f'Band structure calculation done')
        except:
            logger.debug(f'Skipping band structure calculation')

study_registry: dict[str, Study] = {}
def register_study(cls):
    """Registry enrollment so that Study subclasses can be instantiated by string name."""
    study_registry[cls.__name__] = cls
    return cls

@register_study
class Individual(Study):
    """Simplest study consisting of an individual calculation."""
    def init_state(self):
        self.calc_ids = ['calculation']
        super().init_state()

    def build_directory(self):
        self.dir = next_path(self.parent_dir / 'individual')
        self.dir.mkdir()
        self.state['calculation']['dir'] = self.dir
    
    def run_vasp(self):
        super().run_vasp('calculation')

@register_study
class EosFit(Study):
    """Calculate energy for scaled up/down supercells and fit an E(V) equation of state."""
    def init_state(self):
        # load restart file if it exists first to define skip_calc_ids
        try:
            self.skip_calc_ids = []
            with open(self.parent_dir / 'eosfit.restart', 'r') as f:
                for l in f.readlines():
                    self.skip_calc_ids.append(strip_split(l)[0])
            self.skip_calc_ids = [float(sf) for sf in self.skip_calc_ids if sf != 'eq']
            if len(self.skip_calc_ids):
                self.restart = True
                logger.debug(f'eosfit.restart found, restarting from scaling factor {self.skip_calc_ids[-1]}')
            else:
                self.restart = False
        except:
            self.restart = False

        # define state object
        self.calc_ids = self.params['scaling']+['eq']
        super().init_state()

        # go through and update scaling factor for each first step POSCAR
        for calc_id in self.calc_ids:
            step_1_poscar: VaspPoscar = self.state[calc_id][1]['POSCAR']
            if calc_id != 'eq':
                step_1_poscar.update_scaling_factor(calc_id)

    def build_directory(self):
        # determine path
        if self.restart:
            self.dir = Path(self.params['dir'])
        else:
            self.dir = next_path(self.parent_dir / 'eos')
            self.dir.mkdir()
            logger.debug(f'Generated study directory: {self.dir}')
        
        # create subdirectories
        for calc_id in self.calc_ids:
            subdir = self.dir / str(calc_id)
            subdir.mkdir(exist_ok=True)
            self.state[calc_id]['dir'] = subdir

    def run_vasp(self):
        volumes, energies = [], []
        for calc_id in self.calc_ids[:-1]:
            if calc_id == 'eq':
                break

            # run calculations
            if calc_id not in self.skip_calc_ids:
                super().run_vasp(calc_id)

                poscar = VaspPoscar(file_path = self.state[calc_id]['dir'] / 'POSCAR')
                outcar = VaspOutcar(file_path = self.state[calc_id]['dir'] / 'OUTCAR')
                volume, energy = poscar.volume, outcar.get_energy()
                volumes.append(volume)
                energies.append(energy)

                with open(self.dir / 'eosfit.restart', 'a') as restart:
                    restart.write(f'{calc_id}\t{volume}\t{energy}\n')

            # get volume and energy from restart file
            elif calc_id in self.skip_calc_ids:
                with open(self.dir / 'eosfit.restart', 'r') as restart:
                    for line in restart.readlines():
                        c_id, v, e = strip_split(line)
                        if calc_id == float(c_id):
                            volumes.append(float(v))
                            energies.append(float(e))

        # fit EoS
        eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
        eq_vol, eq_energy, bulk_mod = eos.fit()
        eos.plot(self.dir / 'eos.png')
        logger.debug(f"Birch-Murnaghan EOS fitted")
        
        # define eq calculation scaling factor
        eq_sf = eq_vol**(1/3)
        eq_poscar: VaspPoscar = self.state['eq'][1]['POSCAR']
        eq_poscar.update_scaling_factor(eq_sf)

        # perform eq calculation
        if 'eq' not in self.skip_calc_ids:
                super().run_vasp('eq')

                poscar = VaspPoscar(file_path = self.state['eq']['dir'] / 'POSCAR')
                outcar = VaspOutcar(file_path = self.state['eq']['dir'] / 'OUTCAR')
                eq_volume, eq_energy = poscar.volume, outcar.get_energy()

                with open(self.dir / 'eosfit.restart', 'a') as restart:
                    restart.write(f'eq\t{eq_volume}\t{eq_energy}\n')

        # get volume and energy from restart file
        elif 'eq' in self.skip_calc_ids:
            with open(self.dir / 'eosfit.restart', 'r') as restart:
                for line in restart.readlines():
                    c_id, v, e = strip_split(line)
                    if 'eq' == c_id:
                        eq_volume, eq_energy = float(v), float(e)

        # print out data
        with open(self.dir / 'data.out', 'w') as d:
            d.write(f'Volumes: {[float(v) for v in volumes]}\n')
            d.write(f'Energies: {[float(e) for e in energies]}\n')
            d.write(f'Equilibrium volume = {eq_volume} A3\n')
            d.write(f'Equilibrium energy = {eq_energy} eV\n')
            d.write(f"Equilibrium lattice parameters{poscar.lattice_parameters['a']} A\n")
            d.write(f"\ta = {poscar.lattice_parameters['a']} A\n")
            d.write(f"\tc = {poscar.lattice_parameters['c']} A\n")
            d.write(f'Bulk modulus: {bulk_mod / kJ * 1.0e24}')
        logger.debug(f"Printed fit data to {self.dir / 'data.out'}")

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
    def init_input_files(self):
        self.perfect_incar = VaspIncar(contents_str=self.calc_params['perfect']['INCAR'])
        self.perfect_kpoints = VaspKPoints(contents_str=self.calc_params['perfect']['KPOINTS'])
        try:
            self.perfect_path = Path(self.params['perfect'])
            self.perfect_poscar = VaspPoscar(file_path = Path(self.params['perfect']) / 'CONTCAR')
        except:
            self.perfect_path = None
            self.perfect_poscar = VaspPoscar(contents_str=self.calc_params['POSCAR'])
            logger.debug(f"Path to relaxed perfect system could not be found or was not provided")

        self.defective_incar = VaspIncar(contents_str=self.calc_params['defective']['INCAR'])
        self.defective_kpoints = VaspKPoints(contents_str=self.calc_params['defective']['KPOINTS'])
        self.defective_poscar = VaspPoscar(contents_str=self.calc_params['POSCAR'])
        
        # only need one version to manage both systems
        self.potcar = VaspPotcar(self.potcar_names, self.perfect_poscar)
        logger.debug('Initialized VASP input file objects from user input')

    def write_input_files(self, dir, system: str = None, overwrite_path=False):
        if system == 'perfect':
            self.perfect_incar.write_to_file(dir / 'INCAR', overwrite_path=overwrite_path)
            self.perfect_kpoints.write_to_file(dir / 'KPOINTS', overwrite_path=overwrite_path)
            self.perfect_poscar.write_to_file(dir / 'POSCAR', overwrite_path=overwrite_path)
        elif system == 'defective':
            self.defective_incar.write_to_file(dir / 'INCAR', overwrite_path=overwrite_path)
            self.defective_kpoints.write_to_file(dir / 'KPOINTS', overwrite_path=overwrite_path)
            self.defective_poscar.write_to_file(dir / 'POSCAR', overwrite_path=overwrite_path)

        self.potcar.write_to_file(dir / 'POTCAR', overwrite_path=overwrite_path)

    def build_directory(self):
        """Subdirectories for the perfect and defective system."""
        self.dir = next_path(self.parent_dir / (self.params['defect'] + '_formation'))
        # perfect system
        self.perfect_subdir = self.dir / 'perfect'
        self.perfect_subdir.mkdir(parents=True)
        self.write_input_files(self.perfect_subdir, system='perfect', overwrite_path=True)
        # copy in OUTCAR if perfect system has already been relaxed
        if self.perfect_path:          
            perfect_outcar = VaspOutcar(file_path = self.perfect_path / 'OUTCAR')
            perfect_outcar.write_to_file(self.perfect_subdir / 'OUTCAR')
            logger.debug(f"Perfect system OUTCAR copied from {self.perfect_path / 'OUTCAR'}")
        # insert the defect
        defect_pos = np.array(self.params['position'])
        if self.params['defect'] == 'vac':
            self.defective_poscar.remove_ion(defect_pos, self.defective_incar)
            logger.debug(f"Inserted vacancy near {str(defect_pos)}")
        elif self.params['defect'] == 'sub':
            rm_ion_pos = self.defective_poscar.remove_ion(defect_pos, self.defective_incar)
            try:
                magmom = self.params['magmom']
            except:
                magmom = None
            self.defective_poscar.add_ion(self.params['species'], rm_ion_pos, self.defective_incar, magmom=magmom)
            logger.debug(f"Inserted substitutional impurity near {str(defect_pos)}")
        # adjust INCAR
        self.defective_incar.update_tags([{'Add': f'ISYM = 0'}]) # disable symmetry
        # update POTCAR in case species were added
        self.potcar = VaspPotcar(self.potcar_names, self.defective_poscar)
        # defective system
        self.defective_subdir = self.dir / 'defective'
        self.defective_subdir.mkdir()
        self.write_input_files(self.defective_subdir, system='defective', overwrite_path=True)
    
    def run_vasp(self):
        # relax perfect system (if necessary)
        if self.perfect_path is None:
            logger.debug(f"Relaxing perfect system...")
            super().run_vasp(self.steps_params['perfect'], self.perfect_subdir)
        else:
            logger.debug(f"Perfect system already relaxed. Skipping it.")
        # relax defective system
        logger.debug(f"Relaxing defective system...")
        super().run_vasp(self.steps_params['defective'], self.defective_subdir)
        # extract energies
        perfect_outcar = VaspOutcar(file_path = self.perfect_subdir / 'OUTCAR')
        perfect_energy = perfect_outcar.get_energy()
        logger.debug(f"Calculated perfect system energy: {perfect_energy} eV")
        defective_outcar = VaspOutcar(file_path = self.defective_subdir / 'OUTCAR')
        defective_energy = defective_outcar.get_energy()
        logger.debug(f"Calculated defective system energy: {defective_energy} eV")
        # define chemical potential if it is not provided using E/N (bulk metals)
        try:
            chemical_pot = self.params['chemical_pot']
        except:
            num_species = len(self.perfect_poscar.get_species()[-1])
            chemical_pot = perfect_energy / num_species
        logger.debug(f"Defined chemical potential: {chemical_pot} eV")
        # calculate formation energy
        if self.params['defect'] == 'vac':
            formation_energy = defective_energy - perfect_energy + chemical_pot
        else:
            formation_energy = defective_energy - perfect_energy - chemical_pot
        logger.debug(f"Calculated defect formation energy: {formation_energy} eV")
        # write data to a file
        with open(self.dir / 'data.out', 'w') as d:
            d.write(f'Perfect system energy: {perfect_energy} eV\n')
            d.write(f'Defective system energy: {defective_energy} eV\n')
            d.write(f'Chemical potential: {chemical_pot} eV\n')
            d.write(f'Defect formation energy: {formation_energy} eV\n')