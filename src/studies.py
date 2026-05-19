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
from py4vasp import Calculation

logger = logging.getLogger('VaspUtils')
logging.getLogger("matplotlib").setLevel(logging.FATAL)
logging.getLogger("ase").setLevel(logging.FATAL)
logging.getLogger("py4vasp").setLevel(logging.FATAL)

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
        logger.debug(f'Initializing state...')
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
            if key in ['INCAR', 'KPOINTS', 'POSCAR']:
                common_files[key] = vasp_file_registry[key](contents_str=val)

        if 'POSCAR' not in self.calc_params.keys():
            raise KeyError('No POSCAR found in the `calculations` section`. All calculations must share a common POSCAR file')

        if 'POTCAR' in self.calc_params.keys():
            common_files['POTCAR'] = VaspPotcar(contents_str=self.calc_params['POTCAR'], poscar=common_files['POSCAR'])

        self.state['common'] = common_files
        logger.debug('Read in common VASP files from user input')

        # initialize steps for each calculation
        for calc_id in self.calc_ids:
            calc_dict: dict = self.state[calc_id]

            # steps_params is either the same for all calculations or is defined for each one separately
            if calc_id in self.steps_params.keys():
                steps_params: dict = self.steps_params[calc_id]
            else:
                steps_params: dict = deepcopy(self.steps_params)

            # define step dict for first step
            step_dict: dict = deepcopy(steps_params[1])

            # option to skip calculation
            #if calc_id in self.skip_calc_ids:
            #    continue

            # add files shared by all calculations to first step (should be just a name at this point)
            for fn, file in self.state['common'].items():
                if fn in ['INCAR', 'POSCAR', 'KPOINTS', 'POTCAR']:
                    step_dict[fn] = deepcopy(file)
            
            # add files specific to the calculation
            if calc_id in self.calc_params.keys():
                for fn, contents in self.calc_params[calc_id]:
                    # delay defining POTCAR until POSCAR is available (i.e., after the previous calculation is run)
                    if fn != 'POTCAR':
                        step_dict[fn] = vasp_file_registry[fn](contents_str=contents)

            # save step
            calc_dict[1] = step_dict
            logger.debug(f'Defined step 1 for {calc_id}')

            # read next integer-labelled steps (2, 3, ...)
            num_steps = len([k for k in steps_params.keys() if type(k) == int])
            for step_id in range(2, num_steps+1):
                step_dict: dict = deepcopy(steps_params[step_id])

                # replace file modifications with a VaspFile object
                for key, val in step_dict.items():
                    # only want VaspFiles
                    if key == 'name':
                        continue

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
            if 'dos' in steps_params.keys():
                dos_dict: dict = steps_params['dos']
                for fn, contents in dos_dict.items():
                    dos_dict[fn] = vasp_file_registry[fn](contents_str=contents)
                calc_dict['dos'] = dos_dict

            if 'bands' in steps_params.keys():
                bands_dict: dict = steps_params['bands']
                for fn, contents in bands_dict.items():
                    bands_dict[fn] = vasp_file_registry[fn](contents_str=contents)
                calc_dict['bands'] = bands_dict

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
        if 'dos' in calc_dict.keys():
            self.write_input_files(calc_dict['dos'], run_dir)
            logger.debug(f'DOS input files provided. Doing calculation...')
            with open(run_dir / 'vasp.out', 'a') as vasp_out:
                vasp = subprocess.run(vasp_cmd, cwd=run_dir, stdout=vasp_out, stderr=subprocess.STDOUT)
            dos = Calculation.from_path(calc_dict[dir])
            dos_plot = dos.dos.plot()
            plt.savefig("dos.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.debug(f'DOS calculation done')
        else:
            logger.debug(f'Skipping DOS calculation')

        if 'bands' in calc_dict.keys():
            self.write_input_files(calc_dict['bands'], run_dir)
            logger.debug(f'Band structure input files provided. Doing calculation...')
            with open(run_dir / 'vasp.out', 'a') as vasp_out:
                vasp = subprocess.run(vasp_cmd, cwd=run_dir, stdout=vasp_out, stderr=subprocess.STDOUT)
            band = Calculation.from_path(calc_dict[dir])
            band_plot = band.band.plot()
            plt.savefig("bands.png", dpi=300, bbox_inches="tight")
            plt.close()
            logger.debug(f'Band structure calculation done')
        else:
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
class PointDefectFormation(Study):
    """Calculate formation for a vacancy or substitutional impurity."""
    def init_state(self):
        self.calc_ids = ['perfect', 'defective']
        super().init_state()
    
    def build_directory(self):
        self.dir = next_path(self.parent_dir / (self.params['defect'] + '_formation'))
        self.dir.mkdir()

        perfect_subdir = self.dir / 'perfect'
        perfect_subdir.mkdir()
        self.state['perfect']['dir'] = perfect_subdir

        defective_subdir = self.dir / 'defective'
        defective_subdir.mkdir()
        self.state['defective']['dir'] = defective_subdir

    def run_vasp(self):
        # obtain perfect system CONTCAR and OUTCAR first
        try:
            perfect_contcar = VaspContcar(file_path = Path(self.params['perfect']) / 'CONTCAR')
            perfect_output_dir = Path(self.params['perfect'])
            logger.debug(f"Perfect system already relaxed. Skipping it.")
        except:
            logger.debug(f"Relaxing perfect system...")
            super().run_vasp('perfect')
            perfect_output_dir = self.state['perfect']['dir']
        
        perfect_contcar = VaspContcar(file_path = perfect_output_dir / 'CONTCAR')
        perfect_outcar = VaspOutcar(file_path = perfect_output_dir / 'CONTCAR')

        # update defective system POSCAR and grab INCAR
        self.state['defective'][1]['POSCAR'] = perfect_contcar
        defective_poscar: VaspPoscar = self.state['defective'][1]['POSCAR']
        defective_incar: VaspIncar = self.state['defective'][1]['INCAR']

        # insert defect
        defect_pos = np.array(self.params['position'])

        if self.params['defect'] == 'vac':
            defective_poscar.remove_ion(defect_pos, defective_incar)
            logger.debug(f"Inserted vacancy near {str(defect_pos)}")

        elif self.params['defect'] == 'sub':
            rm_ion_pos = defective_poscar.remove_ion(defect_pos, defective_incar)
            if 'magmom' in self.params.keys():
                magmom = self.params['magmom']
            else:
                magmom = None
            defective_poscar.add_ion(self.params['species'], rm_ion_pos, defective_incar, magmom=magmom)
            logger.debug(f"Inserted substitutional impurity near {str(defect_pos)}")
        
        # propogate MAGMOM updates through all steps if it was modified internally
        if defective_incar.check_by_tag('MAGMOM'):
            _, magmom_line = defective_incar.check_by_tag('MAGMOM')
            num_steps = len([k for k in self.state['defective'].keys() if type(k) == int])
            for step_id in range(2, num_steps+1):
                step_incar: VaspIncar = self.state['defective'][step_id]['INCAR']
                if step_incar.check_by_tag('MAGMOM'):
                    self.update_input_file(step_incar, [{'Add': magmom_line}])
        
        # update POTCAR
        if 'POTCAR' in self.calc_params['defective'].keys():
            contents_str = self.calc_params['defective']['POTCAR']
        else:
            contents_str = self.calc_params['POTCAR']
        self.state['defective'][1]['POTCAR'] = VaspPotcar(contents_str=contents_str, poscar=defective_poscar)

        # relax defective system
        logger.debug(f"Relaxing defective system...")
        super().run_vasp('defective')

        # extract energies
        perfect_energy = perfect_outcar.get_energy()
        logger.debug(f"Calculated perfect system energy: {perfect_energy} eV")

        defective_outcar = VaspOutcar(file_path = self.state['defective']['dir'] / 'OUTCAR')
        defective_energy = defective_outcar.get_energy()
        logger.debug(f"Calculated defective system energy: {defective_energy} eV")

        # define chemical potential if it is not provided using E/N (bulk metals)
        if 'chemical_pot' in self.params.keys():
            chemical_pot = self.params['chemical_pot']
        else:
            num_species = len(perfect_contcar.get_species()[-1])
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