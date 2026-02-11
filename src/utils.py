import os

def strip_split(s: str, sep=None, item_type=None):
    s = s.strip()
    s = s.split(sep)
    if item_type is int:
        return [int(x) for x in s]
    elif item_type is float:
        return [float(x) for x in s]
    elif item_type is None:
        return s
    else:
        raise ValueError(f'[{item_type}] Invalid item type. Choose None, int, or float')

def tilps(list_str: list[str], sep: str = ' '):
    s = ''
    for l in list_str:
        s += str(l) + sep
    return s

def check_slurm_var(args_attr, var_name: str):
    num_requested_resources = int(os.environ.get(var_name, 1))
    assert args_attr > 0, f'[{var_name}] Number of requested resources must be greater than 0.'
    assert args_attr <= num_requested_resources, f'[{var_name}] Too many requested resoruces. ({num_requested_resources} available).'