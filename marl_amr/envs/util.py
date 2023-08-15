from marl_amr.envs.solvers.BasePDESolver import BasePDESolver
from marl_amr.envs.solvers.AdvectionSolver import AdvectionSolver


def GetSolver(name, **kwargs):
    classnames = []

    for sclass in BasePDESolver.__subclasses__():
        classnames.append(sclass.name)
        if sclass.name == name:
            return sclass(**kwargs)

    raise ValueError(f'Could not find solver named {name} in list of solvers: {classnames}')

def MakeCheckpointFilename(dir,num):
    return f"{dir}/checkpoint_{num:06d}/checkpoint-{num}"
