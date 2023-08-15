import numpy as np
import mfem.ser as mfem

def LoadEstimatorByName(name, *args, **kwargs):
    classnames = []

    for sclass in BaseEstimator.__subclasses__():
        classnames.append(sclass.name)
        if sclass.name == name.lower():
            return sclass(*args, **kwargs)

    raise ValueError(f'Could not find estimator named {name} in list of estimators: {classnames}')

class BaseEstimator():
    def __init__(self, solver):
        self.solver = solver

    # Output format is of np.array type with size == number of elements.
    # Higher values indicate elements which need to be refined.
    def ComputeScores(self):
        pass

class RandomEstimator(BaseEstimator):
    name = 'random'
    def __init__(self, solver):
        super().__init__(solver)

    def ComputeScores(self):
        NE = self.solver.mesh.GetNE()
        return np.random.rand(NE)

class TrueErrorEstimator(BaseEstimator):
    name = 'true_error'
    def __init__(self, solver):
        super().__init__(solver)

    def ComputeScores(self):
        return self.solver.GetElementErrors()

class ZZEstimator(BaseEstimator):
    name = 'zz'
    def __init__(self, solver):
        super().__init__(solver)
        self.coeff = mfem.ConstantCoefficient(1.0)
        self.diffusion = mfem.DiffusionIntegrator(self.coeff)

    def ComputeScores(self):
        self.estimator =  mfem.LSZienkiewiczZhuEstimator(self.diffusion, self.solver.solution)
        self.estimator.Reset()
        
        return self.estimator.GetLocalErrors().GetDataArray()
