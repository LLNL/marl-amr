import numpy as np
import numba

@numba.njit
def wrapPeriodic(X, S):
    tol = 1e-6

    for i, (x,s) in enumerate(zip(X,S)):
        if not -tol <= x <= s + tol:
            X[i] -= np.floor(x/s)*s

    return X


class BaseCoefficient():
    @classmethod
    def GetParams(cls, params, force_case=None):
        if params['randomize'] == 'discrete':
            # Set parameter key from a random discrete case index for each parameter
            for (pname, pdisc) in zip(cls.param_reqs, cls.param_discrete):
                # Choose random index from number of possible discrete values
                idx = (np.random.randint(len(params[pdisc])) if force_case is None
                       else force_case)
                # Set parameter based on the random index
                params[pname] = params[pdisc][idx]

        elif params['randomize'] == 'uniform':
            for (pname, prange) in zip(cls.param_reqs, cls.param_range):
                # Get low/high end of parameter range
                [r_low, r_high] = params[prange]

                # Set parameter randomly within range
                params[pname] = np.random.uniform(low=r_low, high=r_high)
        # Else if fixed parameters, do nothing

        # Add any necessary auxiliary data to param dict
        params = cls.AddAuxiliaryData(params)

        return params

    def AddAuxiliaryData(params):
        return params



'''
"initial_condition": {
     "coefficient": "ConstantCoefficient",
     "params": {
         "randomize": false,
         "u0": 1.5,
         "u0_range": null,
         "u0_discrete": null,
         "vx": 1.0,
         "vx_range": null,
         "vx_discrete": null,
         "vy": 2.0,
         "vy_range": null,
         "vy_discrete": null,
     }
}
'''
class ConstantCoefficient(BaseCoefficient):
    param_reqs = ['u0', 'vx', 'vy']
    param_discrete = [param + '_discrete' for param in param_reqs]
    param_range = [param + '_range' for param in param_reqs]

    def Eval(x, t):
        return u0


'''
"initial_condition": {
     "coefficient": "GaussianCoefficient",
     "params": {
         "randomize": false,
         "x0": 0.5,
         "x0_range": null,
         "x0_discrete": null,
         "w": 100,
         "w_range": null,
         "w_discrete": null,
         "vx": 1.0,
         "vx_range": null,
         "vx_discrete": null,
         "vy": 0.0,
         "vy_range": null,
         "vy_discrete": null,
     }
}
'''
class GaussianCoefficient(BaseCoefficient):
    param_reqs = ['x0', 'w', 'vx', 'vy']
    param_discrete = [param + '_discrete' for param in param_reqs]
    param_range = [param + '_range' for param in param_reqs]
 
    def Eval(x, t):
        # assert len(x) == 2

        # Time-dependent coefficient (for error calculations)
        ds = vx*t
        xx = x[0] - ds

        # Wrap in the case of periodic meshes
        if periodic:
            [xx] = wrapPeriodic([xx], [sx])

        dr2 = (xx - x0)**2

        return 1 + np.exp(-w*dr2)


'''
"initial_condition": {
     "coefficient": "Gaussian2DCoefficient",
     "params": {
         "randomize": false,
         "x0": 0.5,
         "x0_range": null,
         "x0_discrete": null,
         "y0": 0.5,
         "y0_range": null,
         "y0_discrete": null,
         "w": 100,
         "w_range": null,
         "w_discrete": null,
         "theta": 0.0, // direction, [0, 1] -> [0, 2pi]
         "theta_range": null,
         "theta_discrete": null,
         "u0": 1.0,   // magnitude
         "u0_range": null,
         "u0_discrete": null,
     }
}
'''
class Gaussian2DCoefficient(BaseCoefficient):
    param_reqs = ['x0', 'y0', 'w', 'theta', 'u0']
    param_discrete = [param + '_discrete' for param in param_reqs]
    param_range = [param + '_range' for param in param_reqs]

    def Eval(x, t):
        # Time-dependent coefficient (for error calculations) assumes periodicity
        vx = u0*np.cos(theta*2.0*np.pi)
        vy = u0*np.sin(theta*2.0*np.pi)
        dsx = vx*t
        dsy = vy*t

        xx = x[0] - dsx
        yy = x[1] - dsy

        # Wrap in the case of periodic meshes
        if periodic:
            [xx, yy] = wrapPeriodic([xx, yy], [sx, sy])

        dr2 = (xx - x0)**2 + (yy - y0)**2

        return 1 + np.exp(-w*dr2)

    def AddAuxiliaryData(params):
        # Get Cartesian velocity from polar velocity
        params['vx'] = params['u0']*np.cos(params['theta']*2.0*np.pi)
        params['vy'] = params['u0']*np.sin(params['theta']*2.0*np.pi)

        return params


class TwoGaussian2DCoefficient(BaseCoefficient):
    """Two Gaussians with the same velocity."""
    param_reqs = ['x0', 'y0', 'x1', 'y1', 'w', 'theta', 'u0']
    param_discrete = [param + '_discrete' for param in param_reqs]
    param_range = [param + '_range' for param in param_reqs]

    def Eval(x, t):
        # Time-dependent coefficient (for error calculations) assumes periodicity
        vx = u0*np.cos(theta*2.0*np.pi)
        vy = u0*np.sin(theta*2.0*np.pi)
        dsx = vx*t
        dsy = vy*t

        z = 0
        for x_0, y_0 in [(x0, y0), (x1, y1)]:
            xx = x[0] - dsx
            yy = x[1] - dsy
            # Wrap in the case of periodic meshes
            if periodic:
                [xx, yy] = wrapPeriodic([xx, yy], [sx, sy])
            dr2 = (xx - x_0)**2 + (yy - y_0)**2
            z += 1 + np.exp(-w*dr2)

        return z

    def AddAuxiliaryData(params):
        # Get Cartesian velocity from polar velocity
        params['vx'] = params['u0']*np.cos(params['theta']*2.0*np.pi)
        params['vy'] = params['u0']*np.sin(params['theta']*2.0*np.pi)

        return params


class MultipleGaussian2DCoefficient(BaseCoefficient):
    """Multiple Gaussians with the same velocity.

    x0, y0: coordinates of first bump
    nx, ny: number of bumps in x and y directions
    dx, dy: spacing between bumps
    """
    param_reqs = ['x0', 'y0', 'nx', 'ny', 'dx', 'dy', 'w', 'theta', 'u0']
    param_discrete = [param + '_discrete' for param in param_reqs]
    param_range = [param + '_range' for param in param_reqs]

    def Eval(x, t):
        # Time-dependent coefficient (for error calculations) assumes periodicity
        vx = u0*np.cos(theta*2.0*np.pi)
        vy = u0*np.sin(theta*2.0*np.pi)
        dsx = vx*t
        dsy = vy*t

        list_x = [x0 + dx*i for i in range(0, nx)]
        list_y = [y0 + dy*i for i in range(0, ny)]

        z = 0
        for x_0 in list_x:
            for y_0 in list_y:
                xx = x[0] - dsx
                yy = x[1] - dsy
                # Wrap in the case of periodic meshes
                if periodic:
                    [xx, yy] = wrapPeriodic([xx, yy], [sx, sy])
                dr2 = (xx - x_0)**2 + (yy - y_0)**2
                z += 1 + np.exp(-w*dr2)

        return z

    def AddAuxiliaryData(params):
        # Get Cartesian velocity from polar velocity
        params['vx'] = params['u0']*np.cos(params['theta']*2.0*np.pi)
        params['vy'] = params['u0']*np.sin(params['theta']*2.0*np.pi)

        return params


class OppositeGaussian2DCoefficient(BaseCoefficient):
    """Hard coded Gaussians that move with opposite velocities.

    Bump with index 0 travels left, bump with index 1 travels right.
    """
    param_reqs = ['x0', 'y0', 'x1', 'y1', 'w', 'u0']
    param_discrete = [param + '_discrete' for param in param_reqs]
    param_range = [param + '_range' for param in param_reqs]

    def Eval(x, t):
        # Time-dependent coefficient (for error calculations) assumes periodicity
        theta0 = 0.5 # moves left
        vx0 = u0*np.cos(theta0*2.0*np.pi)
        vy0 = u0*np.sin(theta0*2.0*np.pi)
        dsx0 = vx0*t
        dsy0 = vy0*t

        theta1 = 0.0 # moves right
        vx1 = u0*np.cos(theta1*2.0*np.pi)
        vy1 = u0*np.sin(theta1*2.0*np.pi)
        dsx1 = vx1*t
        dsy1 = vy1*t

        z = 0
        for (x_0, y_0, dsx, dsy) in [(x0, y0, dsx0,dsy0), (x1, y1, dsx1, dsy1)]:
            xx = x[0] - dsx
            yy = x[1] - dsy
            # Wrap in the case of periodic meshes
            if periodic:
                [xx, yy] = wrapPeriodic([xx, yy], [sx, sy])
            dr2 = (xx - x_0)**2 + (yy - y_0)**2
            z += 1 + np.exp(-w*dr2)

        return z

    def AddAuxiliaryData(params):
        params['vx'] = params['u0']*1.0
        params['vy'] = params['u0']*0.0

        return params


class Ring2DCoefficient(BaseCoefficient):
    param_reqs = ['x0', 'y0', 'w', 'r', 'theta', 'u0']
    param_discrete = [param + '_discrete' for param in param_reqs]
    param_range = [param + '_range' for param in param_reqs]

    def Eval(x, t):
        # Time-dependent coefficient (for error calculations) assumes periodicity
        vx = u0*np.cos(theta*2.0*np.pi)
        vy = u0*np.sin(theta*2.0*np.pi)
        dsx = vx*t
        dsy = vy*t

        xx = x[0] - dsx
        yy = x[1] - dsy

        # Wrap in the case of periodic meshes
        if periodic:
            [xx, yy] = wrapPeriodic([xx, yy], [sx, sy])

        dr2 = (xx - x0)**2 + (yy - y0)**2

        return 1 + np.exp(-w*(np.sqrt(dr2) - r)**2)

    def AddAuxiliaryData(params):
        # Get Cartesian velocity from polar velocity
        params['vx'] = params['u0']*np.cos(params['theta']*2.0*np.pi)
        params['vy'] = params['u0']*np.sin(params['theta']*2.0*np.pi)

        return params


class Gaussian3DCoefficient(BaseCoefficient):
    param_reqs = ['x0', 'y0', 'z0', 'w', 'theta', 'phi', 'u0']
    param_discrete = [param + '_discrete' for param in param_reqs]
    param_range = [param + '_range' for param in param_reqs]

    def Eval(x, t):
        # Time-dependent coefficient (for error calculations) assumes periodicity
        vx = u0*np.sin(phi*np.pi)*np.cos(theta*2.0*np.pi)
        vy = u0*np.sin(phi*np.pi)*np.sin(theta*2.0*np.pi)
        vz = u0*np.cos(phi*np.pi)
        dsx = vx*t
        dsy = vy*t
        dsz = vz*t

        xx = x[0] - dsx
        yy = x[1] - dsy
        zz = x[2] - dsz

        # Wrap in the case of periodic meshes
        if periodic:
            [xx, yy, zz] = wrapPeriodic([xx, yy, zz], [sx, sy, sz])

        dr2 = (xx - x0)**2 + (yy - y0)**2 + (zz - z0)**2

        return 1 + np.exp(-w*dr2)

    def AddAuxiliaryData(params):
        # Get Cartesian velocity from spherical velocity
        params['vx'] = params['u0']*np.sin(params['phi']*np.pi)*np.cos(params['theta']*2.0*np.pi)
        params['vy'] = params['u0']*np.sin(params['phi']*np.pi)*np.sin(params['theta']*2.0*np.pi)
        params['vz'] = params['u0']*np.cos(params['phi']*np.pi)

        return params


class AnisotropicGaussian2DCoefficient(BaseCoefficient):
    param_reqs = ['x0', 'y0', 'wx', 'wy', 'wxy', 'theta', 'u0']
    param_discrete = [param + '_discrete' for param in param_reqs]
    param_range = [param + '_range' for param in param_reqs]

    def Eval(x, t):
        # Time-dependent coefficient (for error calculations) assumes periodicity
        vx = u0*np.cos(theta*2.0*np.pi)
        vy = u0*np.sin(theta*2.0*np.pi)
        dsx = vx*t
        dsy = vy*t

        xx = x[0] - dsx
        yy = x[1] - dsy

        # Wrap in the case of periodic meshes
        if periodic:
            [xx, yy] = wrapPeriodic([xx, yy], [sx, sy])

        dr2 = wx*(xx - x0)**2 + wy*(yy - y0)**2 + wxy*(xx-x0)*(yy-y0)

        return 1 + np.exp(-dr2)

    def AddAuxiliaryData(params):
        # Get Cartesian velocity from polar velocity
        params['vx'] = params['u0']*np.cos(params['theta']*2.0*np.pi)
        params['vy'] = params['u0']*np.sin(params['theta']*2.0*np.pi)

        return params


class OrbitingGaussian2DCoefficient(BaseCoefficient):
    param_reqs = ['x0', 'xc', 'y0', 'yc',  'w', 'omega']
    param_discrete = [param + '_discrete' for param in param_reqs]
    param_range = [param + '_range' for param in param_reqs]

    def Eval(x, t):
        # Time-dependent coefficient (for error calculations) assumes periodicity
        vx = omega * (x[1] - yc)
        vy = -omega * (x[0] - xc)
        dsx = vx*t
        dsy = vy*t

        xx = x[0] - dsx
        yy = x[1] - dsy

        # Wrap in the case of periodic meshes
        if periodic:
            [xx, yy] = wrapPeriodic([xx, yy], [sx, sy])

        dr2 = (xx - x0)**2 + (yy - y0)**2

        return 1 + np.exp(-w*dr2)

    def AddAuxiliaryData(params):
        params['vx'] = params['omega']
        params['vy'] = params['omega']

        return params
