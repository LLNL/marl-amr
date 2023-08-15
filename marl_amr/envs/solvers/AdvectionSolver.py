import copy
import numpy as np
import mfem.ser as mfem

from marl_amr.envs.solvers.BasePDESolver import BasePDESolver
from marl_amr.envs.solvers.coefficients import *

GLOBAL_dim = 1

def VelocityCoefficient(x, out, sdim_, dim_):
    out[0] = vx
    out[1] = vy

def VelocityCoefficient3D(x, out, sdim_, dim_):
    out[0] = vx
    out[1] = vy
    out[2] = vz

def OppositeVelocityCoefficient(x, out, sdim_, dim_):
    """Velocities for two opposite-moving features.

    Assumes 'center' y-coordinate is 1.0.
    Feature with y > 1.0 has velocity [vx, vy]
    Feature with y < 1.0 has velocity [-vx, vy]
    """
    out[0] = np.tanh(100*(x[1]-1.0)) * vx
    out[1] = vy

def OrbitingVelocityCoefficient(x, out, sdim_, dim_):
    """Velocities for two opposite-moving features.

    vx = vy = omega (angular velocity)
    """
    out[0] = vx * (x[1] - yc)
    out[1] = -vy * (x[0] - xc)

def InflowCoefficient(x, t):
    return 1.0

class FE_Evolution(mfem.PyTimeDependentOperator):
    def __init__(self, M, K, B):
        mfem.PyTimeDependentOperator.__init__(self, M.Size())

        self.K = K
        self.M = M
        self.B = B
        self.z = mfem.Vector(M.Size())
        self.zp = np.zeros(M.Size())
        self.M_prec = mfem.DSmoother()
        self.M_solver = mfem.CGSolver()
        self.M_solver.SetPreconditioner(self.M_prec)
        self.M_solver.SetOperator(M)
        self.M_solver.iterative_mode = False
        self.M_solver.SetRelTol(1e-9)
        self.M_solver.SetAbsTol(0.0)
        self.M_solver.SetMaxIter(100)
        self.M_solver.SetPrintLevel(0)

    def Mult(self, x, y):
        self.K.Mult(x, self.z)
        self.z += self.B
        self.M_solver.Mult(self.z, y)

class AdvectionSolver(BasePDESolver):
    name = 'advection'

    def __init__(self, **kwargs):
        global GLOBAL_dim
        super().__init__()
        self.nvars = 1

        self.single_step = kwargs.get('single_step', False)

        self.nx = kwargs['nx']
        self.ny = kwargs['ny']
        self.nz = kwargs.get('nz', 1)
        GLOBAL_dim = 1 if self.ny == 1 else (2 if self.nz == 1 else 3)


        self.aniso = kwargs.get('aniso', None)
        if self.aniso == None:
            self.aniso = True if GLOBAL_dim == 1 else False

        self.sx = kwargs['sx']
        self.sy = kwargs['sy']
        self.sz = kwargs.get('sz', 1.0)
        self.order = kwargs['order']
        self.t_step = kwargs['t_step']
        self.periodic = kwargs['periodic']
        self.opposite_velocities = kwargs.get('opposite_velocities', False)
        self.orbiting_velocities = kwargs.get('orbiting_velocities', False)
        self.mesh_file = kwargs.get('mesh_file', '')

        self.dt = kwargs.get('dt', None)
        self.CFL = kwargs.get('CFL', None)

        if not self.dt and not self.CFL:
            raise ValueError('Must specify either dt or CFL.')
        elif self.dt and self.CFL:
            raise ValueError('Cannot specify both dt and CFL.')
        elif self.CFL:
            # WARNING: Time step from CFL is calculated based on the original mesh and polynomial order using RK4 stepping.
            #          Adjust CFL constant as needed if large variations in order are expected.
            h = min(self.sx/self.nx, self.sy/self.ny)
            c = 1. # Assume a propogation speed of 1
            self.dt = self.CFL*(h/c)/(2*self.order + 1) # 2p+1 scaling to dt

        # refinement mode: p (order) or h(upto element depth = 1 for now) 
        self.ref_mode = kwargs['refinement_mode'] # p or h
        print("Solver refinement mode = ", self.ref_mode)

        self.mfem_element_shape = self.map_element_shape_to_mfem[
            kwargs['element_shape']]

        ic_config = kwargs['initial_condition']
        self.coeff_name = ic_config['coefficient']
        self.ic_coefficient = eval(self.coeff_name)

        self.ic_params = ic_config['params']
        self.ic_params['periodic'] = self.periodic
        self.ic_params['sx'] = self.sx
        self.ic_params['sy'] = self.sy
        self.ic_params['sz'] = self.sz
        self.ic_params['GLOBAL_dim'] = GLOBAL_dim
        self.refine_IC = kwargs['refine_IC']
        self.error_threshold = 0.0

        assert self.ic_params['randomize'] in [False, 'discrete', 'uniform'], f"{self.ic_params['randomize']} \
                                                                                   not in [False, 'discrete', 'uniform']"
        if self.ic_params['randomize'] == 'discrete':
            self.param_reqs = [param + '_discrete' for param in self.ic_coefficient().param_reqs]

            # Verify discrete parameter are of list type with same length
            assert(all(isinstance(self.ic_params[param], list) for param in self.param_reqs)), \
                    f'Discrete IC params must be of list type.'

        self.t = 0.0
        self.t_history = 0
        self.elem_action_prev = []

    def SetupMesh(self):
        global GLOBAL_dim

        # Create mesh
        if self.mesh_file == '':
            if self.nz == 1:
                self.mesh = mfem.Mesh_MakeCartesian2D(self.nx, self.ny,
                                                      self.mfem_element_shape,
                                                      sx=self.sx, sy=self.sy)
            else:
                self.mesh = mfem.Mesh_MakeCartesian3D(self.nx, self.ny, self.nz,
                                                      self.mfem_element_shape,
                                                      sx=self.sx, sy=self.sy, sz=self.sz)
        else:
            # args: str filename, int generate_edges, int refine
            self.mesh = mfem.Mesh(self.mesh_file, 1, 1)

        # Setup periodicity along x direction
        if self.periodic:
            if GLOBAL_dim == 1:
                translations = (mfem.Vector([self.sx, 0.]),)
            elif GLOBAL_dim == 2:
                assert self.ny >= 3, '2D periodic meshes require at least 3 elements in y-direction.'
                translations = (mfem.Vector([self.sx, 0.]), mfem.Vector([0., self.sy]))
            elif GLOBAL_dim == 3:
                assert self.ny >= 3 and self.nz >= 3, '3D periodic meshes require at least 3 elements in y and z-direction.'
                translations = (mfem.Vector([self.sx, 0., 0.]), mfem.Vector([0., self.sy, 0.]),
                                mfem.Vector([0., 0., self.sz]))

            mapping = self.mesh.CreatePeriodicVertexMapping(translations)
            self.mesh = mfem.Mesh_MakePeriodic(self.mesh, mapping)

        self.mesh.EnsureNCMesh(self.mfem_element_shape == mfem.Element.TRIANGLE)

        self.initial_mesh = mfem.Mesh(self.mesh)

        # Set up boundary data
        self.ess_tdof_list = mfem.intArray()
        self.ess_bdr = mfem.intArray([1]*self.mesh.bdr_attributes.Size())
        self.dim = self.initial_mesh.Dimension()

    def SetupCoefficients(self, force_case=None):
        """Sets coefficients and velocity.

        Args:
            force_case: None, or int index of IC params case to use.
        """
        # Create initial conditions
        self.ic_params = self.ic_coefficient.GetParams(self.ic_params, force_case)

        self.true_solution = mfem.jit.scalar(sdim=2, td=True, params=self.ic_params)(self.ic_coefficient.Eval)

        if self.opposite_velocities:
            velocity_function = OppositeVelocityCoefficient
        elif self.orbiting_velocities:
            velocity_function = OrbitingVelocityCoefficient
        else:
            velocity_function = VelocityCoefficient

        if self.nz == 1:
            self.velocity = mfem.jit.vector(sdim=2, vdim=2, params=self.ic_params)(velocity_function)
            self.vel = [self.ic_params['vx'], self.ic_params['vy']]
        else:
            # Not sure what sdim and vdim should be but this works
            self.velocity = mfem.jit.vector(sdim=2, vdim=3, params=self.ic_params)(velocity_function)
            self.vel = [self.ic_params['vx'], self.ic_params['vy'], self.ic_params['vz']]

        self.true_solution.SetTime(0.0)
        self.inflow = mfem.jit.scalar(sdim=2, td=True)(InflowCoefficient)

    def SetupFEM(self):
        self.fec = mfem.DG_FECollection(self.order, self.mesh.Dimension(), mfem.BasisType.GaussLobatto)
        self.fespace = mfem.FiniteElementSpace(self.mesh, self.fec)

        # Create FEM matrices
        self.M = mfem.BilinearForm(self.fespace)
        self.M.AddDomainIntegrator(mfem.MassIntegrator())
        self.K = mfem.BilinearForm(self.fespace)
        self.K.AddDomainIntegrator(
                 mfem.ConvectionIntegrator(self.velocity, -1.0))
        self.K.AddInteriorFaceIntegrator(
                 mfem.TransposeIntegrator(mfem.DGTraceIntegrator(self.velocity, 1.0, -0.5)))
        self.K.AddBdrFaceIntegrator(
                 mfem.TransposeIntegrator(mfem.DGTraceIntegrator(self.velocity, 1.0, -0.5)))
        self.B = mfem.LinearForm(self.fespace)
        self.B.AddBdrFaceIntegrator(
                 mfem.BoundaryFlowIntegrator(self.inflow, self.velocity, -1.0, -0.5))

        self.M.Assemble()
        self.M.Finalize()
        self.skip_zeros = 0
        self.K.Assemble(self.skip_zeros)
        self.K.Finalize(self.skip_zeros)
        self.B.Assemble()

    def SetupODESolver(self):
        # Create system and ODE solver
        self.ode_solver = mfem.RK4Solver()
        self.adv = FE_Evolution(self.M.SpMat(), self.K.SpMat(), self.B)
        self.ode_solver.Init(self.adv)

    def SetInitialCondition(self):

        # We will follow a standard AMR generation process where the
        # solution is projected onto a coarse grid, the error is
        # computed, and elements of high error are refined.

        # Start simple with a one-pass initialization. This can be
        # generalized to an n-pass loop.

        self.solution = mfem.GridFunction(self.fespace)
        self.solution.ProjectCoefficient(self.true_solution)

        self.elem_action_prev = np.zeros(self.mesh.GetNE(), dtype=int)
        elem_errors = self.GetElementErrors()

        if self.refine_IC:
            idxs = np.where(elem_errors > self.error_threshold)[0] 
            self.elem_action_prev[idxs] = 1 

            if (self.ref_mode == 'p'): 
                self.pRefine(self.elem_action_prev, "increment_base", initial_ref=True) 
            else: 
                self.hRefine(self.elem_action_prev, aniso=self.aniso, depth_limit=1)

            self.solution.ProjectCoefficient(self.true_solution)

    def Reset(self, force_case=None):
        """Resets the solver for a new episode.

        Args:
            force_case: None, or int index of IC params case to use in
                the case with discrete set of params
        """
        self.t = 0.0

        if self.solver_initialized: self.Delete()
        self.SetupMesh()
        self.SetupCoefficients(force_case)
        self.SetupFEM()
        self.SetupODESolver()

        self.solver_initialized = True
        # table to keep track of elements to agents map on each mesh
        # (only applies to the h-ref case) 
        self.initial_to_current_table = None

        self.SetInitialCondition()
        self.SetupEstimator()

        if self.single_step:
            self.Render()

    def Render(self):
        self.VisualizeMesh()
        self.VisualizeSolution()
        self.VisualizeError()

        gf = mfem.GridFunction(self.fespace)
        gf.ProjectCoefficient(self.true_solution)
        self.VisualizeGridFunction(gf, 'True Solution')

        if self.single_step:
            input('Render() called. press enter to continue...')


    def Delete(self):
        objs = [self.K, self.B, self.M,
                self.ode_solver, self.adv, self.solution,
                self.true_solution, self.inflow, self.velocity,
                self.ess_tdof_list, self.ess_bdr]

        for obj in objs:
            del(obj)

    @BasePDESolver.AssertSolver
    def Update(self):
        GetProlongation = False if self.ref_mode == 'p' else True
        self.fespace.Update(GetProlongation)
        self.M.Update()
        self.K.Update()
        self.B.Update()
        self.solution.Update()

        self.M.Assemble()
        self.K.Assemble()
        self.B.Assemble()

        self.M.Finalize()
        self.K.Finalize()

        # Should replace this with an update() function
        del(self.adv)
        self.adv = FE_Evolution(self.M.SpMat(), self.K.SpMat(), self.B)

        self.ode_solver.Init(self.adv)
        self.SetupEstimator()

    @BasePDESolver.AssertSolver
    def UpdateTrueSolution(self):
        self.true_solution.SetTime(self.t)

    @BasePDESolver.AssertSolver
    def hRefine(self, agent_actions, aniso=True, depth_limit=100):
        '''
        Performs all the feasible actions suggested from the agent_actions and
            returns old_to_new_element_map ()
            returns elements_to_be_created (new mesh element ids)
            returns elements_to_be_deleted (old mesh element ids)
            overwrite to no action if an element is marked for refinement but it exceeds the depth limit

        Current policy to map agent_actions to actions
         1. All elements that are marked for refinement are to perform the refinement
         2. All of the "siblings" (i) of a marked element for refinement are assigned action=max(0,agent_actions[i])
            i.e., a) if the action is to be refined then they are refined
                  b) if the action is to be derefined or no action then they get no action
         3. If among the "siblings" there is no refinement action then the group is marked
            for derefinement if the majority (including a tie) of the siblings are marked for derefinement
            otherwise they are marked for no action

        h-refine:   action =  1
        h-derefine: action = -1
        do nothing: action =  0

        '''

        # Set agent actions to 0 if trying to refine past max-depth
        agent_actions = np.array(agent_actions, dtype=int)
        for i in range(self.mesh.GetNE()):
            depth = self.mesh.ncmesh.GetElementDepth(i)
            if depth >= depth_limit and agent_actions[i] == 1:
                agent_actions[i] = 0

        #  List of all possible derefinement actions
        actions = np.zeros(self.mesh.GetNE(), dtype=int)
        actions_marker = np.zeros(self.mesh.GetNE(), dtype=int)
        deref_table = self.mesh.ncmesh.GetDerefinementTable()

        for i in range(deref_table.Size()):
            size = deref_table.RowSize(i)
            row = mfem.intArray(size)
            row.Assign(deref_table.GetRow(i))
            sum_of_actions = 0
            # flag if at least 1 element is marked for refinement
            refflag = False
            for j in range(size):
                sum_of_actions += agent_actions[row[j]]
                if (agent_actions[row[j]] == 1):
                    refflag = True
                    break

            if refflag:
               for j in range(size):
                    actions[row[j]] = max(0,int(agent_actions[row[j]]))
                    actions_marker[row[j]] = 1

            else:
                drefflag = False
                #  check if greater of equal than half are marked for derefinement 
                if 2*abs(sum_of_actions) >= size:
                    drefflag = True

                for j in range(size):
                    # actions[row[j]] = (-1 if drefflag else 0)
                    if (drefflag):
                        actions[row[j]] = -1
                    else:
                        actions[row[j]] = 0
                    actions_marker[row[j]] = 1

        #  Copy the action
        idxs = np.where(actions_marker != 1)[0]
        actions[idxs] = agent_actions[idxs]

        #  If they are marked for derefinement but they were not in the list of
        #  available derefinements then they are marked for no action
        idxs = np.where((actions_marker != 1) & (agent_actions == -1))[0]
        actions[idxs] = 0

        # Now the actions array holds feasible actions of -1,0,1
        elems_to_refine = np.where(actions == 1)[0]
        if aniso:
            ref_array = mfem.RefinementArray()
            for elem in elems_to_refine:
                # 0b01 = x only
                # 0b10 = y only
                # 0b11 = x/y
                ref_array.Append(mfem.Refinement(elem, 0b01))
        else:
            ref_array = mfem.intArray(list(elems_to_refine))
        
        if ref_array.Size() > 0:
            self.mesh.GeneralRefinement(ref_array)
            self.Update()

        #  Now take care of the derefinements
        new_actions = np.ones(self.mesh.GetNE(), dtype=int)
        #  Map old actions to new mesh
        ref_table = mfem.Table()
        dref_table = mfem.Table()
        if (ref_array.Size() > 0):
            tr = self.mesh.GetRefinementTransforms()
            tr.MakeCoarseToFineTable(ref_table)
            for i in range(ref_table.Size()):
                size = ref_table.RowSize(i)
                if size == 1:
                    row = mfem.intArray(size)
                    row.Assign(ref_table.GetRow(i))
                    new_actions[row[0]] = actions[i]
        else:
            new_actions = actions

        dummy_errors = np.ones(self.mesh.GetNE())
        idxs = np.where(new_actions < 0)[0]
        if len(idxs) > 0:
            dummy_errors[idxs] = 0.
            dummy_errors = mfem.Vector(list(dummy_errors))

            dummy_thresh = 0.5
            self.mesh.DerefineByError(dummy_errors, dummy_thresh)

            tr = self.mesh.ncmesh.GetDerefinementTransforms()
            tr.MakeCoarseToFineTable(dref_table)
            dref_table = mfem.Transpose(dref_table)
            self.Update()

        T = mfem.Table()
        if ref_table.Size() > 0 and dref_table.Size() > 0 :
            T = mfem.Mult(ref_table, dref_table)
        elif ref_table.Size() > 0:
            T = ref_table
        elif dref_table.Size() > 0:
            T = dref_table
        else:
            T = None

        if not self.initial_to_current_table:
            self.initial_to_current_table = T
        else:
            if T:
                self.initial_to_current_table = mfem.Mult(self.initial_to_current_table,T)

        # return old_to_new_element_map ()
        # return elements_to_be_created (new mesh element ids)
        # return elements_to_be_deleted (old mesh element ids)
        old_to_new_element_map = mfem.intArray()
        elements_to_be_created = mfem.intArray()
        elements_to_be_deleted = mfem.intArray()
        if T is not None:
            Tt = mfem.Transpose(T)
            old_to_new_element_map.SetSize(T.Size())
            elements_to_be_created = mfem.intArray()
            elements_to_be_deleted = mfem.intArray()
            for i in range(T.Size()):
                n = T.RowSize(i)
                row = mfem.intArray(n)
                row.Assign(T.GetRow(i))
                if n == 1:
                    m = Tt.RowSize(row[0])
                    if m == 1 :
                        old_to_new_element_map[i] = row[0]
                    else:
                        old_to_new_element_map[i] = -1
                        elements_to_be_deleted.Append(i)
                        elements_to_be_created.Append(row[0])
                else:
                    old_to_new_element_map[i] = -1
                    elements_to_be_deleted.Append(i)
                    for j in range(n):
                        elements_to_be_created.Append(row[j])

            elements_to_be_created.Sort()
            elements_to_be_created.Unique()

        else:
            # no mesh modifications happened
            old_to_new_element_map.SetSize(self.mesh.GetNE())
            for i in range(self.mesh.GetNE()):
                old_to_new_element_map[i] = i

        return old_to_new_element_map, elements_to_be_created, elements_to_be_deleted

    @BasePDESolver.AssertSolver
    def pRefine(self, elem_action, action_type, initial_ref = False):

        if not initial_ref:
            # if action_type == increment_base and elem_action_prev == elem_action, the solution space is not
            # changing and there is no need to transfer solution from one space to another.
            # if action_type == increment_current and no elements are being refined, the solution space is not changing.
            if (action_type == "increment_base" and np.array_equal(self.elem_action_prev, elem_action)) or \
                (action_type == "increment_current" and np.all((elem_action==1))):
                return

            # Create copies mesh, fes, gf in old state
            self.mesh_old = mfem.Mesh(self.mesh)
            self.fes_old = mfem.FiniteElementSpace(self.mesh_old, self.fec)
            self.sol_old = mfem.GridFunction(self.fes_old)

            # Update orders in old fes to current state
            for elem in range(self.mesh.GetNE()):
                order = self.fespace.GetElementOrder(elem)
                self.fes_old.SetElementOrder(elem, order)

            # Update for order change
            self.fes_old.Update(False)
            self.sol_old.Update()

            # Copy over old data
            self.sol_old.Assign(self.solution)

        # Update fes to new refinement state
        if (action_type == "increment_base"):
            for elem in range(self.mesh.GetNE()):
                self.fespace.SetElementOrder(elem, self.order+elem_action[elem])
        else:
            for elem in range(self.mesh.GetNE()):
                order = self.fespace.GetElementOrder(elem)
                action = elem_action[elem]-1
                if self.min_order <= order+int(action) <= self.max_order:
                    self.fespace.SetElementOrder(elem, order+action)
        if not initial_ref:
            self.fespace.Update(False)
            self.solution.Update()

            T = mfem.PRefinementTransferOperator(self.fes_old, self.fespace)
            self.solution = mfem.GridFunction(self.fespace)
            T.Mult(self.sol_old, self.solution)

        self.elem_action_prev = elem_action
        self.Update()

    @BasePDESolver.AssertSolver
    def Step(self):
        self.max_err = np.zeros((self.mesh.GetNE(), self.nvars))

        # t_step == 0.0 means no-op the integration. this is useful for testing.
        if self.t_step == 0.0:
            return 0

        assert self.t_step > self.dt/2.0, 'Time increment must be greater than time step/2.0. '
        t_next = self.t + self.t_step
        sum_dofs = 0
        while self.t <= t_next - self.dt/2.0:
            dt = min(self.dt, self.t_final-self.t)
            self.t, _ = self.ode_solver.Step(self.solution, self.t, dt)
            sum_dofs += self.fespace.GetNDofs()
            self.GetMaxElementErrors()

        return sum_dofs

    def GetMaxElementErrors(self, use_true_error=True):
        if use_true_error:
            err = self.GetElementErrors()
        else:
            err = self.GetElementErrorEstimates()
        self.max_err = np.maximum(err, self.max_err)
        return self.max_err

    def GetMaxErrorGridFunction(self):
        self.errorsfec = mfem.L2_FECollection(0, self.dim)
        self.errorsfes = mfem.FiniteElementSpace(self.mesh, self.errorsfec)
        self.maxerrors = mfem.GridFunction(self.errorsfes)

        for i in range(self.mesh.GetNE()):
            elem_dofs = self.errorsfes.GetElementDofs(i)
            self.maxerrors[elem_dofs[0]] = self.max_err[i]
        
        return self.maxerrors
