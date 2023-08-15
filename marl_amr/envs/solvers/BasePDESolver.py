import mfem.ser as mfem
import numpy as np
from mfem._ser.gridfunc import ProlongToMaxOrder


class BasePDESolver():
    warnings = True

    def __init__(self):
        self.initial_mesh = None
        self.mesh = None
        self.solution = None
        self.true_solution = None
        self.nx = self.ny = None
        self.t = 0.0
        self.t_history = 0
        self.t_final = np.inf
        self.solver_initialized = False
        self.sol_sock_soln = None
        self.gf_sock_soln = None
        self.sol_sock_err = None
        self.sol_sock_mesh = None
        self.min_order = 0
        self.max_order = 30
        self.single_step = False
        self.initial_to_current_table = None
        self.nvars = 1
        self.map_element_shape_to_mfem = {'quad': mfem.Element.QUADRILATERAL,
                                          'triangle': mfem.Element.TRIANGLE
        }

    def AssertSolver(func):
        '''
        Decorator for methods that require the solver objects to be initialized (using Reset()) prior to call.
        '''

        def wrapper(self, *args, **kwargs):
            assert self.solver_initialized, 'Solver must be initialized using Reset() before calling methods.'
            return func(self, *args, **kwargs)

        return wrapper

    def SetSingleStep(self):
        '''
        A debug mode which requires hitting enter to proceed through each step.
        '''
        print('SetSingleStep called.')
        self.single_step = True
        print(f'Printing self.single_step : {self.single_step}')

    def VerifyElementIndexBounds(func):
        '''
        Decorator for refinement/derefinement methods to verify that the element indices are within bounds.
        '''

        def wrapper(self, elems_to_refine, *args, **kwargs):
            if isinstance(elems_to_refine, int):
                elems_to_refine = [elems_to_refine]
            assert isinstance(elems_to_refine, list), 'Elements to refine must be of list type.'

            # It is valid to refine no elements
            if elems_to_refine != []:
                assert np.min(elems_to_refine) >= 0, f'Indices of elements to refine must be greater than zero. {np.min(elems_to_refine)}'
                assert np.max(elems_to_refine) < self.mesh.GetNE(), f'Indices of elements to refine must be less than number of elements. {np.max(elems_to_refine)}, {self.mesh.GetNE()}'

            return func(self, elems_to_refine, *args, **kwargs)

        return wrapper

    def GetMesh(self):
        '''
        Returns current state of mesh.

        Input: N/A
        Output: self.mesh (PyMFEM Mesh)
        '''

        return self.mesh

    @AssertSolver
    def GetMeshOrderingMap(self):
        NE = self.initial_mesh.GetNE()
        nx = self.nx
        ny = self.ny

        centroids = np.empty((NE, 2))
        idxs = np.arange(NE, dtype=int)

        v = mfem.Vector()
        for i in range(NE):
            self.initial_mesh.GetElementCenter(i, v)
            centroids[i, :] = v.GetDataArray()

        ix = np.argsort(centroids[:, 0])
        idxs = idxs[ix]
        centroids = centroids[ix]

        iy = np.argsort(centroids[:, 1], kind='mergesort')
        idxs = idxs[iy]
        centroids = centroids[iy]

        elem_map = idxs.reshape((nx, ny))

        return elem_map


    def SetFinalTime(self, t_final):
        '''
        Sets final time for simulation.

        Input: time
        Output: N/A
        '''
        self.t_final = t_final

    def GetSolution(self):
        '''
        Returns current state of solution.

        Input: N/A
        Output: self.solution (PyMFEM GridFunction)
        '''

        return self.solution

    def GetTrueSolution(self):
        '''
        Returns current state of true solution.

        Input: N/A
        Output: self.true_solution (PyMFEM Coefficient)
        '''

        return self.true_solution

    def Reset(self, seed=None):
        '''
        Resets solver to initial state.

        Input: RNG seed (optional)
        Output: N/A
        '''

        raise NotImplementedError('Reset() must be implemented in subclasses.')

    def Delete(self):
        '''
        Deletes objects in solver class.
        
        Input: N/A
        Output: N/A
        '''

        raise NotImplementedError('Delete() must be implemented in subclasses.')

    def hRefine(self, element_action_list):
        '''
        Performs h-(de)refinement 
        element_action_list = list with action for each element
        action_type: "increment_current": current element is refined (1), de-refined (-1) or no action (0)

        Input: actions
        Output: N/A
        '''

        raise NotImplementedError('hRefine() must be implemented in subclasses.')

    def pRefine(self, element_action_list, action_type):
        '''
        Performs p-refinement on elements.

        element_action_list = list with action for each element
        action_type: "increment_base": element order is set to either base order or base order+1
                     "increment_current": current element order is increased or decreased by 1.
        
        Output: N/A
        '''

        raise NotImplementedError('pRefine() must be implemented in subclasses.')

    def hDerefine(self, elems_to_derefine):
        '''
        Performs h-derefinement on elements with indices in elems_to_refine.
        
        Input: elems_to_derefine (list) 
        Output: N/A
        '''

        raise NotImplementedError('hDerefine() must be implemented in subclasses.')

    def pDerefine(self, elems_to_derefine):
        '''
        Performs p-derefinement on elements with indices in elems_to_refine.
        
        Input: elems_to_derefine (list) 
        Output: N/A
        '''

        raise NotImplementedError('pDerefine() must be implemented in subclasses.')

    def Step(self):
        '''
        Advances solver to next state.
        
        Input: N/A 
        Output: N/A
        '''

        raise NotImplementedError('Step() must be implemented in subclasses.')

    def UpdateTrueSolution(self):
        '''
        Updates true solution to current state if necessary (e.g. for time-dependent simulations).
        
        Input: N/A 
        Output: N/A
        '''

        pass

    def SetupEstimator(self):
        '''
        Sets up ZZ estimator for the current solution (must be called again after each refinement action). 
        
        Input: N/A
        Output: N/A
        '''

        self.diffusion = mfem.DiffusionIntegrator(mfem.ConstantCoefficient(1.0))
        self.estimator =  mfem.LSZienkiewiczZhuEstimator(self.diffusion, self.solution)

    @AssertSolver
    def GetElementErrors(self, norm=2):
        '''
        Computes errors (against self.true_solution) for each element in mesh. 
        
        Input: norm (optional)
        Output: elem_errs (np.array)
        '''

        assert norm in [1,2]

        self.UpdateTrueSolution()
        elem_errs = mfem.Vector(self.mesh.GetNE())

        if norm == 1:
            self.solution.ComputeElementL1Errors(self.true_solution, elem_errs)
        elif norm == 2:
            self.solution.ComputeElementL2Errors(self.true_solution, elem_errs)

        # explicitly copy into numpy array. the ref counting for the
        # mfem.Vector() seems unreliable outside this scope.
        np_errs = np.empty(self.mesh.GetNE())
        for n in range(self.mesh.GetNE()):
            np_errs[n] = elem_errs[n]

        np_errs = np.reshape(np_errs, (-1, self.nvars))
        return np_errs

    @AssertSolver
    def GetGlobalError(self, norm=2):
        '''
        Computes total error (against self.true_solution) for the solution. 
        
        Input: norm (optional)
        Output: error (np.array)
        '''

        assert norm in [1,2]
        elem_errs = self.GetElementErrors(norm)

        return np.linalg.norm(elem_errs, axis=0, ord=norm)


    @AssertSolver
    def GetElementErrorEstimates(self):
        '''
        Computes error estimates for each element in mesh using error estimator. 
        
        Input: N/A
        Output: elem_errs (np.array)
        '''
        self.estimator.Reset()

        elem_errs = self.estimator.GetLocalErrors().GetDataArray()
        elem_errs = np.reshape(elem_errs, (-1, self.nvars))
        return elem_errs

    @AssertSolver
    def GetGlobalErrorEstimate(self, norm=2):
        '''
        Computes total error estimate for the solution using error estimator. 
        
        Input: norm (optional)
        Output: error (float)
        '''
        assert norm in [1,2]
        elem_errs = self.GetElementErrorEstimates()

        return np.linalg.norm(elem_errs, axis=0, ord=norm)

    @AssertSolver
    def VisualizeSolution(self):
        '''
        Visualizes solution via GLVis. 
        
        Input: N/A
        Output: N/A
        '''
        if not self.sol_sock_soln:
            self.sol_sock_soln = mfem.socketstream('localhost', 19916)
            self.sol_sock_soln.precision(8)

        self.sol_sock_soln.send_solution(self.mesh, ProlongToMaxOrder(self.solution))

        title = f"Solution. Time = {self.t:.2f}'"
        self.sol_sock_soln.send_text(" window_title '" + title)


    @AssertSolver
    def VisualizeGridFunction(self, gf, title):
        '''
        Visualizes a gridfunction via GLVis.

        Input: gridfunction, title
        Output: N/A
        '''
        if not self.gf_sock_soln:
            self.gf_sock_soln = mfem.socketstream('localhost', 19916)
            self.gf_sock_soln.precision(8)

        self.gf_sock_soln.send_solution(self.mesh, ProlongToMaxOrder(gf))

        titlestr = title + f"Time = {self.t:.2f}'"
        self.gf_sock_soln.send_text(" window_title '" + titlestr)


    @AssertSolver
    def VisualizeError(self):
        '''
        Visualizes error via GLVis. 

        Input: N/A
        Output: N/A
        '''
        if not self.sol_sock_err:
            self.sol_sock_err = mfem.socketstream('localhost', 19916)
            self.sol_sock_err.precision(8)

        self.errfec = mfem.L2_FECollection(0, self.dim)
        self.errfes = mfem.FiniteElementSpace(self.mesh, self.errfec)
        self.err = mfem.GridFunction(self.errfes)

        elem_errs = self.GetElementErrors()

        for i in range(self.mesh.GetNE()):
            self.err[i] = elem_errs[i]

        self.sol_sock_err.send_solution(self.mesh, self.err)

        title = f"Error. Time = {self.t:.2f}'"
        self.sol_sock_err.send_text(" window_title '" + title)


    @AssertSolver
    def VisualizeMesh(self):
        '''
        Visualizes mesh order via GLVis. 

        Input: N/A
        Output: N/A
        '''
        prefix = ''
        if not self.sol_sock_mesh:
            prefix = 'keys ARjlmp*******'
            self.sol_sock_mesh = mfem.socketstream('localhost', 19916)
            self.sol_sock_mesh.precision(8)

        self.sol_sock_mesh.send_solution(self.mesh, self.GetOrderGridFunction())

        title = f"hp mesh. Time = {self.t}'"
        self.sol_sock_mesh.send_text(prefix + " window_title '" + title)

    @AssertSolver
    def GetOrderGridFunction(self):
        self.ordersfec = mfem.L2_FECollection(0, self.dim)
        self.ordersfes = mfem.FiniteElementSpace(self.mesh, self.ordersfec)
        self.orders = mfem.GridFunction(self.ordersfes)

        for i in range(self.mesh.GetNE()):
            elem_dofs = self.ordersfes.GetElementDofs(i)
            self.orders[elem_dofs[0]] = self.fespace.GetElementOrder(i)
        
        return self.orders

    @AssertSolver
    def GetElementDepthGridFunction(self):
        self.depthfec = mfem.L2_FECollection(0, self.dim)
        self.depthfes = mfem.FiniteElementSpace(self.mesh, self.depthfec)
        self.depth = mfem.GridFunction(self.depthfes)
        
        for i in range(self.mesh.GetNE()):
            elem_dofs = self.depthfes.GetElementDofs(i)
            self.depth[elem_dofs[0]] = self.mesh.ncmesh.GetElementDepth(i)
        
        return self.depth


    @AssertSolver
    def VectorToGridFunction(self, vec):
        self.gffec = mfem.L2_FECollection(0, self.dim)
        self.gffes = mfem.FiniteElementSpace(self.mesh, self.gffec)
        self.gf = mfem.GridFunction(self.gffes)
        for i in range(self.mesh.GetNE()):
            elem_dofs = self.gffes.GetElementDofs(i)
            self.gf[elem_dofs[0]] = np.float(vec[i])
        
        return self.gf


    def element_to_initial_elements(self,elem_errors):

        T = self.initial_to_current_table
        if T:
            coarse_elem_errors = np.zeros((T.Size(),1), dtype=np.float32)
            
            for i in range(T.Size()):
                n = T.RowSize(i)
                row = mfem.intArray(n)
                row.Assign(T.GetRow(i))
                for j in range(n):
                    coarse_elem_errors[i,:] += elem_errors[row[j],:]*elem_errors[row[j],:]
                coarse_elem_errors[i,:] = np.sqrt(coarse_elem_errors[i,:])  
        else:
            coarse_elem_errors = elem_errors

        return coarse_elem_errors

    
    def initial_elements_to_elements(self,coarse_action_list, type_double = False):

        T = self.initial_to_current_table
        if T:
            if (type_double):
                element_action_list = np.zeros(T.Width(), dtype=np.float32)
            else:
                element_action_list = np.zeros(T.Width(), dtype=int)
            for i in range(T.Size()):
                n = T.RowSize(i)
                row = mfem.intArray(n)
                row.Assign(T.GetRow(i))
                for j in range(n):
                    element_action_list[row[j]] = coarse_action_list[i]
        else:
           element_action_list = coarse_action_list

        return element_action_list    


    def GetRewardGridFunction(self, reward_dict):
        self.rewardsfec = mfem.L2_FECollection(0, self.dim)
        self.rewardsfes = mfem.FiniteElementSpace(self.mesh, self.rewardsfec)
        self.rewards = mfem.GridFunction(self.rewardsfes)

        if (self.ref_mode == 'h'):
            reward = np.zeros(self.mesh.GetNE(), dtype=np.float32)
            for key in reward_dict:
                reward[int(key)] = reward_dict[key]
            reward = self.initial_elements_to_elements(reward,type_double=True)
            for j in range(len(reward)):
                self.rewards[j] = reward[j].item()

        else:
            for key in reward_dict:
                self.rewards[int(key)] = reward_dict[key]

        return self.rewards

    
    def GetErrorGridFunction(self):
        self.errorsfec = mfem.L2_FECollection(0, self.dim)
        self.errorsfes = mfem.FiniteElementSpace(self.mesh, self.errorsfec)
        self.errors = mfem.GridFunction(self.errorsfes)
        err = self.GetElementErrors()

        for i in range(self.mesh.GetNE()):
            elem_dofs = self.errorsfes.GetElementDofs(i)
            self.errors[elem_dofs[0]] = err[i]
        
        return self.errors        
