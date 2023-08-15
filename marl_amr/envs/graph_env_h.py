"""Wrapper around env.py for policies that require graph information.

If edge feature is relative:
Edge feature between nodes i and j is a 1-hot vector of length 2*max_depth + 1
Value 1 at index k (k=0,...,2*max_depth) means that
depth(i) - depth(j) = k - max_depth

If edge feature is absolute:
Edge feature between nodes i and j is a 1-hot vector of length max_depth + 1
Value 1 at index k (k=0,...,max_depth) means that
depth(i) - depth(j) = k

Edge (i,j) also includes inner product <normed(x_i - x_j), v_i>
where v_i is velocity at i
"""

import os

import numpy as np

from marl_amr.envs.agents.h_ref_agent import AgentManager
from marl_amr.envs import graph_utils, util
from marl_amr.tools import amr_utils


class GraphEnv():
    """Multi-agent environment for h-refinement with graph information.

    Attributes:
        dim_action: int
            size of discrete action space
        dim_edge: int
            size of edge feature
        dim_obs: int
            size of observation vector in 1D case;
            tuple, sizes of observation matrix in 2D case
        dof_threshold: int
            max dof allowed, used in termination criteria
        enable_deref: Bool
            if True then de-refinement is enabled
        error_threshold: float
            used by solver for initial refinements
        log_obs: Bool
            if true apply np.log to error component of observation
        max_depth: int
            constraint on maximum depth of each element
        multi_objective: Bool
            if True then reward is a vector
        obs_uses_true_error: Bool
            if True, then observation uses true error
        observe_depth: Bool
            if true then include 1-hot depth
        observe_dof_and_time_balance: Bool
            if true then include dof/dof_threshold and t/t_final in observation
        penalize_dof_excess: Bool
            if True then applies penalty if exceeded dof_threshold at final time
        reward_index: int
            Advection only has one component
        reward_uses_true_error: Bool
            if True, then reward uses true error
        solver: instance of marl_amr.env.solvers.<solver>
        stopping_criteria: str
            'time', 'budget_or_time'
        t_final: float
            terminates at this simulation time (depending on stopping_criteria)
        t_history: int
            length of agent's observation history
            -1 means observation is 1-step difference in base obs
    """
    def __init__(self, config):

        # --------------- SOLVER ---------------------

        solver_name = config['solver_name']
        solver_config = config['solver']
        self.solver = util.GetSolver(solver_name, **solver_config)
        self.error_threshold = config['error_threshold']
        self.solver.error_threshold = self.error_threshold

        # Index of solution component used for reward
        self.reward_index = config.get('reward_index', 0)
        assert self.reward_index < self.solver.nvars

        self.penalize_dof_excess = config['penalize_dof_excess']
        self.reward_uses_true_error = config['reward_uses_true_error']
        self.multi_objective = config['multi_objective']

        # refinement mode
        self.ref_mode = self.solver.ref_mode
        print(f'Refinement mode: {self.ref_mode}')

        # stopping criteria type
        self.stopping_criteria = config['stopping_criteria']
        assert self.stopping_criteria in ['budget_or_time', 'time']

        self.dof_threshold = config['dof_threshold'] # Default: 1e5
        self.t_final = config.get('t_final', np.inf)
        self.solver.SetFinalTime(self.t_final)
        self.enable_deref = config.get('enable_deref', True)

        self.obs_uses_true_error = config['obs_uses_true_error'] # default: False

        self.log_obs = config['log_obs']
        self.max_depth = config['max_depth']
        self.observe_depth = config['observe_depth']
        self.observe_dof_and_time_balance = config['observe_dof_and_time_balance']
        self.t_history = config['t_history']
        self.agent_obs_type = config['agent_obs_type']

        if self.agent_obs_type == 'self':
            agent_base_obs_dim = (1*self.solver.nvars if self.t_history == -1
                                  else 1*self.solver.nvars*self.t_history)
        elif self.agent_obs_type == 'face_neighbors':
            agent_base_obs_dim = (5*self.solver.nvars if self.t_history == -1
                                  else 5*self.solver.nvars*self.t_history)

        low = -np.inf if self.log_obs else 0.0
        high = 1.0
        self.dim_obs = (agent_base_obs_dim +
                        (self.max_depth+1) * self.observe_depth +
                        2 * self.observe_dof_and_time_balance
                        + 2) # 2D velocity

        self.edge_feature_is_relative = config['edge_feature_is_relative']
        if self.edge_feature_is_relative:
            # relative difference in depth is a value within
            # -d_max, -d_max+1,...,0,...,d_max-1, d_max
            self.dim_edge_depth = 2 * self.max_depth + 1
            # map from difference in depth to active index in 1-hot vector
            # diff in depth ranges from -d_max to d_max
            self.map_diff_depth_to_index = lambda x: x + self.max_depth
        else:
            # +1 to indicate no difference in order
            self.dim_edge_depth = self.max_depth + 1
            # diff in depth ranges from 0 to max_depth
            self.map_diff_depth_to_index = lambda x: x
        self.dim_edge = self.dim_edge_depth + 1*hasattr(self.solver, 'vel')
        if self.enable_deref:
            # 0:de-refine, 1:no-op, 2:refine, will be shifted by -1
            self.dim_action = 3
            self.map_to_solver_action = lambda x: int(x-1)
        else:
            self.dim_action = 2 # 0:no-op, 1:refine
            self.map_to_solver_action = lambda x: int(x)

        self.dimensionless = config['dimensionless']

    def reset(self, force_case=None):
        """Resets the environment.

        Args:
            force_case: None, or int index of IC params case to use.

        Returns:
            3-tuple consisting of (
                np.array of node observations, shape [(N), dim_node]
                np.array of edge observations, shape [(N), dim_edge]
                2D np.array adjacency matrix
                ), and info dict
        """
        self.step_count = 0
        self.global_error_prvs = 1.0
        if self.solver.name == 'euler':
            self.solver.Reset()
        else:
            self.solver.Reset(force_case)
        self.mesh = self.solver.mesh
        self.init_global_error = self.solver.GetGlobalError()
        self.num_refined_cumulative = 0

        self.sum_of_dofs = self.solver.fespace.GetTrueVSize()
        self.dof_budget_balance = self.sum_of_dofs/self.dof_threshold
        self.time_budget_balance = self.solver.t/self.t_final
        self.dof_prvs_normed = self.dof_budget_balance

        self.agent_manager = AgentManager(self.mesh, self.t_history,
                                          self.agent_obs_type, self.solver.nvars)
        self.previous_reward = 0.0
        self.env_done = self.done()

        graph = nodes, edges, adj_matrix = self.get_obs()

        info = self.get_info_dict()

        return graph, info

    def apply_refinement_constraints(self, elems_choose_refine):
        """Applies constraints on elements that choose to be refined.

        Constraints implemented: max_depth constraint

        Args:
            elems_choose_refine: A list of element numbers that want to refine.
                                 Each number is that returned by node.elem.

        Returns:
            list of elements that want and are permitted to be refined.
            list of indicators for each element:
            1: element chose to refine and is permitted to refine
            -1: element is at max depth and not permitted to refine
        """
        action_codes = [-1] * len(elems_choose_refine)
        elems_to_refine = []
        for idx, elem in enumerate(elems_choose_refine):
            depth = self.mesh.ncmesh.GetElementDepth(elem)
            if depth < self.max_depth:
                action_codes[idx] = 1
                elems_to_refine.append(elem)

        return elems_to_refine, action_codes

    def step(self, action_dict, save_mesh_all_steps=False):
        """Takes one environment step.

        1. Extract map: elem->action from map: agent_id->action
        2. Apply constraints
        3. Calls solver.hRefine
        4. Updates refinement tree
        5. Deletes agents that are done
        6. Steps solver forward in time
        7. Compute new observation and reward        

        Args:
            action_dict: map from (str) agent_id (key in agent_manager.agents)
                         to int action in [dim_action]

        Returns: 
            3-tuple consisting of
                (np.array of node observations, shape [(N), dim_node]
                 np.array of edge observations, shape [(N), dim_edge]
                 2D np.array adjacency matrix),
            reward dict
            done dict
            info dict
        """
        self.step_count += 1

        # Make a list that defines the action of each element.
        # element_action_list = np.zeros(self.mesh.GetNE(), dtype=int)
        element_action_list = [0] * self.mesh.GetNE()
        for (agent_name, action) in action_dict.items():
            elem = self.agent_manager.agents[agent_name].elem()
            element_action_list[elem] = self.map_to_solver_action(action)
            # At this point, values in element_action_list mean:
            # if enable_deref: -1:deref, 0:no-op, 1:ref
            # else: 0:no-op, 1:ref

        (old_to_new_element_map, elements_to_be_created,
         elements_to_be_deleted) = self.solver.hRefine(
             element_action_list, aniso=self.solver.aniso,
             depth_limit=self.max_depth)
        self.agent_manager.refinement_update(old_to_new_element_map,
                                             elements_to_be_created,
                                             elements_to_be_deleted)
        # In this case, this count is not precise because it includes
        # elements that want to refine but are prevented from doing so
        # in solver.Refine. Don't need to be precise in the case of
        # h-ref-deref, since this measurement is used only to see roughly
        # how the policy behaves during training.
        map_action_to_count = dict(zip(
            *np.unique(element_action_list, return_counts=True)))
        if 1 in map_action_to_count:
            self.num_refined_cumulative += map_action_to_count[1]

        # Save mesh after action, but before solver step
        # to visualize the actions (mesh) chosen based on previous obs
        if save_mesh_all_steps:
            amr_utils.output_mesh(self, '../results/mesh_files',
                                  str(self.step_count)+'a')

        # Advance the PDE K timesteps
        self.solver.Step()

        # Sum the DOFs based on the current FESpace size
        self.sum_of_dofs += self.solver.fespace.GetTrueVSize()

        # DOF and TIME Budget balance
        self.dof_budget_balance = self.sum_of_dofs/self.dof_threshold
        self.time_budget_balance = self.solver.t/self.t_final

        # Check if the env is done based on DOF and Time
        self.env_done = self.done()

        graph = nodes, edges, adj_matrix = self.get_obs()
        reward_dict = self.get_r_dict()
        done_dict = self.get_done_dict()
        info_dict = self.get_info_dict()

        self.dof_prvs_normed = self.dof_budget_balance

        return graph, reward_dict, done_dict, info_dict        

    def get_obs(self):
        """Gets all agents' observations.

        Returns:
            np.array of node observations, shape [(N), dim_node]
            np.array of edge observations, shape [(N), dim_edge]
            2D np.array adjacency matrix
        """
        obs_dict = self.get_obs_dict()

        # This assigns a linear ordering on the agents.
        # adj_matrix and list_edges must be consistent with this ordering.
        list_nodes = list(obs_dict.values())

        self.map_element_id_to_idx = graph_utils.get_map_element_id_to_idx(
            self.agent_manager, obs_dict)
        self.map_agent_id_to_idx = graph_utils.get_map_agent_id_to_idx(obs_dict)
        adj_matrix = graph_utils.create_adjacency_matrix_by_element(
            self.agent_manager, self.solver, self.map_element_id_to_idx, 'h',
            self.edge_feature_is_relative)

        # Construct edge features. Must traverse adj_matrix in this order
        # because linear ordering of edges in list_edges must match
        # the order returned by np.nonzero(adj_matrix)[0] (sender)
        # and np.nonzero(adj_matrix)[1] (receiver)
        list_edges = []
        for idx_row, a_row in enumerate(self.agent_manager.agents.values()):
            for idx_col, a_col in enumerate(self.agent_manager.agents.values()):
                if idx_row == idx_col:
                    continue
                # Compute 1-hot initial edge feature.
                edge = np.zeros(self.dim_edge_depth)
                adj_value = adj_matrix[idx_row, idx_col]
                if adj_value != -np.inf:
                    active_index = int(self.map_diff_depth_to_index(adj_value))
                    edge[active_index] = 1
                    if hasattr(self.solver, 'vel'):
                        # normalized displacement vector
                        displacement = a_row.elem_center - a_col.elem_center
                        if self.solver.opposite_velocities:
                            vel = [np.tanh(100*(a_row.elem_center[1]-1.0)) *
                                   self.solver.ic_params['vx'],
                                   self.solver.ic_params['vy']]
                        elif self.solver.orbiting_velocities:
                            omega = self.solver.ic_params['omega']
                            vel = [omega * a_row.elem_center[1] -
                                   self.solver.ic_params['yc'],
                                   -omega * a_row.elem_center[0] -
                                   self.solver.ic_params['xc']]
                        else:
                            if hasattr(self.solver, 'nz') and self.solver.nz != 1:
                                vel = [self.solver.ic_params['vx'],
                                       self.solver.ic_params['vy'],
                                       self.solver.ic_params['vz']]
                            else:
                                vel = [self.solver.ic_params['vx'],
                                       self.solver.ic_params['vy']]
                        ip = np.dot(displacement/np.linalg.norm(displacement),
                                    vel)
                        if self.dimensionless:
                            ip *= self.solver.t_step / np.linalg.norm(displacement)
                        edge = np.concatenate((edge, [ip]))
                    list_edges.append(edge)

        # convert adj_matrix back to 0/1
        adj_matrix = np.where(adj_matrix != -np.inf, 1, 0)

        return np.array(list_nodes), np.array(list_edges), adj_matrix        

    def get_obs_dict(self):
        """Gets a map from agent_id to observation.

        Note that linear ordering of element errors given by
        solver.GetElementErrors does not match the spatial
        adjacency of elements in the case of h-refinement.

        Returns:
            map from agent_id (str) to agent observation (np.array)
        """
        observation_dict = {}
        if self.obs_uses_true_error:
            error_estimates = self.solver.GetElementErrors()
        else:
            error_estimates = self.solver.GetElementErrorEstimates()
        if self.observe_depth and self.observe_dof_and_time_balance:
            for agent_id, agent in self.agent_manager.agents.items():
                depth = self.mesh.ncmesh.GetElementDepth(agent.elem())
                depth_1hot = np.eye(self.max_depth+1)[depth]
                observation_dict[agent_id] = np.concatenate((
                    [self.dof_budget_balance, self.time_budget_balance],
                    agent.observation(error_estimates, self.log_obs), depth_1hot)
                )
        elif self.observe_depth:
            for agent_id, agent in self.agent_manager.agents.items():
                depth = self.mesh.ncmesh.GetElementDepth(agent.elem())
                depth_1hot = np.eye(self.max_depth+1)[depth]
                observation_dict[agent_id] = np.concatenate((
                    agent.observation(error_estimates, self.log_obs), depth_1hot)
                )
        elif self.observe_dof_and_time_balance:
            for agent_id, agent in self.agent_manager.agents.items():
                observation_dict[agent_id] = np.concatenate((
                    [self.dof_budget_balance, self.time_budget_balance],
                    agent.observation(error_estimates, self.log_obs))
                )
        else:
            for agent_id, agent in self.agent_manager.agents.items():
                observation_dict[agent_id] = np.concatenate((
                    agent.observation(error_estimates, self.log_obs))
                )

        return observation_dict

    def get_r_dict(self):
        """Computes a single team reward.

        Returns:
            reward_dict: dict
                map from int agent id to float reward value (same for all agents)
        """
        reward_dict = {}

        if self.reward_uses_true_error:
            global_error = self.solver.GetGlobalError()
        else:
            global_error = self.solver.GetGlobalErrorEstimate()
        global_error = global_error[self.reward_index]

        if self.multi_objective:
            reward = np.empty(2)
            # \sum_t r_t[0] = -log(e_T)
            reward[0] = np.log(self.global_error_prvs / global_error)
            # \sum_t r_t[1] = (dof_0 - dof_T)/(dof_threshold)
            reward[1] = self.dof_prvs_normed - self.sum_of_dofs/self.dof_threshold
        else:
            factor = 20
            penalty = factor*(self.solver.t - self.t_final)
            if self.env_done and abs(penalty)/factor > self.solver.dt:
                # Exceeded dof_threshold before reaching t_final
                reward = -(-penalty - np.log(self.global_error_prvs))
            else:
                reward = -np.log(global_error/self.global_error_prvs)

            if self.penalize_dof_excess and self.env_done and (
                    self.solver.t + self.solver.dt >= self.t_final):
                # reached final time
                reward -= factor*max(0, self.sum_of_dofs / self.dof_threshold - 1)

        self.global_error_prvs = global_error

        for (agent_id, agent) in self.agent_manager.agents.items():
            reward_dict[agent_id] = reward

        return reward_dict

    def get_info_dict(self):
        """Gets a map from agent_id to agent information, and global info.

        Returns:
            map from agent_id (str) to map from (str) keys to arbitrary types,
                and from key (str) to arbitrary types
        """
        info_dict = {}
        for (agent_id, agent) in self.agent_manager.agents.items():
            elem = agent.elem()
            agent_info = {}
            agent_info['depth'] = self.mesh.ncmesh.GetElementDepth(elem)
            info_dict[agent_id] = agent_info

        info_dict['dof_budget_balance'] = self.dof_budget_balance
        info_dict['global_error'] = self.global_error_prvs # computed in env.py
        info_dict['init_global_error'] = self.init_global_error
        info_dict['map_agent_id_to_idx'] = self.map_agent_id_to_idx
        info_dict['num_refined_cumulative'] = self.num_refined_cumulative
        info_dict['step_count'] = self.step_count
        info_dict['sum_of_dofs'] = self.sum_of_dofs
        info_dict['time_budget_balance'] = self.time_budget_balance

        if self.env_done:
            info_dict['true_global_error'] = self.solver.GetGlobalError()

        return info_dict

    def get_done_dict(self):
        """Gets a map from agent_id to Boolean indicator of agent termination."""
        done_dict = {}

        for (agent_id, agent) in self.agent_manager.agents.items():
            done_dict[agent_id] = self.env_done

        done_dict['__all__'] = self.env_done

        return done_dict

    def done(self):
        """Checks and returns Boolean indicator of whether episode is done."""
        if self.stopping_criteria == 'budget_or_time':
            return (self.sum_of_dofs > self.dof_threshold or
                    self.solver.t + self.solver.dt > self.t_final)
        elif self.stopping_criteria == 'time':
            return self.solver.t + self.solver.dt > self.t_final    
