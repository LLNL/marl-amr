"""Functions used by various envs to construct graph information."""

import numpy as np


def get_map_element_id_to_idx(agent_manager, obs_dict):
    """Create map from agent's element id to its index in a linear ordering.

    Linear ordering is imposed by list_nodes = list(obs_dict.values()).
    Need this for consistency between list_obs, list_edges, and adj_matrix.

    Args:
        agent_manager: advection.envs.agents.{P,H}RefAgent.AgentManager
        obs_dict: map from agent_id (str) to agent observation (np.array)

    Returns: 
        map from int to int
    """
    map_element_id_to_index = {}
    for idx, agent_id in enumerate(obs_dict.keys()):
        agent = agent_manager.agents[agent_id]
        map_element_id_to_index[agent.elem()] = idx

    return map_element_id_to_index


def get_map_agent_id_to_idx(obs_dict):
    """Create map from agent's agent_id to its index in a linear ordering.

    Linear ordering is imposed by list_nodes = list(obs_dict.values()).

    Args:
        obs_dict: map from agent_id (str) to agent observation (np.array)

    Returns:
        map from str(int) to int
    """
    map_agent_id_to_idx = {}
    for idx, key in enumerate(obs_dict.keys()):
        map_agent_id_to_idx[key] = idx

    return map_agent_id_to_idx


def find_common_neighbors(map_eid_adjlist):
    """Finds the common neighbors among the input set of neighbors.

    Args:
        map_eid_adjlist: dictionary from int to np.array
    
    Returns: 
        map from int to np.array
    """
    new_map = {}
    for i, neighbors in map_eid_adjlist.items():
        list_2step_neighbors = []
        for eid in neighbors:
            list_2step_neighbors += list(map_eid_adjlist[eid])
        array_2step_neighbors = np.array(list_2step_neighbors)
        # An element j is a corner neighbor of element i if
        # j occurs more than once in the list of 2-step neighbors of i
        u, c = np.unique(array_2step_neighbors, return_counts=True)
        shared_neighbors = u[c > 1]
        # Remove element i from this set
        shared_neighbors = np.delete(shared_neighbors,
                                     np.where(shared_neighbors==i)[0])
        new_map[i] = np.unique(np.concatenate([
            neighbors, shared_neighbors]))

    return new_map


def create_adjacency_matrix_by_element(agent_manager, solver,
                                       map_element_id_to_idx,
                                       refinement_mode,
                                       edge_feature_is_relative):
    """Populates an adjacency matrix by searching through elements.

    Let i and j be the indices in the
    linear ordering of agents imposed by list(obs_dict.values()).
    If no edge exists between i and j, the adj_matrix[i,j] = -np.inf.
    If an edge exists, then 
    adj_matrix[i,j] is absolute or relative difference in order or depth between
    agent i and agent j.

    Args:
        agent_manager: advection.envs.agents.{P,H}RefAgent.AgentManager
        solver: advection.envs.solvers.<a solver>
        map_element_id_to_idx: map from int to int
        refinement_mode: 'p' or 'h'
        edge_feature_is_relative: bool

    Return: 
        2D np.array
    """
    elem_to_elem_table = solver.mesh.ElementToElementTable()

    # Map from (int) agent's element id to list (int)
    # adjacent agent's element id
    map_eid_adjlist = {}
    for agent in agent_manager.agents.values():
        map_eid_adjlist[agent.elem()] = elem_to_elem_table.GetRowList(
            agent.elem())

    # Here, map_edi_adjlist contains only neighbors that share a face,
    # without corner neighbors

    # Update adjacency list with corner neighbors
    map_eid_adjlist = find_common_neighbors(map_eid_adjlist)

    # Create adjacency matrix
    n_nodes = len(agent_manager.agents)
    adj_matrix = -np.inf * np.ones((n_nodes, n_nodes))
    diff_op = (lambda x: x) if edge_feature_is_relative else abs
    for e1, neighbors in map_eid_adjlist.items():
        for e2 in neighbors:
            if refinement_mode == 'p':
                diff = diff_op(solver.fespace.GetElementOrder(e1) -
                               solver.fespace.GetElementOrder(e2))
            elif refinement_mode == 'h':
                diff = diff_op(solver.mesh.ncmesh.GetElementDepth(e1) -
                               solver.mesh.ncmesh.GetElementDepth(e2))
            idx1 = map_element_id_to_idx[e1]
            idx2 = map_element_id_to_idx[e2]
            adj_matrix[idx1, idx2] = diff
    
    return adj_matrix
