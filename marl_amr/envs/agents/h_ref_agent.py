"""Bookkeeping for agent -> element id mapping.

Provides the same interface as HRefAgent.AgentManager.
Does not maintain a tree.
"""

import mfem.ser as mfem
import numpy as np


def get_element_face_neighbors(mesh, tel, iface):
    """Gets the element(s) that share face with input element

    Args:
        mesh: solver.mesh
        tel: this element's number
        iface: edge label

    Returns:
        neighbor_elems: mfem.intArray
        neighbor_faces: mfem.intArray
        case: (int) -1: bdr, 0: conforming, 1: slave, 2: master
              see mfem mesh.cpp
    """
    neighbor_elems = mfem.intArray()
    neighbor_faces = mfem.intArray()
    elems = mfem.intArray()
    efaces = mfem.intArray()
    case = mesh.GetFaceElementsAndFaces(iface, elems, efaces)
    if case != -1: 
        index = -1
        for i in range(elems.Size()):
            if elems[i] == tel: # skip over self
               index = i
               continue
            neighbor_elems.Append(elems[i])
            neighbor_faces.Append(efaces[i])

        neighbor_elems.Append(tel)
        neighbor_faces.Append(efaces[index])

    return neighbor_elems, neighbor_faces, case


class Agent():
    """Agent associated with a mesh element.

    Attributes:
        elem_id: int, an element id in the current mesh
        mesh: An mfem.Mesh object created inside solver
        t_history: int length of agent's observation history
            -1 means observation is 1-step difference in base obs
        obs_type: str, either 'self' or 'face_neighbors'
    """
    def __init__(self, elem_id, mesh, t_history=1, obs_type='face_neighbors',
                 solver_nvars=1):

        self.elem_id = elem_id
        self.mesh = mesh
        self.t_history = t_history
        self.obs_type = obs_type
        self.solver_nvars = solver_nvars
        if self.obs_type == 'self':
            self.dim = 1 * solver_nvars
        elif self.obs_type == 'face_neighbors':
            self.dim = 5 * solver_nvars

        if t_history != -1:
            self.obs_history = np.zeros([t_history, self.dim])
        else:
            self.obs_prev = np.zeros(self.dim)

        v = mfem.Vector()
        self.mesh.GetElementCenter(self.elem_id, v)
        self.elem_center = np.array(v.GetDataArray())

    def elem(self):
        return self.elem_id

    def observation(self, error_estimates, log_obs=False):
        """Gets this agent's observation.

        Observation is currently defined as the error_estimates at this
        agent's element and those of neighboring face elements.

        Args:
            error_estimates: error norms indexed by element
            log_obs: bool, whether observation is log of error or just error

        Returns:
            obs: 1D np.array
        """
        op = (lambda x: np.log(x+1e-15)) if log_obs else (lambda x: x)

        obs = np.zeros(self.dim) # TODO: generalize if more than 4 face elements
        # obs[0] = op(error_estimates[self.elem_id])
        obs[:self.solver_nvars] = op(error_estimates[self.elem_id])
        
        # Note: observing face_neighbors is not implemented for solver_nvars > 1
        if self.obs_type == 'face_neighbors':
            # both are lists of length 4
            el_edges, or0 = self.mesh.GetElementEdges(self.elem_id)
            # loop over all the faces
            for ei, edge in enumerate(el_edges):
                # mfem.intArrays of elem and face on this edge
                neighbor_elems, neighbor_faces, case = get_element_face_neighbors(
                    self.mesh, self.elem_id, edge)
                # now construct the observation corresponding to this face
                if case == -1: #boundary face
                    obs[ei+1] = 0.0
                else:
                    face_errors = []
                    for neighbor in neighbor_elems:
                        if neighbor != self.elem_id:
                            face_errors.append(error_estimates[neighbor])
                    obs[ei+1] = op(sum(face_errors)/len(face_errors))

        if self.t_history == -1:
            output_obs = obs - self.obs_prev
            self.obs_prev = obs
        else:
            self.obs_history = np.roll(self.obs_history, shift=1, axis=0)
            self.obs_history[0, :] = obs
            output_obs = self.obs_history.flatten()

        return output_obs


class AgentManager():
    """Interface between agents and the h-refinement environment.

    Note: Every agent that ever existed has a unique agent_ix.
    In contrast, elem numbers always between 0 and mesh.GetNE()-1.

    Attributes:
        mesh: An mfem.Mesh object created inside solver
        agents: A map from agent_id to Agent object
        agent_id: Unique monotonically increasing integer label for each
            agent that ever existed
    """    
    def __init__(self, mesh, t_history=1, agent_obs_type='face_neighbors',
                 solver_nvars=1):
        """Creates initial agents.

        Args:
            mesh: solver.mesh
            t_history: int length of agent's observation history
                -1 means observation is 1-step difference in base obs
            agent_obs_type: str, either 'self' or 'face_neighbors'
            solver_nvars: int, number of PDE variables
        """
        self.agents = {}
        self.mesh = mesh
        self.t_history = t_history
        self.agent_obs_type = agent_obs_type
        self.solver_nvars = solver_nvars

        self.agent_id = 0
        self.create_initial_agents()

    def create_initial_agents(self):
        """Populates the initial map from agent_id to Agent."""
        for elem in range(self.mesh.GetNE()):
            self.agents[str(self.agent_id)] = Agent(
                elem, self.mesh, self.t_history, self.agent_obs_type,
                self.solver_nvars)
            self.agent_id += 1

    def refinement_update(self, old_to_new_element_map, elements_to_be_created,
                          elements_to_be_deleted):
        """Updates the map from agent_id to Agent object.

        To be executed after a solver.Refine step.

        Args:
            old_to_new_element_map: 1D array where arr[i] = -1 means the old 
                element i either refined or derefined, and arr[i]=j!=-1 means 
                old element i was renumbered to j
            elements_to_be_created: array of new mesh element ids
            elements_to_be_deleted: array of old mesh element ids who either 
                refined or derefined
        """
        temp = {}

        # Delete agents who refined or derefined, and relabel the agents
        # who did nothing
        delete_set = set(elements_to_be_deleted)
        for agent_id, agent in self.agents.items():
            old_elem_id = agent.elem_id
            new_elem_id = old_to_new_element_map[old_elem_id]
            if old_elem_id in delete_set:
                # del self.agents[agent_id]
                continue
            else:
                assert new_elem_id != -1
                agent.elem_id = new_elem_id
                temp[agent_id] = agent

        # create the new agents
        for elem_id in elements_to_be_created:
            temp[str(self.agent_id)] = Agent(
                elem_id, self.mesh, self.t_history, self.agent_obs_type,
                self.solver_nvars)
            self.agent_id += 1

        del self.agents
        self.agents = temp
