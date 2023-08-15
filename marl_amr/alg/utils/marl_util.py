"""Utility functions for MARL."""

import numpy as np


def batch_action_int_to_1hot(actions, dim_action):
    """Convert a batch of integer representation to 1-hot.

    Note that "batch_size" may be the real batch_size and (n_agents)
    combined

    Args:
        actions: np.array of shape [batch_size]
        dim_action: int

    Returns: np.array of shape [batch_size, dim_action]
    """
    # Convert to [batch_size, l_action]
    actions_1hot = np.zeros([len(actions), dim_action], dtype=int)
    actions_1hot[np.arange(len(actions)), actions] = 1

    return actions_1hot


def actions_int_to_1hot(actions, dim_action):
    """Convert a batch of integer representation to 1-hot.

    Args:
        actions: np.array of shape [batch_size, n_agents]
                 Each row of actions is one time step, containing
                 action integers for all agents
        dim_action: int

    Returns: np.array of shape [batch_size * n_agents, dim_action]
    where each row is a 1-hot representation for an agent.
    Batch and agent dimensions are combined.
    """
    batch_size = actions.shape[0]
    n_agents = actions.shape[1]

    # Convert to [time, agents, l_action]
    # so each agent gets its own 1-hot row vector
    actions_1hot = np.zeros([batch_size, n_agents, dim_action], dtype=int)
    grid = np.indices((batch_size, n_agents))
    actions_1hot[grid[0], grid[1], actions] = 1
    actions_1hot.shape = (batch_size * n_agents, dim_action)

    return actions_1hot


def unpack_batch_local(batch):
    """Extracts components from a batch of transition tuples.

    Each transition tuple has the following format:
    [array of obs, actions, reward, array of next obs, done]
    Global state is not explicitly recorded (it may or may not be
    fully defined by the collection of all obs).
    Groups components of the same type (e.g. obs) into its own np.array.

    batch: np.array of many transition tuples, where each transition
           is an np.array consisting of multi-agent observations,
           actions, reward(s), next observations, done, etc...

    Returns: an np.array for each component of the same type
             Batch dimension and agent dimension are combined:
             e.g., obs is [batch_size*n_agents, dim_obs]
    """
    # Note that (n_agents) indicates that the value could be
    # different for each batch entry
    obs = np.vstack(batch[:,0]) # [batch*(n_agents), dim_obs]
    n_agents = np.stack(batch[:,1]) # [batch]
    actions = np.hstack(batch[:,2]) # [batch*(n_agents)]
    reward = np.stack(batch[:,3]) # [batch]
    obs_next = np.vstack(batch[:,4]) # [batch*(n_agents), dim_obs]
    n_agents_next = np.stack(batch[:,5]) # [batch]
    done = np.stack(batch[:,6]) # [batch]

    return obs, n_agents, actions, reward, obs_next, n_agents_next, done


def unpack_batch(batch):
    """Extracts components from a batch of transition tuples.

    Groups components of the same type (e.g. obs) into its own np.array.
    Refer to replay_buffer for exact specification of batch.

    batch: np.array of many transition tuples, where each transition
           is an np.array consisting of multi-agent observations,
           actions, reward(s), next observations, done, etc...

    Returns: an np.array for each component of the same type
             Batch dimension and agent dimension are combined:
             e.g., obs is [batch_size*n_agents, dim_obs]
    """
    state = np.stack(batch[:,0]) # [batch, dim_state]
    obs = np.stack(batch[:,1]) # [batch, n_agents, dim_obs]
    actions = np.stack(batch[:,2]) # [batch, n_agents]
    reward = np.stack(batch[:,3]) # [batch]
    state_next = np.stack(batch[:,4]) # [batch, dim_state]
    obs_next = np.stack(batch[:,5]) # [batch, n_agents, dim_obs]
    done = np.stack(batch[:,6]) # [batch]

    batch = None

    batch_size = state.shape[0]
    n_agents = obs.shape[1]
    dim_obs = obs.shape[2]

    # In-place reshape for *_local quantities,
    # so that batch and agent dimensions are combined
    obs.shape = (batch_size*n_agents, dim_obs)
    obs_next.shape = (batch_size*n_agents, dim_obs)

    return (batch_size, state, obs, actions, reward, state_next,
            obs_next, done)
