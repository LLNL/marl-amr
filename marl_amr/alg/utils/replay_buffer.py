import random
import numpy as np

class Replay_Buffer():

    def __init__(self, size=1e6):
        self.memory = []
        self.maxsize = int(size)
        self.idx = 0

    def add(self, transition):
        if self.idx >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.idx] = transition
        self.idx = (self.idx + 1) % self.maxsize

    def sample_batch(self, size):
        """Samples a batch of transitions.
        
        Assumes that each entry in the replay buffer is a 
        single transition.
        """
        if len(self.memory) <= size:
            return np.array(self.memory)
        else:
            return np.array(random.sample(self.memory, size))

    def __len__(self):
        return len(self.memory)


class ReplayBufferNumpy():

    def __init__(self, size=1e6):
        self.memory = np.empty(size, dtype=object)
        self.size = int(size)
        self.idx = 0
        self.num_filled = 0

    def add(self, transition):
        self.memory[self.idx] = transition
        self.num_filled = min(self.size, self.num_filled + 1)
        self.idx = (self.idx + 1) % self.size

    def sample_batch(self, size):
        """Samples a batch of transitions.

        Assumes that each entry in the replay buffer is a
        single transition.
        """
        if self.num_filled <= size:
            return np.array(list(self.memory[:self.num_filled]))
        else:
            return np.array(list(
                np.random.choice(self.memory[:self.num_filled], size=size)))

    def __len__(self):
        return self.num_filled


class ReplayBufferEpisodic():
    """Transitions within each episode are kept together.

    Each episode may have different lengths.

    Uses the `done` value of each transition to determine whether
    to switch to a new episode.

    Attributes:
    idx_episode: int circular int index of episode
    idx_step: int index of episode step
    max_n_episodes: max number of episodes to store
    max_steps: int maximum number of time steps in each episode
    memory: list of lists
    n_samples: int total number of transitions currently in buffer
    """
    def __init__(self, max_n_episodes, max_steps):

        self.idx_step = 0
        self.idx_episode = 0
        self.max_size = size * max_steps
        self.max_steps = max_steps
        # self.memory = np.empty(size, dtype=object)
        # self.memory = [[] for _ in range(max_n_episodes)]
        self.memory = []
        self.n_samples = 0
        self.max_n_episodes = max_n_episodes

    def add(self, transition):
        """Inserts one transiiton tuple into buffer.

        Args:
            transition: list or tuple, where last entry is a Bool indicator
                of whether the episode is done
        """
        if self.idx_step == 0:
            if self.idx_episode >= len(self.memory):
                # first pass through memory, new episode, so start a new list
                self.memory.append([])
            else:
                # not first pass, new episode, clear out previous data
                self.memory[self.idx_episode] = []

        self.memory[self.idx_episode].append(transition)

        self.n_samples = min(self.max_size, self.n_samples + 1)

        if transition[-1] == True: # episode is done
            self.idx_episode = (self.idx_episode + 1) % self.max_n_episodes
            self.idx_step = 0
        else:
            self.idx_step += 1

    def sample_batch(self, batch_size, time_steps):
        """Randomly samples contiguous sequences from episodes.
        
        Samples according to the "random" method in DRQN paper.

        Args:
            batch_size: int number of episodes to sample
            time_steps: int number of contiguous sequence from each episode

        Returns:
            np.array of shape [batch_size, time_steps]
        """
        batch = np.empty((batch_size, time_steps), dtype=object)
        # sample random episode indices
        indices = random.sample(range(len(self.memory)), batch_size)
        for idx, idx_episode in enumerate(indices):
            episode = self.memory[idx_episode]
            # sample random t_start, subject to constraint that
            # [t_start, t_start + time_steps - 1] lies within the episode
            t_start = random.sample(range(len(episode) - time_steps), 1)[0]
            batch[idx] = episode[t_start : t_start + time_steps]

        return batch

    def __len__(self):
        return self.num_samples
