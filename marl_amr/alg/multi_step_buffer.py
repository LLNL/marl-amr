"""n-step returns.

R_t^(n) = \sum_{k=0}^{n-1} \gamma^(k)_t R_{t+k}
So n=3 return is R_t + gamma R_{t+1} + gamma^2 R_{t+2}

Loss is
(R_t^(n) + \gamma^(n)_t \max_{a'} Q(s_{t+n}, a') - Q(s_t,a_t))^2

Note that the loss uses \gamma^n
"""
import numpy as np


class NStepReturns():

    def __init__(self, n, gamma):
        self.buf = []
        self.gamma = gamma
        self.n = n

    def update(self, state, action, reward, state_next):
        """Updates buffer and computes n-step returns

        Returns: (
            if there are n steps, (s_t, a_t, n-step return from t, s_{t+n})
            else, None
        """
        self.buf.append((state, action, reward, state_next))
        if len(self.buf) < self.n:
            return None

        n_step_return = np.sum([self.buf[i][2]*(self.gamma**i)
                                for i in range(self.n)])
        s_past, a_past, _, _ = self.buf.pop(0)

        return s_past, a_past, n_step_return, state_next
