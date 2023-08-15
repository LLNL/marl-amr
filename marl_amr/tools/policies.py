import numpy as np


def LoadPolicyByName(name, *args, **kwargs):
    classnames = []

    for sclass in BasePolicy.__subclasses__():
        classnames.append(sclass.name)
        if sclass.name == name.lower():
            return sclass(*args, **kwargs)

    raise ValueError(f'Could not find policy named {name} in list of policies: {classnames}')

class BasePolicy:
    '''
    Policy class takes in an Estimator instance and computes the action dict (elements to refine)
    for the current solution.
    '''
    def __init__(self, estimator):
        self.estimator = estimator
        self.action_type = None
        self.action_mapping = None

    def SetActionMapping(self, action_type):
        assert action_type in ['increment_base', 'increment_current'], action_type

        self.action_type = action_type

        if action_type == 'increment_base':
            self.action_mapping = {'no-op'      :  0,
                                   'refine'     :  1}
        elif action_type == 'increment_current':
            self.action_mapping = {'derefine'   : -1,
                                   'no-op'      :  0,
                                   'refine'     :  1}


    def GetEnvActionFromPolicyActionType(self, policy_action_type):
        '''
        Given a policy action type, compute the corresponding action for the environment using self.action_mapping.
        '''
        assert policy_action_type in ['derefine', 'no-op', 'refine'], f'Policy action type must be in [derefine, no-op, refine], {policy_action_type}'
        assert self.action_type and self.action_mapping, 'Must set action mapping using SetActionMapping prior to calling policy.'

        if policy_action_type not in self.action_mapping:
            raise ValueError(f'Action ({policy_action_type}) not supported by action_type ({self.action_type}).')

        return self.action_mapping[policy_action_type]

    def ComputeRefinementActions(self, scores):
        '''
        Computes a list of element indices to refine from the estimator.
        This method is wrapped by GetActionDict() to compute scores from the estimator
        and transform the list of element indices to an action dict.
        '''
        pass

    def ComputeDerefinementActions(self, scores):
        '''
        Computes a list of element indices to derefine from the estimator.
        This method is wrapped by GetActionDict() to compute scores from the estimator
        and transform the list of element indices to an action dict.
        '''
        pass

    def GetActionDict(func):
        '''
        Decorator for methods to compute action dictionary from list of elements to refine.
        '''

        def wrapper(self):
            NE = self.estimator.solver.mesh.GetNE()
            scores = self.estimator.ComputeScores()
            assert len(scores) == NE, 'Size of estimator scores must be equal to the number of elements.'
            
            [actions, policy_action_type] = func(self, scores)
            action_dict = {}

            # Set all actions to no-op initially
            for i in range(NE):
                action_dict[str(i)] = self.GetEnvActionFromPolicyActionType('no-op')

            # Set actions in action list to action_type
            for i in actions:
                action_dict[str(i)] = self.GetEnvActionFromPolicyActionType(policy_action_type)

            return action_dict

        return wrapper

class RandomPolicy(BasePolicy):
    '''
    RandomPolicy randomly refines an element with a probability equal to p.
    '''
    name = 'random'
    def __init__(self, estimator, p):
        super().__init__(estimator)
        self.p = float(p) # Probability of refining

        assert self.p >= 0 and self.p <= 1, 'Probability must be between 0 and 1.'

    @BasePolicy.GetActionDict
    def ComputeRefinementActions(self, scores):
        policy_action_type = 'refine'
        NE = self.estimator.solver.mesh.GetNE()
        k = int(self.p*NE)

        actions = np.random.choice(NE, size=k, replace=False)

        return [actions, policy_action_type]

    @BasePolicy.GetActionDict
    def ComputeDerefinementActions(self, scores):
        policy_action_type = 'derefine'
        NE = self.estimator.solver.mesh.GetNE()
        k = int(self.p*NE)

        actions = np.random.choice(NE, size=k, replace=False)

        return [actions, policy_action_type]

class TopKPolicy(BasePolicy):
    '''
    TopKPolicy refines the top k elements according to the estimator scores.
    '''
    name = 'top_k'
    def __init__(self, estimator, k_ref, k_deref):
        super().__init__(estimator)

        self.k_ref = int(k_ref)
        self.k_deref = int(k_deref)

        NE = self.estimator.solver.mesh.GetNE()
        assert self.k_ref >= 0 and self.k_deref >= 0, 'TopKPolicy requires k_ref >= 0 and k_deref >= 0.'
        assert (self.k_ref + self.k_deref) > 0,       'TopKPolicy requires k_ref + k_deref > 0.'
        assert (self.k_ref + self.k_deref) <= NE,     'TopKPolicy requires k_ref + k_deref <= NE.'


    @BasePolicy.GetActionDict
    def ComputeRefinementActions(self, scores):
        policy_action_type = 'refine'
        assert len(scores) >= self.k_ref, 'TopKPolicy requires number of elements to be >= k.'

        sorted_score_idxs = np.argsort(scores)[::-1]
        actions = sorted_score_idxs[:self.k_ref]

        return [actions, policy_action_type]

    @BasePolicy.GetActionDict
    def ComputeDerefinementActions(self, scores):
        policy_action_type = 'derefine'
        assert len(scores) >= self.k_deref, 'TopKPolicy requires number of elements to be >= k.'

        scores *= -1
        sorted_score_idxs = np.argsort(scores)[::-1]
        actions = sorted_score_idxs[:self.k_deref]

        return [actions, policy_action_type]


class TopPPolicy(BasePolicy):
    '''
    TopPPolicy refines the top p percent of elements according to the estimator scores.
    '''
    name = 'top_p'
    def __init__(self, estimator, p_ref, p_deref):
        super().__init__(estimator)

        self.p_ref = float(p_ref)
        self.p_deref = float(p_deref)

        assert 0 <= self.p_ref <= 1 and 0 <= self.p_deref <= 1, 'TopPPolicy requires p_ref and p_deref in [0., 1.].'
        assert (self.p_ref + self.p_deref) > 0,                 'TopPPolicy requires p_ref + p_deref > 0.'
        assert (self.p_ref + self.p_deref) <= 1,                'TopPPolicy requires p_ref + p_deref <= 1.'

    @BasePolicy.GetActionDict
    def ComputeRefinementActions(self, scores):
        policy_action_type = 'refine'
        NE = self.estimator.solver.mesh.GetNE()
        k = int(self.p_ref*NE)

        sorted_score_idxs = np.argsort(scores)[::-1]
        actions = sorted_score_idxs[:k]

        return [actions, policy_action_type]

    @BasePolicy.GetActionDict
    def ComputeDerefinementActions(self, scores):
        policy_action_type = 'derefine'
        NE = self.estimator.solver.mesh.GetNE()
        k = int(self.p_deref*NE)

        scores *= -1
        sorted_score_idxs = np.argsort(scores)[::-1]
        actions = sorted_score_idxs[:k]

        return [actions, policy_action_type]

class SingleThresholdPolicy(BasePolicy):
    '''
    SingleThresholdPolicy refines the elements with scores greater/less than threshold.
    '''
    name = 'single_threshold'
    def __init__(self, estimator, threshold):
        super().__init__(estimator)
        self.threshold = float(threshold)

    @BasePolicy.GetActionDict
    def ComputeRefinementActions(self, scores):
        policy_action_type = 'refine'
        NE = self.estimator.solver.mesh.GetNE()

        actions = []
        for i in range(NE):
            if scores[i] >= self.threshold:
                actions.append(i)

        return [actions, policy_action_type]

    @BasePolicy.GetActionDict
    def ComputeDerefinementActions(self, scores):
        policy_action_type = 'derefine'
        NE = self.estimator.solver.mesh.GetNE()

        actions = []
        for i in range(NE):
            if scores[i] <= self.threshold:
                actions.append(i)

        return [actions, policy_action_type]

class DoubleThresholdPolicy(BasePolicy):
    '''
    DoubleThresholdPolicy refines the elements with scores greater than threshold_high
    and derefines elements with scores less than threshold_low.
    '''
    name = 'double_threshold'
    def __init__(self, estimator, threshold_low, threshold_high):
        super().__init__(estimator)
        self.threshold_low = float(threshold_low)
        self.threshold_high = float(threshold_high)

        assert self.threshold_high >= self.threshold_low

    @BasePolicy.GetActionDict
    def ComputeRefinementActions(self, scores):
        policy_action_type = 'refine'
        NE = self.estimator.solver.mesh.GetNE()

        actions = []
        for i in range(NE):
            if scores[i] > self.threshold_high:
                actions.append(i)

        return [actions, policy_action_type]

    @BasePolicy.GetActionDict
    def ComputeDerefinementActions(self, scores):
        policy_action_type = 'derefine'
        NE = self.estimator.solver.mesh.GetNE()

        actions = []
        for i in range(NE):
            if scores[i] < self.threshold_low:
                actions.append(i)

        return [actions, policy_action_type]



class DoubleThresholdPolicyPRef(BasePolicy):
    '''
    DoubleThresholdPolicy refines the elements with scores greater than threshold_high
    and derefines elements with scores less than threshold_low.
    '''
    name = 'double_threshold_pref'
    def __init__(self, estimator, threshold_low, threshold_high):
        super().__init__(estimator)
        self.threshold_low = float(threshold_low)
        self.threshold_high = float(threshold_high)

        assert self.threshold_high >= self.threshold_low

    def GetActionDict(self, scores):
        action_dict = {}
        for i in range(len(scores)):
            if scores[i] > self.threshold_high:
                action_dict[str(i)] = 1
            elif  scores[i] < self.threshold_low:
                action_dict[str(i)] = 0
            else:
                action_dict[str(i)] = self.estimator.solver.fespace.GetElementOrder(i)-1 #keep the order same as now

        return action_dict

class DoubleThresholdPolicyHRef(BasePolicy):
    '''
    DoubleThresholdPolicy refines the elements with scores greater than threshold_high
    and derefines elements with scores less than threshold_low.
    '''
    name = 'double_threshold_href'
    def __init__(self, estimator, threshold_low, threshold_high):
        super().__init__(estimator)
        self.threshold_low = float(threshold_low)
        self.threshold_high = float(threshold_high)

        assert self.threshold_high >= self.threshold_low

    def GetActionDict(self, scores):
        action_dict = {}
        for i in range(len(scores)):
            if scores[i] > self.threshold_high:
                action_dict[str(i)] = 2
            elif  scores[i] < self.threshold_low:
                action_dict[str(i)] = 0
            else:
                action_dict[str(i)] = 1

        return action_dict