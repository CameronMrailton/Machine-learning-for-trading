import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, bags=20, verbose=False):
        self.learners = [bl.BagLearner(lrl.LinRegLearner, {},20,False,verbose) for i in range(20)]
        self.verbose = verbose
        pass
    def author(self):
        return "crailton3"
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x,data_y)
    def query(self, x_test):
        preds = []
        for learner in self.learners:
            preds.append(learner.query(x_test))
        return np.mean(preds,axis=0)
