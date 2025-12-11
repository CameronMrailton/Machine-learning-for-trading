import numpy as np


class BagLearner(object):

    def __init__(self, learner, kwargs, bags=20,boost=False, verbose=False):
        self.kwargs = kwargs
        self.learner_cls = learner
        self.bags = bags
        self.boost = boost
        self.tree = None
        self.learners = []
        self.verbose = verbose
        pass

    def author(self):
        return "crailton3"

    def add_evidence(self, data_x, data_y):
        rows = data_x.shape[0]
        for i in range(self.bags):
            indexs = np.random.randint(0,rows,size = rows)
            selected_x_train = data_x[indexs]
            selected_y_train = data_y[indexs]
            model = self.learner_cls(**self.kwargs)
            model.add_evidence(selected_x_train, selected_y_train)
            self.learners.append(model)

        if self.verbose:
            print(f"BagLearner Trained {len(self.learners)} bags on {rows}")

    def query(self, test_y):
        preds = []
        for model in self.learners:
            preds.append(model.query(test_y))  # each returns shape (m,)
        return np.mean(np.vstack(preds), axis=0)

