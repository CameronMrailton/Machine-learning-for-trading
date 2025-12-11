import numpy as np

class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None
        pass

    def author(self):
        return "crailton3"

    def add_evidence(self, data_x, data_y):
        data = np.column_stack((data_x,data_y))
        self.tree = self.build_tree(data)

    def build_tree(self,data):
        rows, cols = data.shape
        X = data[:, :-1]
        y = data[:, -1]
        if rows <= self.leaf_size:
            return np.array([['leaf', float(np.mean(y)), np.nan, np.nan]], dtype=object)
        if np.all(y == y[0]):
            return np.array([['leaf', float(np.mean(y)), np.nan, np.nan]], dtype=object)

        else:
            corr_vals = []
            y_std = np.std(y)
            for i in range(cols - 1):
                xi = X[:, i]
                x_std = np.std(xi)
                if x_std == 0 or y_std == 0:
                    r = 0.0
                else:
                    r = np.corrcoef(xi, y)[0, 1]
                    if np.isnan(r):
                        r = 0.0
                corr_vals.append(abs(r))

            corr = np.array(corr_vals)
            best_feature = np.argmax(corr)
            SplitVal = float(np.median(X[:, best_feature]))
            left_mask = X[:, best_feature] <= SplitVal
            right_mask = ~left_mask

            if not left_mask.any() or not right_mask.any():
                return np.array([['leaf', float(np.mean(y)), np.nan, np.nan]], dtype=object)

            lefttree = self.build_tree(data[left_mask])
            righttree = self.build_tree(data[right_mask])
            root = np.array([[best_feature, SplitVal, 1, lefttree.shape[0] + 1]], dtype=object)

        return np.vstack((root, lefttree, righttree)).astype(object)

    def query(self, points):
        y_pred = []
        for x in points:
            node = 0
            while self.tree[node, 0] != 'leaf':
                feat_index = int(float(self.tree[node, 0]))
                split_val = float(self.tree[node, 1])
                if x[feat_index] <= split_val:
                    node += int(float(self.tree[node, 2]))
                else:
                    node += int(float(self.tree[node, 3]))
            y_pred.append(float(self.tree[node, 1]))
        return np.array(y_pred)
