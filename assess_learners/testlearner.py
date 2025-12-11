""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	 	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	 	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 		  		  		    	 		 		   		 		  
or edited.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	 	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import math  		  	   		 	 	 		  		  		    	 		 		   		 		  
import sys  		  	   		 	 	 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt

import numpy as np  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl
import DTLearner as DTLearn
import RTLearner as RTLearn
import BagLearner as bl
import InsaneLearner as il
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		 	 	 		  		  		    	 		 		   		 		  
        sys.exit(1)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])  		  	   		 	 	 		  		  		    	 		 		   		 		  
    data = np.array(
    [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()[1:]]
)
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    random = np.random.default_rng(42)          # pick any seed you like
    N = data.shape[0]
    n_train = int(0.60 * N)
    perm = random.permutation(N)
    tr_idx, te_idx = perm[:n_train], perm[n_train:]

    train_x = data[tr_idx, 0:-1]
    train_y = data[tr_idx, -1]
    test_x  = data[te_idx,  0:-1]
    test_y  = data[te_idx,  -1]
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # print(f"{test_x.shape}")
    # print(f"{test_y.shape}")
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    learner = DTLearn.DTLearner(verbose=True)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())


    leaf_sizes = [1, 2, 3, 5, 10, 20, 50, 100, 200]
    rmse_in, rmse_out = [], []

    for ls in leaf_sizes:
        learner = DTLearn.DTLearner(leaf_size=ls, verbose=False)
        learner.add_evidence(train_x, train_y)

        pred_in = learner.query(train_x)
        rmse_in.append(np.sqrt(np.mean((train_y - pred_in) ** 2)))

        pred_out = learner.query(test_x)
        rmse_out.append(np.sqrt(np.mean((test_y - pred_out) ** 2)))
    best_idx = int(np.argmin(rmse_out))

    plt.figure(figsize=(7, 4.5))
    plt.plot(leaf_sizes, rmse_in, marker='o', label='In sample RMSE')
    plt.plot(leaf_sizes, rmse_out, marker='o', label='Out of sample RMSE')
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.title('DTLearner with varying leaf size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/Figure_1.png')

    # experiment 2
    bags = 20

    rmse_out_single, rmse_out_bag = [], []
    rmse_in_single, rmse_in_bag = [], []

    for ls in leaf_sizes:
        base = DTLearn.DTLearner(leaf_size=ls, verbose=False)
        base.add_evidence(train_x, train_y)
        pred_in_single = base.query(train_x)
        pred_out_single = base.query(test_x)
        rmse_in_single.append(np.sqrt(np.mean((train_y - pred_in_single) ** 2)))
        rmse_out_single.append(np.sqrt(np.mean((test_y  - pred_out_single) ** 2)))

        bagger = bl.BagLearner(
            learner=DTLearn.DTLearner,
            kwargs={"leaf_size": ls, "verbose": False},
            bags=bags,
            boost=False,
            verbose=False
        )
        bagger.add_evidence(train_x, train_y)
        pred_in_bag = bagger.query(train_x)
        pred_out_bag = bagger.query(test_x)
        rmse_in_bag.append(np.sqrt(np.mean((train_y - pred_in_bag) ** 2)))
        rmse_out_bag.append(np.sqrt(np.mean((test_y  - pred_out_bag) ** 2)))


    plt.figure(figsize=(7, 4.5))
    plt.plot(leaf_sizes, rmse_out_single, marker='o', label='Single DT Out of sample RMSE')
    plt.plot(leaf_sizes, rmse_out_bag, marker='o', label=f'Bagged DT ({bags} bags) Out of sample RMSE')
    plt.plot(leaf_sizes, rmse_in_single, marker='o', linestyle='--', label='Single DT In sample RMSE')
    plt.plot(leaf_sizes, rmse_in_bag, marker='o', linestyle='--', label=f'Bagged DT ({bags} bags) In sample RMSE')
    plt.xlabel('leaf size')
    plt.ylabel('RMSE')
    plt.title(f'Bagging effect on DTLearner ({bags} bags)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/Figure_2.png')

    import time, numpy as np

    repeats = 10

    def mae(actual, prediction):
        return float(np.mean(np.abs(np.ravel(actual) - np.ravel(prediction))))

    def time_build(learner_cls, leaf_size):
        m = learner_cls(leaf_size=leaf_size, verbose=False)
        t0 = time.perf_counter()
        m.add_evidence(train_x, train_y)
        return time.perf_counter() - t0, m

    def build_time_stats(learner_cls):
        mean_ms, std_ms = [], []
        for ls in leaf_sizes:
            ts = [time_build(learner_cls, ls)[0] for _ in range(repeats)]
            mean_ms.append(1000 * np.mean(ts))
            std_ms.append(1000 * np.std(ts))
        return mean_ms, std_ms

    def oos_mae_curve(learner_cls):
        vals = []
        for ls in leaf_sizes:
            elapsed_sec, model = time_build(learner_cls, ls)
            vals.append(mae(test_y, model.query(test_x)))
        return vals

    dt_mean, dt_std = build_time_stats(DTLearn.DTLearner)
    rt_mean, rt_std = build_time_stats(RTLearn.RTLearner)
    dt_mae_oos = oos_mae_curve(DTLearn.DTLearner)
    rt_mae_oos = oos_mae_curve(RTLearn.RTLearner)

    x = np.arange(len(leaf_sizes))
    w = 0.38
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.bar(x - w/2, dt_mean, yerr=dt_std, width=w, label="DT build time", capsize=3)
    ax1.bar(x + w/2, rt_mean, yerr=rt_std, width=w, label="RT build time", capsize=3)
    ax1.set_xlabel("leaf size")
    ax1.set_ylabel("Build time (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(leaf_sizes)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, dt_mae_oos, marker='o', label="DT OOS MAE")
    ax2.plot(x, rt_mae_oos, marker='o', label="RT OOS MAE")
    ax2.set_ylabel("OOS MAE")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))

    plt.title("Build time vs leaf size with out of sample MAE ")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("images/Figure_3.png", bbox_inches="tight")
