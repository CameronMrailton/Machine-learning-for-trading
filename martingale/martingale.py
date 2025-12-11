""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT User ID: crailton3 (replace with your User ID)  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT ID: 904071082 (replace with your GT ID)  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np
import matplotlib.pyplot as plt

  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def author():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return "crailton3"  # replace tb34 with your Georgia Tech username.
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def gtid():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return 904071082  # replace with your GT ID number
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    result = False  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        result = True  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return result  		  	   		 	 	 		  		  		    	 		 		   		 		  

def test_code():
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """
    np.random.seed(gtid())

    win_prob = 18/38
    spins = 300
    target = 80

    # ---------------------------
    # Figure 1: 10 episodes
    # ---------------------------
    plt.figure(figsize=(10, 6))
    for i in range(10):
        episode = run_episode(win_prob, spins, target)
        plt.plot(episode, label=f"Episode {i+1}")
    plt.title("Figure 1: 10 Martingale Episodes")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig("images/figure1.png")
    # plt.show()
    # plt.savefig("images/figure1.png")

    # ---------------------------
    # Run 1000 episodes for Figures 2 & 3
    # ---------------------------
    all_episodes = np.array([run_episode(win_prob, spins, target) for _ in range(1000)])
    mean_winnings = np.mean(all_episodes, axis=0)
    std_winnings = np.std(all_episodes, axis=0)
    median_winnings = np.median(all_episodes, axis=0)

    # ---------------------------
    # Figure 2: Mean ± Std Dev
    # ---------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(mean_winnings, label="Mean", color="blue")
    plt.plot(mean_winnings + std_winnings, label="Mean + Std Dev", linestyle='--', color="green")
    plt.plot(mean_winnings - std_winnings, label="Mean - Std Dev", linestyle='--', color="red")
    plt.title("Figure 2: Mean ± Std Dev over 1000 Martingale Episodes")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig("images/figure2.png")
    # plt.show()

    # ---------------------------
    # Figure 3: Median ± Std Dev
    # ---------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(median_winnings, label="Median", color="blue")
    plt.plot(median_winnings + std_winnings, label="Median + Std Dev", linestyle='--', color="green")
    plt.plot(median_winnings - std_winnings, label="Median - Std Dev", linestyle='--', color="red")
    plt.title("Figure 3: Median ± Std Dev over 1000 Martingale Episodes")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig("images/figure3.png")
    # plt.show()
    # test_code()
# --- Figure 4: Realistic simulator, mean ---
    realistic_data = [realistic_simulator() for _ in range(1000)]
    realistic_data = np.array(realistic_data)

    mean_real = np.mean(realistic_data, axis=0)
    std_real = np.std(realistic_data, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_real, label="Mean", color="blue")
    plt.plot(mean_real + std_real, label="Mean + Std Dev", linestyle='--', color="green")
    plt.plot(mean_real - std_real, label="Mean - Std Dev", linestyle='--', color="red")
    plt.title("Figure 4: Realistic Simulator - Mean ± Std Dev")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig("images/figure4.png")
    # plt.show()



# --- Figure 5: Realistic simulator, median ---
    median_real = np.median(realistic_data, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(median_real, label="Median", color="blue")
    plt.plot(median_real + std_real, label="Median + Std Dev", linestyle='--', color="green")
    plt.plot(median_real - std_real, label="Median - Std Dev", linestyle='--', color="red")
    plt.title("Figure 5: Realistic Simulator - Median ± Std Dev")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.grid(True)
    plt.savefig("images/figure5.png")
    # plt.show()

  		  	   		 	 	 		  		  		    	 		 		   		 		  
def run_episode(win_prob=0.4737, spins=300, target=80):
    """
    Runs one episode of the Martingale strategy.
    Returns winnings after each spin.
    """
    total_winnings = 0
    bet_amount = 1
    winnings = []

    for _ in range(spins):
        if total_winnings >= target:
            winnings.append(total_winnings)
            continue

        if get_spin_result(win_prob):
            total_winnings += bet_amount
            bet_amount = 1
        else:
            total_winnings -= bet_amount
            bet_amount *= 2

        winnings.append(total_winnings)

    return winnings
def realistic_simulator(win_prob=0.4737, spins=300, target=80, bankroll=256):
    total_winnings = 0
    bet_amount = 1
    winnings = []

    for _ in range(spins):
        if total_winnings >= target:
            winnings.append(total_winnings)
            continue
        elif total_winnings <= -bankroll:
            winnings.append(-bankroll)
            continue

        # Adjust bet if funds are insufficient
        available_cash = bankroll + total_winnings
        bet = min(bet_amount, available_cash)

        if get_spin_result(win_prob):
            total_winnings += bet
            bet_amount = 1
        else:
            total_winnings -= bet
            bet_amount *= 2

        winnings.append(total_winnings)

    return winnings


if __name__ == "__main__":
    test_code()
