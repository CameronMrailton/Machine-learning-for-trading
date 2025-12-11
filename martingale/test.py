"""
Assess a betting strategy.

Template code for CS 4646/7646

Student Name: Tucker Balch (replace with your name)
GT User ID: crailton3 (replace with your User ID)
GT ID: 904071082 (replace with your GT ID)
"""

import numpy as np
import matplotlib.pyplot as plt


def author():
    return "crailton3"

def gtid():
    return 904071082

def study_group():
    return ["crailton3"]  # Add others if applicable

def get_spin_result(win_prob):
    return np.random.random() <= win_prob

def run_episode(win_prob=0.4737, spins=300, target=80):
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
        if total_winnings <= -bankroll:
            winnings.append(-bankroll)
            continue

        cash_available = bankroll + total_winnings
        bet = min(bet_amount, cash_available)

        if get_spin_result(win_prob):
            total_winnings += bet
            bet_amount = 1
        else:
            total_winnings -= bet
            bet_amount *= 2

        winnings.append(total_winnings)

    return winnings

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(gtid())

    win_prob = 18/38
    spins = 300
    target = 80

    # ---------------------------
    # Figure 1: 10 Martingale Episodes
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
    plt.show()

    # ---------------------------
    # Figure 2 and 3: Run 1000 episodes
    # ---------------------------
    all_episodes = [run_episode(win_prob, spins, target) for _ in range(1000)]
    all_episodes = np.array(all_episodes)

    mean_winnings = np.mean(all_episodes, axis=0)
    std_winnings = np.std(all_episodes, axis=0)
    median_winnings = np.median(all_episodes, axis=0)

    # ✅ Estimated probability of reaching $80
    reached_target = np.sum(np.max(all_episodes, axis=1) >= 80)
    estimated_prob = reached_target / all_episodes.shape[0]
    print(f"Estimated probability of reaching $80 in 300 spins: {estimated_prob:.4f}")
    reached_exactly_80 = np.sum([80 in episode for episode in all_episodes])
    estimated_prob_exact_80 = reached_exactly_80 / all_episodes.shape[0]
    print(f"Estimated probability of reaching exact $80 in 300 spins: {estimated_prob_exact_80:.4f}")

    #
    # # ---------------------------
    # # Figure 2: Mean ± Std Dev
    # # ---------------------------
    # plt.figure(figsize=(10, 6))
    # plt.plot(mean_winnings, label="Mean", color="blue")
    # plt.plot(mean_winnings + std_winnings, label="Mean + Std Dev", linestyle='--', color="green")
    # plt.plot(mean_winnings - std_winnings, label="Mean - Std Dev", linestyle='--', color="red")
    # plt.title("Figure 2: Mean ± Std Dev over 1000 Martingale Episodes")
    # plt.xlabel("Spin Number")
    # plt.ylabel("Winnings ($)")
    # plt.xlim(0, 300)
    # plt.ylim(-256, 100)
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("images/figure2.png")
    # plt.show()
    #
    # # ---------------------------
    # # Figure 3: Median ± Std Dev
    # # ---------------------------
    # plt.figure(figsize=(10, 6))
    # plt.plot(median_winnings, label="Median", color="blue")
    # plt.plot(median_winnings + std_winnings, label="Median + Std Dev", linestyle='--', color="green")
    # plt.plot(median_winnings - std_winnings, label="Median - Std Dev", linestyle='--', color="red")
    # plt.title("Figure 3: Median ± Std Dev over 1000 Martingale Episodes")
    # plt.xlabel("Spin Number")
    # plt.ylabel("Winnings ($)")
    # plt.xlim(0, 300)
    # plt.ylim(-256, 100)
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("images/figure3.png")
    # plt.show()
    #
    # # ---------------------------
    # # Figures 4 & 5: Realistic simulator (bankroll limit)
    # # ---------------------------
    # realistic_episodes = [realistic_simulator(win_prob, spins, target, bankroll=256) for]()_
    # ---------------------------
    # Experiment 2: Realistic Simulator (bankroll limited)
    # ---------------------------
    spins = 1000
    bankroll = 256
    realistic_episodes = [realistic_simulator(win_prob, spins, target, bankroll) for _ in range(1000)]
    realistic_episodes = np.array(realistic_episodes)

    # Calculate estimated probability of reaching exactly $80
    reached_exactly_80_realistic = np.sum([80 in episode for episode in realistic_episodes])
    estimated_prob_exact_80_realistic = reached_exactly_80_realistic / realistic_episodes.shape[0]

    print(f"Estimated probability of reaching exact $80 in 300 spins (Realistic): {estimated_prob_exact_80_realistic:.4f}")
