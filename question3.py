import numpy as np
import matplotlib.pyplot as plt

# Stop-Loss Strategy
def stop_loss(S0, K, T, r, sigma, M, transaction_cost=0.0, I=1000000):
    dt = T / M

    # Stock price paths
    S = np.zeros((M+1, I))
    S[0] = S0
    for t in range(1, M+1):
        z = np.random.randn(I)
        z -= z.mean()
        z /= z.std()
        S[t] = S[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

    # Stop-loss delta strategy (0 or 1)
    Delta = np.where(S > K, 1, 0)

    # Number of shares calculation
    Num_Stk = np.zeros((M+1, I))
    Num_Stk[0] = Delta[0] * 100000
    for t in range(1, M+1):
        Num_Stk[t] = (Delta[t] - Delta[t-1]) * 100000

    # Cost calculation with transaction costs
    Cost = np.zeros((M+1, I))
    Cost[0] = S[0] * Num_Stk[0] * (1 + transaction_cost)

    for t in range(1, M+1):
        Cost[t] = S[t] * Num_Stk[t] * (1 + transaction_cost)

    # Cumulative cost with financing
    Cum_Cost = np.zeros((M+1, I))
    Cum_Cost[0] = Cost[0]
    for t in range(1, M+1):
        Cum_Cost[t] = Cum_Cost[t-1] + Cost[t] + (Cum_Cost[t-1] * (np.exp(r*dt) - 1))

    # Final discounted
    final_cost = np.where(Cum_Cost[-1] >= K*100000,
                          (Cum_Cost[-1] - K*100000) * np.exp(-r*T),
                          Cum_Cost[-1] * np.exp(-r*T))

    # Compute statistics
    avg_cost = final_cost.mean()
    std_cost = final_cost.std()
    performance_ratio = std_cost / avg_cost if avg_cost != 0 else np.nan

    return avg_cost, std_cost, performance_ratio

def calculate_rebalancing(S0, K, r, T, sigma, transaction_cost, I, frequencies):
    print("Rebalancing Frequency Analysis:")
    results = []
    for frequency in frequencies:
        avg_cost, std_cost, performance_ratio = stop_loss(S0, K, T, r, sigma, frequency, transaction_cost, I)
        results.append(performance_ratio)
        print(f"Average hedging cost: ${avg_cost:.2f}, Standard deviation of cost: ${std_cost:.2f}, Performance ratio: {performance_ratio:.4f} for frequency: {frequency}")
    return results
\
if __name__ == "__main__":
    S0 = 50
    k = 50
    T = 1
    r = 0.05
    sigma = 0.3

    # daily, weekly, monthly, quarterly
    monitoring_frequencies = [1, 5, 22, 63]

    results = calculate_rebalancing(S0, k, r, T, sigma, 0.0, 1000000, monitoring_frequencies)

    plt.figure(figsize=(10,6))
    plt.plot(monitoring_frequencies, results, label='Performance Ratio', marker='o')
    plt.xlabel('Rebalancing Frequency')
    plt.ylabel('Performance Ratio')
    plt.title('Rebalancing Frequency vs Performance Ratio')
    plt.grid(True)
    plt.legend()
    plt.show()

