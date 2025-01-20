import numpy as np
from hmm import HMM
from visualize import (
    plot_forward_backward,
    plot_transition_emission,
    plot_viterbi_path,
    plot_log_likelihood
)

def test_hmm_weather():
    """
    A test and visualization for the HMM class applied to a weather prediction problem.
    Verifies:
    - Synthetic sequence generation
    - Forward/Backward calculations + heatmaps
    - Observation probability
    - Viterbi decoding + path visualization
    - Baum-Welch training + log-likelihood plotting
    - Transition/Emission matrix plotting (before/after training)
    """

    # 1. Define HMM parameters for the Weather Problem
    states = ['Sunny', 'Rainy']
    observations = ['Walk', 'Shop', 'Clean']
    n_states = len(states)
    n_observations = len(observations)

    # Mapping from state/observation to index
    state2idx = {state: idx for idx, state in enumerate(states)}
    obs2idx = {obs: idx for idx, obs in enumerate(observations)}
    idx2state = {idx: state for state, idx in state2idx.items()}
    idx2obs = {idx: obs for obs, idx in obs2idx.items()}

    # Transition Probabilities (A)
    transition_probs = np.array([
        [0.8, 0.2],  # From Sunny to Sunny/Rainy
        [0.4, 0.6]   # From Rainy to Sunny/Rainy
    ])

    # Emission Probabilities (B)
    emission_probs = np.array([
        [0.6, 0.3, 0.1],  # Sunny: Walk, Shop, Clean
        [0.1, 0.4, 0.5]   # Rainy: Walk, Shop, Clean
    ])

    # Initial State Probabilities (pi)
    initial_probs = np.array([0.5, 0.5])

    # 2. Instantiate the HMM
    hmm = HMM(transition_probs, emission_probs, initial_probs)

    # 3. Generate a synthetic sequence
    sequence_length = 15
    observations_seq, hidden_states_seq = hmm.generate_sequence(sequence_length)
    obs_names = [idx2obs[idx] for idx in observations_seq]
    state_names = [idx2state[idx] for idx in hidden_states_seq]
    print("Generated Observations :", obs_names)
    print("Generated Hidden States:", state_names)

    # 4. Visualize initial transition/emission probabilities
    print("\n--- Initial Transition and Emission Probabilities ---")
    plot_transition_emission(hmm.transition_probs, hmm.emission_probs, states, observations)

    # 5. Run the Forward algorithm
    try:
        forward_probs, scaling_factors = hmm.forward_algorithm(observations_seq)
        print("\n--- Forward Algorithm ---")
        print("Forward Probability Matrix:\n", forward_probs)
        print("Scaling Factors:", scaling_factors)
    except ValueError as e:
        print(f"Error during forward algorithm: {e}")
        return

    # 6. Run the Backward algorithm
    try:
        backward_probs = hmm.backward_algorithm(observations_seq, scaling_factors)
        print("\n--- Backward Algorithm ---")
        print("Backward Probability Matrix:\n", backward_probs)
    except ValueError as e:
        print(f"Error during backward algorithm: {e}")
        return

    # 7. Visualize Forward/Backward probabilities + scaling
    plot_forward_backward(forward_probs, backward_probs, scaling_factors, states)

    # 8. Compute log probability of the observation sequence
    try:
        log_prob = hmm.compute_observation_probability(observations_seq)
        print("\n--- Observation Probability ---")
        print(f"Log Probability of the sequence = {log_prob:.4f}")
    except ValueError as e:
        print(f"Error computing observation probability: {e}")

    # 9. Viterbi decoding
    try:
        best_log_prob, best_path = hmm.decode_state_path(observations_seq)
        best_states = [idx2state[idx] for idx in best_path]
        print("\n--- Viterbi Decoding ---")
        print(f"Best Log Probability: {best_log_prob:.4f}")
        print("Viterbi Path       :", best_states)
    except ValueError as e:
        print(f"Error in Viterbi decoding: {e}")
        return

    # 10. Visualize the Viterbi path
    plot_viterbi_path(observations_seq, best_path, states, observations)

    # 11. Baum-Welch training
    try:
        print("\n--- Baum-Welch Training ---")
        log_likelihoods = hmm.train_baum_welch(
            observations_seq,
            convergence_threshold=0.01,
            max_iterations=1000
        )
        print("Log-Likelihoods across iterations:", log_likelihoods)
    except ValueError as e:
        print(f"Error in Baum-Welch training: {e}")
        return

    # 12. Visualize the log-likelihood progression
    if log_likelihoods:
        plot_log_likelihood(log_likelihoods)

    # 13. Visualize updated transition/emission probabilities
    print("\n--- Updated Transition and Emission Probabilities ---")
    plot_transition_emission(hmm.transition_probs, hmm.emission_probs, states, observations)

def main():
    test_hmm_weather()

if __name__ == "__main__":
    main()
