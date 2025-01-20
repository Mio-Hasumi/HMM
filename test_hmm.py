import numpy as np
from hmm import HMM  # The HMM class
from visualize import (
    plot_forward_backward,
    plot_transition_emission,
    plot_viterbi_path,
    plot_log_likelihood
)

def test_hmm():
    """
    A minimal test and visualization for the HMM class, verifying:
    - Synthetic sequence generation
    - Forward/Backward calculations + heatmaps
    - Observation probability
    - Viterbi decoding + path visualization
    - Baum-Welch training + log-likelihood plotting
    - Transition/Emission matrix plotting (before/after training)
    """

    # 1. Define simple HMM parameters (2 states, 3 observations).
    transition_probs = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    emission_probs = np.array([
        [0.1, 0.4, 0.5],  # State 0: p(obs=0)=0.1, obs=1=0.4, obs=2=0.5
        [0.6, 0.3, 0.1]   # State 1: p(obs=0)=0.6, obs=1=0.3, obs=2=0.1
    ])
    initial_probs = np.array([0.6, 0.4])

    # 2. Instantiate the HMM
    hmm = HMM(transition_probs, emission_probs, initial_probs)

    # 3. Generate a synthetic sequence
    sequence_length = 10
    observations, hidden_states = hmm.generate_sequence(sequence_length)
    print("Generated Observations :", observations)
    print("Generated Hidden States:", hidden_states)

    # 4. (Optional) Visualize initial transition/emission probabilities
    print("\n--- Initial Transition and Emission Probabilities ---")
    plot_transition_emission(hmm.transition_probs, hmm.emission_probs)

    # 5. Run the Forward algorithm
    try:
        forward_probs, scaling_factors = hmm.forward_algorithm(observations)
        print("\n--- Forward Algorithm ---")
        print("Forward Probability Matrix:\n", forward_probs)
        print("Scaling Factors:", scaling_factors)
    except ValueError as e:
        print(f"Error during forward algorithm: {e}")
        return

    # 6. Run the Backward algorithm
    try:
        backward_probs = hmm.backward_algorithm(observations, scaling_factors)
        print("\n--- Backward Algorithm ---")
        print("Backward Probability Matrix:\n", backward_probs)
    except ValueError as e:
        print(f"Error during backward algorithm: {e}")
        return

    # 7. Visualize Forward/Backward probabilities + scaling
    plot_forward_backward(forward_probs, backward_probs, scaling_factors)

    # 8. Compute log probability of the observation sequence
    try:
        log_prob = hmm.compute_observation_probability(observations)
        print("\n--- Observation Probability ---")
        print(f"Log Probability of the sequence = {log_prob:.4f}")
    except ValueError as e:
        print(f"Error computing observation probability: {e}")

    # 9. Viterbi decoding
    try:
        best_log_prob, best_path = hmm.decode_state_path(observations)
        print("\n--- Viterbi Decoding ---")
        print(f"Best Log Probability: {best_log_prob:.4f}")
        print("Viterbi Path       :", best_path)
    except ValueError as e:
        print(f"Error in Viterbi decoding: {e}")
        return

    # 10. Visualize the Viterbi path
    plot_viterbi_path(observations, best_path)

    # 11. Baum-Welch training
    # We'll train on the same single sequence we generated.
    # Typically you'd want multiple sequences or a longer one.
    try:
        print("\n--- Baum-Welch Training ---")
        log_likelihoods = hmm.train_baum_welch(
            observations,
            convergence_threshold=0.001,
            max_iterations=20
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
    plot_transition_emission(hmm.transition_probs, hmm.emission_probs)

def main():
    test_hmm()

if __name__ == "__main__":
    main()

