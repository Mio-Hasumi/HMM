import numpy as np

class HMM:
    """
    First-order Hidden Markov Model (HMM)
    
    Attributes
    ----------
    transition_probs : numpy.ndarray
        State transition probability matrix of shape (N, N)
    emission_probs : numpy.ndarray
        Emission probability matrix of shape (N, M), where N is the number of states
        and M is the number of observation symbols
    initial_probs : numpy.ndarray
        Initial state probability vector of shape (N,)
    epsilon : float
        Small constant to prevent zero probabilities
    """
    def __init__(self, transition_probs, emission_probs, initial_probs, epsilon=1e-6):
        """
        Initialize the HMM model, ensuring that probability matrices are normalized
        and avoiding zero probabilities.
        
        Parameters
        ----------
        transition_probs : numpy.ndarray
            State transition probability matrix
        emission_probs : numpy.ndarray
            Emission probability matrix
        initial_probs : numpy.ndarray
            Initial state probability vector
        epsilon : float, optional
            Small constant to avoid zero probabilities, default is 1e-6
        """
        self.epsilon = epsilon
        # Avoid zero probabilities by adding epsilon and renormalizing
        self.transition_probs = transition_probs + self.epsilon
        self.transition_probs /= self.transition_probs.sum(axis=1, keepdims=True)
        
        self.emission_probs = emission_probs + self.epsilon
        self.emission_probs /= self.emission_probs.sum(axis=1, keepdims=True)
        
        self.initial_probs = initial_probs + self.epsilon
        self.initial_probs /= self.initial_probs.sum()

    def generate_sequence(self, sequence_length):
        """
        Generate a sequence of observations and hidden states.
        
        Parameters
        ----------
        sequence_length : int
            Length of the sequence to generate
        
        Returns
        -------
        observations : numpy.ndarray
            Generated observation sequence
        states : numpy.ndarray
            Generated hidden state sequence
        """
        observations = np.zeros(sequence_length, dtype=int)
        states = np.zeros(sequence_length, dtype=int)

        # Initial state and observation
        states[0] = np.random.choice(len(self.initial_probs), p=self.initial_probs)
        observations[0] = np.random.choice(
            len(self.emission_probs[states[0]]),
            p=self.emission_probs[states[0], :]
        )

        # Generate subsequent states and observations
        for t in range(1, sequence_length):
            states[t] = np.random.choice(
                len(self.transition_probs[states[t-1]]),
                p=self.transition_probs[states[t-1], :]
            )
            observations[t] = np.random.choice(
                len(self.emission_probs[states[t]]),
                p=self.emission_probs[states[t], :]
            )

        return observations, states

    def forward_algorithm(self, observation_sequence):
        """
        Perform the Forward algorithm with scaling to compute the forward probability
        matrix and scaling factors.
        
        Parameters
        ----------
        observation_sequence : list or np.ndarray
            Sequence of observations
        
        Returns
        -------
        forward_probs : numpy.ndarray
            Scaled forward probability matrix of shape (N, T)
        scaling_factors : numpy.ndarray
            Scaling factors for each time step of shape (T,)
        """
        num_states = self.transition_probs.shape[0]
        seq_length = len(observation_sequence)

        forward_probs = np.zeros((num_states, seq_length))
        scaling_factors = np.zeros(seq_length)

        # Initialization
        forward_probs[:, 0] = self.initial_probs * self.emission_probs[:, observation_sequence[0]]
        scaling_factors[0] = forward_probs[:, 0].sum()
        if scaling_factors[0] == 0:
            raise ValueError("Scaling factor at t=0 is zero.")
        forward_probs[:, 0] /= scaling_factors[0]

        # Recursion
        for t in range(1, seq_length):
            for state in range(num_states):
                forward_probs[state, t] = (
                    np.dot(forward_probs[:, t-1], self.transition_probs[:, state])
                    * self.emission_probs[state, observation_sequence[t]]
                )
            scaling_factors[t] = forward_probs[:, t].sum()
            if scaling_factors[t] == 0:
                raise ValueError(f"Scaling factor at t={t} is zero.")
            forward_probs[:, t] /= scaling_factors[t]

        return forward_probs, scaling_factors

    def backward_algorithm(self, observation_sequence, scaling_factors):
        """
        Perform the Backward algorithm with scaling to compute the backward probability
        matrix.
        
        Parameters
        ----------
        observation_sequence : list or np.ndarray
            Sequence of observations
        scaling_factors : numpy.ndarray
            Scaling factors from the Forward algorithm
        
        Returns
        -------
        backward_probs : numpy.ndarray
            Scaled backward probability matrix of shape (N, T)
        """
        num_states = self.transition_probs.shape[0]
        seq_length = len(observation_sequence)

        backward_probs = np.zeros((num_states, seq_length))

        # Initialization
        backward_probs[:, -1] = 1.0 / scaling_factors[-1]

        # Recursion
        for t in reversed(range(seq_length - 1)):
            for state in range(num_states):
                backward_probs[state, t] = np.sum(
                    self.transition_probs[state, :]
                    * self.emission_probs[:, observation_sequence[t+1]]
                    * backward_probs[:, t+1]
                )
            backward_probs[:, t] /= scaling_factors[t]

        return backward_probs

    def compute_observation_probability(self, observation_sequence):
        """
        Compute the log probability of the observation sequence using the
        Forward algorithm.
        
        Parameters
        ----------
        observation_sequence : list or np.ndarray
            Sequence of observations
        
        Returns
        -------
        log_prob : float
            Log probability of the observation sequence
        """
        _, scaling_factors = self.forward_algorithm(observation_sequence)
        log_prob = np.sum(np.log(scaling_factors + self.epsilon))
        return log_prob

    def viterbi_algorithm(self, observation_sequence):
        """
        Perform the Viterbi algorithm to find the most probable state path.
        
        Parameters
        ----------
        observation_sequence : list or np.ndarray
            Sequence of observations
        
        Returns
        -------
        viterbi_probs : numpy.ndarray
            Log probability matrix of shape (N, T)
        pointers : numpy.ndarray
            Backpointers matrix of shape (N, T)
        """
        num_states = self.transition_probs.shape[0]
        seq_length = len(observation_sequence)

        pointers = np.zeros((num_states, seq_length), dtype=int)
        viterbi_probs = np.zeros((num_states, seq_length))

        # Initialization
        viterbi_probs[:, 0] = (
            np.log(self.initial_probs + self.epsilon)
            + np.log(self.emission_probs[:, observation_sequence[0]] + self.epsilon)
        )

        # Recursion
        for t in range(1, seq_length):
            for state in range(num_states):
                prob = viterbi_probs[:, t-1] + np.log(self.transition_probs[:, state] + self.epsilon)
                pointers[state, t] = np.argmax(prob)
                viterbi_probs[state, t] = (
                    np.max(prob)
                    + np.log(self.emission_probs[state, observation_sequence[t]] + self.epsilon)
                )

        return viterbi_probs, pointers

    def decode_state_path(self, observation_sequence):
        """
        Decode the most probable state path using the Viterbi algorithm.
        
        Parameters
        ----------
        observation_sequence : list or np.ndarray
            Sequence of observations
        
        Returns
        -------
        best_log_prob : float
            Log probability of the best state path
        best_path : list
            Most probable state path
        """
        V, pointers = self.viterbi_algorithm(observation_sequence)
        last_state = np.argmax(V[:, -1])
        best_log_prob = V[last_state, -1]
        best_path = [last_state]
        for t in range(len(observation_sequence) - 1, 0, -1):
            last_state = pointers[last_state, t]
            best_path.insert(0, last_state)
        return best_log_prob, best_path

    def train_baum_welch(self, observations, convergence_threshold=0.05, max_iterations=1000):
        """
        Train the HMM using the Baum-Welch algorithm, with smoothing to prevent zero probabilities.
        
        Parameters
        ----------
        observations : list or numpy.ndarray
            Observation sequence (or a list of them, if you wish to extend)
        convergence_threshold : float, optional
            Convergence threshold (default 0.05)
        max_iterations : int, optional
            Maximum number of iterations (default 1000)
        
        Returns
        -------
        log_likelihoods : list
            List of log-likelihoods for each iteration
        """
        # Convert
        if isinstance(observations, np.ndarray) and observations.ndim == 1:
            observations = [observations]

        num_states = self.transition_probs.shape[0]
        log_likelihoods = []

        for iteration in range(max_iterations):
            # Only one sequence in this snippet here
            seq = observations[0]

            try:
                # Forward and Backward passes
                alpha, scaling_factors = self.forward_algorithm(seq)
                beta = self.backward_algorithm(seq, scaling_factors)
            except ValueError as e:
                print(f"Iteration {iteration}: {e}")
                break

            # Compute xi and gamma
            T = len(seq)
            xi = np.zeros((num_states, num_states, T - 1))
            for t in range(T - 1):
                denominator = np.sum(
                    alpha[:, t] * self.transition_probs
                    * self.emission_probs[:, seq[t+1]] * beta[:, t+1]
                )
                if denominator == 0:
                    print(f"Iteration {iteration}: Denominator zero at time {t}. Skipping this iteration.")
                    break
                for i in range(num_states):
                    numer = (
                        alpha[i, t]
                        * self.transition_probs[i, :]
                        * self.emission_probs[:, seq[t+1]]
                        * beta[:, t+1]
                    )
                    xi[i, :, t] = numer / (denominator + self.epsilon)
            else:
                # If no break occurred
                gamma = np.sum(xi, axis=1)
                # final gamma for time T-1
                final_gamma = alpha[:, -1] * beta[:, -1]
                final_gamma /= np.sum(final_gamma) + self.epsilon
                gamma = np.hstack((gamma, final_gamma.reshape(-1, 1)))

                # Update initial probs
                updated_initial_probs = gamma[:, 0]

                # Update transition probs
                updated_transition_probs = (
                    np.sum(xi, axis=2) / (np.sum(gamma[:, :-1], axis=1).reshape(-1, 1) + self.epsilon)
                )

                # Update emission probs
                updated_emission_probs = np.zeros_like(self.emission_probs)
                for state in range(num_states):
                    for symbol in range(self.emission_probs.shape[1]):
                        mask = (seq == symbol)
                        updated_emission_probs[state, symbol] = np.sum(gamma[state, mask])
                denom_emission = np.sum(gamma, axis=1) + self.epsilon
                updated_emission_probs = (updated_emission_probs.T / denom_emission).T

                # Compute log-likelihood
                log_prob = np.sum(np.log(scaling_factors + self.epsilon))
                log_likelihoods.append(log_prob)
                print(f"Iteration {iteration}: Log-Likelihood = {log_prob:.6f}")

                # ------------------
                # *** Smoothing Step ***
                # to prevent updated parameters from collapsing to zero
                updated_initial_probs += self.epsilon
                updated_initial_probs /= updated_initial_probs.sum()

                updated_transition_probs += self.epsilon
                updated_transition_probs /= updated_transition_probs.sum(axis=1, keepdims=True)

                updated_emission_probs += self.epsilon
                updated_emission_probs /= updated_emission_probs.sum(axis=1, keepdims=True)
                # ------------------

                # Check for convergence
                delta_initial = np.max(np.abs(self.initial_probs - updated_initial_probs))
                delta_transition = np.max(np.abs(self.transition_probs - updated_transition_probs))
                delta_emission = np.max(np.abs(self.emission_probs - updated_emission_probs))
                if max(delta_initial, delta_transition, delta_emission) < convergence_threshold:
                    print(f"Converged at iteration {iteration}.")
                    self.initial_probs = updated_initial_probs
                    self.transition_probs = updated_transition_probs
                    self.emission_probs = updated_emission_probs
                    break

                # Update parameters
                self.initial_probs = updated_initial_probs
                self.transition_probs = updated_transition_probs
                self.emission_probs = updated_emission_probs
        else:
            print("Baum-Welch did not converge within the maximum number of iterations.")

        return log_likelihoods

