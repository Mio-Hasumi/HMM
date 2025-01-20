import numpy as np

class HMM:
    """
    First-order Hidden Markov Model (HMM)
    
    Attributes
    ----------
    transition_probs : numpy.ndarray
        State transition probability matrix of shape (N, N).
    emission_probs : numpy.ndarray
        Emission probability matrix of shape (N, M), where N is the number of states
        and M is the number of observation symbols.
    initial_probs : numpy.ndarray
        Initial state probability vector of shape (N,).
    epsilon : float
        Small constant to prevent zero probabilities (default=1e-6).
    """
    def __init__(self, transition_probs, emission_probs, initial_probs, epsilon=1e-6):
        """
        Initialize the HMM model, ensuring that probability matrices are normalized
        and avoiding zero probabilities.
        
        Parameters
        ----------
        transition_probs : numpy.ndarray
            State transition probability matrix (N x N).
        emission_probs : numpy.ndarray
            Emission probability matrix (N x M).
        initial_probs : numpy.ndarray
            Initial state probability vector (length N).
        epsilon : float, optional
            Small constant to avoid zero probabilities, default is 1e-6.
        """
        self.epsilon = epsilon
        # Avoid zero probabilities
        self.transition_probs = transition_probs + self.epsilon
        self.transition_probs /= self.transition_probs.sum(axis=1, keepdims=True)
        
        self.emission_probs = emission_probs + self.epsilon
        self.emission_probs /= self.emission_probs.sum(axis=1, keepdims=True)
        
        self.initial_probs = initial_probs + self.epsilon
        self.initial_probs /= self.initial_probs.sum()

    def generate_sequence(self, sequence_length):
        """
        Generate a random sequence of observations and hidden states based on the HMM.
        
        Parameters
        ----------
        sequence_length : int
            Length of the sequence to generate
        
        Returns
        -------
        observations : np.ndarray, shape=(sequence_length,)
            Generated observation indices.
        states : np.ndarray, shape=(sequence_length,)
            Generated hidden state indices.
        """
        observations = np.zeros(sequence_length, dtype=int)
        states = np.zeros(sequence_length, dtype=int)

        states[0] = np.random.choice(
            len(self.initial_probs),
            p=self.initial_probs
        )
        observations[0] = np.random.choice(
            len(self.emission_probs[states[0]]),
            p=self.emission_probs[states[0]]
        )
        for t in range(1, sequence_length):
            states[t] = np.random.choice(
                len(self.transition_probs[states[t-1]]),
                p=self.transition_probs[states[t-1]]
            )
            observations[t] = np.random.choice(
                len(self.emission_probs[states[t]]),
                p=self.emission_probs[states[t]]
            )

        return observations, states

    def forward_algorithm(self, observation_sequence):
        """
        Perform the Forward algorithm (with scaling) to compute the forward probability
        matrix and scaling factors.
        
        Parameters
        ----------
        observation_sequence : list or np.ndarray
            Sequence of observation indices.
        
        Returns
        -------
        forward_probs : np.ndarray, shape=(N, T)
            Scaled forward probabilities.
        scaling_factors : np.ndarray, shape=(T,)
            Scaling factors at each time step.
        """
        num_states = self.transition_probs.shape[0]
        seq_length = len(observation_sequence)

        forward_probs = np.zeros((num_states, seq_length))
        scaling_factors = np.zeros(seq_length)


        forward_probs[:, 0] = self.initial_probs * self.emission_probs[:, observation_sequence[0]]
        scaling_factors[0] = forward_probs[:, 0].sum()
        if scaling_factors[0] == 0:
            raise ValueError("Scaling factor at t=0 is zero.")
        forward_probs[:, 0] /= scaling_factors[0]

        # Recursion
        for t in range(1, seq_length):
            forward_probs[:, t] = (
                forward_probs[:, t-1] @ self.transition_probs
            ) * self.emission_probs[:, observation_sequence[t]]
            scaling_factors[t] = forward_probs[:, t].sum()
            if scaling_factors[t] == 0:
                raise ValueError(f"Scaling factor at t={t} is zero.")
            forward_probs[:, t] /= scaling_factors[t]

        return forward_probs, scaling_factors

    def backward_algorithm(self, observation_sequence, scaling_factors):
        """
        Perform the Backward algorithm (with scaling) to compute the backward probability
        matrix.
        
        Parameters
        ----------
        observation_sequence : list or np.ndarray
            Sequence of observation indices.
        scaling_factors : np.ndarray
            Scaling factors from the forward algorithm.
        
        Returns
        -------
        backward_probs : np.ndarray, shape=(N, T)
            Scaled backward probabilities.
        """
        num_states = self.transition_probs.shape[0]
        seq_length = len(observation_sequence)

        backward_probs = np.zeros((num_states, seq_length))

        backward_probs[:, -1] = 1.0 / scaling_factors[-1]

        # Recursion
        for t in reversed(range(seq_length - 1)):
            backward_probs[:, t] = (
                self.transition_probs
                @ (self.emission_probs[:, observation_sequence[t+1]] * backward_probs[:, t+1])
            )
            backward_probs[:, t] /= scaling_factors[t]

        return backward_probs

    def compute_observation_probability(self, observation_sequence):
        """
        Compute the log probability of the observation sequence using the scaled Forward algorithm.
        
        Parameters
        ----------
        observation_sequence : list or np.ndarray
            Sequence of observation indices.
        
        Returns
        -------
        log_prob : float
            Log probability of the observation sequence.
        """
        _, scaling_factors = self.forward_algorithm(observation_sequence)
        log_prob = -np.sum(np.log(scaling_factors + self.epsilon))
        return log_prob

    def viterbi_algorithm(self, observation_sequence):
        """
        Perform the Viterbi algorithm to find the most probable state path in log-space.
        
        Parameters
        ----------
        observation_sequence : list or np.ndarray
            Sequence of observation indices.
        
        Returns
        -------
        viterbi_probs : np.ndarray, shape=(N, T)
            Log probability matrix for each state at each time step.
        pointers : np.ndarray, shape=(N, T)
            Pointers to the most probable previous state for path reconstruction.
        """
        num_states = self.transition_probs.shape[0]
        seq_length = len(observation_sequence)

        pointers = np.zeros((num_states, seq_length), dtype=int)
        viterbi_probs = np.zeros((num_states, seq_length))

        viterbi_probs[:, 0] = (
            np.log(self.initial_probs + self.epsilon)
            + np.log(self.emission_probs[:, observation_sequence[0]] + self.epsilon)
        )

        # Recursion
        for t in range(1, seq_length):
            for s in range(num_states):
                trans_probs = viterbi_probs[:, t-1] + np.log(self.transition_probs[:, s] + self.epsilon)
                pointers[s, t] = np.argmax(trans_probs)
                viterbi_probs[s, t] = (
                    np.max(trans_probs)
                    + np.log(self.emission_probs[s, observation_sequence[t]] + self.epsilon)
                )

        return viterbi_probs, pointers

    def decode_state_path(self, observation_sequence):
        """
        Decode the most probable state path using the Viterbi algorithm.
        
        Parameters
        ----------
        observation_sequence : list or np.ndarray
            Sequence of observation indices.
        
        Returns
        -------
        best_log_prob : float
            Log probability of the best path.
        best_path : list of int
            The state indices representing the most probable state path.
        """
        viterbi_probs, pointers = self.viterbi_algorithm(observation_sequence)
        # Identify the best final state
        last_state = np.argmax(viterbi_probs[:, -1])
        best_log_prob = viterbi_probs[last_state, -1]

        # Full path
        best_path = [last_state]
        for t in range(len(observation_sequence) - 1, 0, -1):
            last_state = pointers[last_state, t]
            best_path.insert(0, last_state)

        return best_log_prob, best_path

    def train_baum_welch(self, observations, convergence_threshold=0.01, max_iterations=1000):
        """
        Train the HMM using the Baum-Welch (EM) algorithm with scaled Forward/Backward computations.
        
        Parameters
        ----------
        observations : list of np.ndarray or np.ndarray
            Single or multiple observation sequences. If a single sequence is passed directly,
            it will be converted into a list of length 1.
        convergence_threshold : float, optional
            Threshold for stopping based on log-likelihood improvement (default=0.01).
        max_iterations : int, optional
            Maximum number of EM iterations (default=1000).
        
        Returns
        -------
        log_likelihoods : list of float
            Log-likelihood values for each iteration.
        """
        # Ensure we can handle multiple sequences
        if isinstance(observations, np.ndarray):
            if observations.ndim == 1:
                observations = [observations]
            else:
                observations = list(observations)
        elif isinstance(observations, list):
            if not all(isinstance(seq, (list, np.ndarray)) for seq in observations):
                observations = [observations]
        else:
            raise ValueError("Observations must be a list or numpy array.")

        num_states = self.transition_probs.shape[0]
        num_emissions = self.emission_probs.shape[1]
        log_likelihoods = []

        for iteration in range(max_iterations):
            total_log_prob = 0.0
            xi_sum = np.zeros_like(self.transition_probs)
            gamma_sum = np.zeros_like(self.emission_probs)
            gamma_initial_sum = np.zeros_like(self.initial_probs)

            for seq in observations:
                alpha, scaling_factors = self.forward_algorithm(seq)
                beta = self.backward_algorithm(seq, scaling_factors)

                T = len(seq)
                # xi: shape (N, N, T-1)
                xi = np.zeros((num_states, num_states, T - 1))

                # Compute xi for each time step (except the last, T-1)
                for t in range(T - 1):
                    denominator = np.sum(
                        alpha[:, t]
                        * self.transition_probs
                        * self.emission_probs[:, seq[t+1]]
                        * beta[:, t+1]
                    )
                    if denominator == 0:
                        print(f"Iteration {iteration}: Zero denominator at time {t}, skipping this seq.")
                        break
                    for i in range(num_states):
                        numerator = (
                            alpha[i, t]
                            * self.transition_probs[i, :]
                            * self.emission_probs[:, seq[t+1]]
                            * beta[:, t+1]
                        )
                        xi[i, :, t] = numerator / (denominator + self.epsilon)
                else:
                    gamma = np.sum(xi, axis=1)
                    # final gamma for time T-1
                    final_gamma = alpha[:, -1] * beta[:, -1]
                    final_gamma /= (np.sum(final_gamma) + self.epsilon)
                    gamma = np.hstack((gamma, final_gamma.reshape(-1, 1)))

                    # Accumulate E-steps
                    xi_sum += np.sum(xi, axis=2)
                    gamma_initial_sum += gamma[:, 0]

                    for state in range(num_states):
                        for symbol in range(num_emissions):
                            mask = (seq == symbol)
                            gamma_sum[state, symbol] += np.sum(gamma[state, mask])

                    # Accumulate the NEGATIVE log-likelihood from this sequence
                    log_prob_seq = -np.sum(np.log(scaling_factors + self.epsilon))
                    total_log_prob += log_prob_seq

            # M-step: update initial, transition, and emission probabilities
            new_initial_probs = gamma_initial_sum / (np.sum(gamma_initial_sum) + self.epsilon)

            new_transition_probs = xi_sum / (
                np.sum(xi_sum, axis=1, keepdims=True) + self.epsilon
            )

            new_emission_probs = gamma_sum / (
                np.sum(gamma_sum, axis=1, keepdims=True) + self.epsilon
            )

            # Smoothing
            new_initial_probs += self.epsilon
            new_initial_probs /= new_initial_probs.sum()

            new_transition_probs += self.epsilon
            new_transition_probs /= new_transition_probs.sum(axis=1, keepdims=True)

            new_emission_probs += self.epsilon
            new_emission_probs /= new_emission_probs.sum(axis=1, keepdims=True)

            # Store and print iteration log-likelihood
            log_likelihoods.append(total_log_prob)
            print(f"Iteration {iteration + 1}: Log-Likelihood = {total_log_prob:.6f}")

            # Check for convergence
            if iteration > 0:
                improvement = abs(log_likelihoods[-1] - log_likelihoods[-2])
                if improvement < convergence_threshold:
                    print(f"Converged at iteration {iteration + 1}.")
                    break

            # Update parameters
            self.initial_probs = new_initial_probs
            self.transition_probs = new_transition_probs
            self.emission_probs = new_emission_probs

        else:
            print("Baum-Welch did not converge within the maximum number of iterations.")

        return log_likelihoods


