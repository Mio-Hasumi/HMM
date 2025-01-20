import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Set a font that supports English and other characters, adjust as needed
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False  # Ensure that minus signs are displayed correctly

def plot_forward_backward(forward_probs, backward_probs, scaling_factors=None):
    """
    Visualize the Forward and Backward matrices.
    
    Parameters
    ----------
    forward_probs : numpy.ndarray
        Scaled forward matrix of shape (N, T)
    backward_probs : numpy.ndarray
        Scaled backward matrix of shape (N, T)
    scaling_factors : numpy.ndarray, optional
        Scaling factors for each time step of shape (T,)
    """
    num_states, seq_length = forward_probs.shape

    # Forward Probability Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(forward_probs, annot=True, fmt=".4f", cmap="viridis",
                xticklabels=range(1, seq_length + 1),
                yticklabels=[f"State {i}" for i in range(num_states)])
    plt.title("Forward Probability Matrix")
    plt.xlabel("Time Step")
    plt.ylabel("State")
    plt.show()

    # Backward Probability Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(backward_probs, annot=True, fmt=".4f", cmap="viridis",
                xticklabels=range(1, seq_length + 1),
                yticklabels=[f"State {i}" for i in range(num_states)])
    plt.title("Backward Probability Matrix")
    plt.xlabel("Time Step")
    plt.ylabel("State")
    plt.show()

    if scaling_factors is not None:
        # Scaling Factors Line Plot
        plt.figure(figsize=(12, 4))
        plt.plot(range(1, seq_length + 1), scaling_factors, marker='o')
        plt.title("Scaling Factors")
        plt.xlabel("Time Step")
        plt.ylabel("Scaling Factor")
        plt.grid(True)
        plt.show()

def plot_transition_emission(transition_probs, emission_probs):
    """
    Visualize the Transition and Emission probability matrices.
    
    Parameters
    ----------
    transition_probs : numpy.ndarray
        Transition probability matrix of shape (N, N)
    emission_probs : numpy.ndarray
        Emission probability matrix of shape (N, M)
    """
    num_states = transition_probs.shape[0]
    num_symbols = emission_probs.shape[1]

    # Transition Probability Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_probs, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[f"State {i}" for i in range(num_states)],
                yticklabels=[f"State {i}" for i in range(num_states)])
    plt.title("Transition Probability Matrix")
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.show()

    # Emission Probability Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(emission_probs, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=[f"Obs {i}" for i in range(num_symbols)],
                yticklabels=[f"State {i}" for i in range(num_states)])
    plt.title("Emission Probability Matrix")
    plt.xlabel("Observation Symbol")
    plt.ylabel("State")
    plt.show()

def plot_viterbi_path(observations, best_path):
    """
    Visualize the observation sequence and the corresponding Viterbi path.
    
    Parameters
    ----------
    observations : list or numpy.ndarray
        Observation sequence
    best_path : list
        Most probable hidden state path
    """
    seq_length = len(observations)
    time_steps = range(seq_length)

    plt.figure(figsize=(14, 6))

    # Observation Sequence
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, observations, marker='o', linestyle='-', color='blue')
    plt.title("Observation Sequence")
    plt.xlabel("Time Step")
    plt.ylabel("Observation Symbol")
    plt.grid(True)

    # Viterbi Path
    plt.subplot(2, 1, 2)
    plt.step(time_steps, best_path, where='mid', color='red', linewidth=2)
    plt.title("Viterbi Most Probable State Path")
    plt.xlabel("Time Step")
    plt.ylabel("Hidden State")
    plt.yticks([0, 1], ['State 0', 'State 1'])
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_log_likelihood(log_likelihoods):
    """
    Visualize the progression of log-likelihoods during Baum-Welch training.
    
    Parameters
    ----------
    log_likelihoods : list
        List of log-likelihoods for each iteration
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(log_likelihoods) + 1), log_likelihoods, marker='o')
    plt.title("Log-Likelihood Progression During Baum-Welch Training")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid(True)
    plt.show()
