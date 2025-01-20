import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_transition_emission(transition_probs, emission_probs, states=None, observations=None):
    """
    Plots the transition and emission probability matrices as separate heatmaps.
    
    Args:
        transition_probs (np.ndarray): Transition probability matrix.
        emission_probs (np.ndarray): Emission probability matrix.
        states (list, optional): List of state names. Defaults to ['State 0', 'State 1', ...].
        observations (list, optional): List of observation names. Defaults to ['Obs 0', 'Obs 1', ...].
    """
    if states is None:
        states = [f"State {i}" for i in range(transition_probs.shape[0])]
    if observations is None:
        observations = [f"Obs {i}" for i in range(emission_probs.shape[1])]
    
    # Plot Transition Probabilities
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        transition_probs,
        annot=True,
        cmap='Blues',
        fmt=".2f",                
        annot_kws={"size": 8},    
        xticklabels=states,
        yticklabels=states
    )
    plt.title('Transition Probabilities')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.tight_layout()
    plt.show()
    
    # Plot Emission Probabilities
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        emission_probs,
        annot=True,
        cmap='Greens',
        fmt=".2f",
        annot_kws={"size": 8},
        xticklabels=observations,
        yticklabels=states
    )
    plt.title('Emission Probabilities')
    plt.xlabel('Observations')
    plt.ylabel('States')
    plt.tight_layout()
    plt.show()

def plot_forward_backward(forward_probs, backward_probs, scaling_factors, states=None):
    """
    Plots the forward and backward probabilities as separate heatmaps, plus a line plot of scaling factors.
    
    Args:
        forward_probs (np.ndarray): Forward probability matrix.
        backward_probs (np.ndarray): Backward probability matrix.
        scaling_factors (np.ndarray): Scaling factors used in computations.
        states (list, optional): List of state names. Defaults to ['State 0', 'State 1', ...].
    """
    if states is None:
        states = [f"State {i}" for i in range(forward_probs.shape[0])]
    
    # Plot Forward Probabilities
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        forward_probs,
        annot=True,
        cmap='Purples',
        fmt=".2e",                
        annot_kws={"size": 8},
        yticklabels=states,
        xticklabels=False
    )
    plt.title('Forward Probabilities')
    plt.xlabel('Time Step')
    plt.ylabel('States')
    plt.tight_layout()
    plt.show()
    
    # Plot Backward Probabilities
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        backward_probs,
        annot=True,
        cmap='Oranges',
        fmt=".2e",
        annot_kws={"size": 8},
        yticklabels=states,
        xticklabels=False
    )
    plt.title('Backward Probabilities')
    plt.xlabel('Time Step')
    plt.ylabel('States')
    plt.tight_layout()
    plt.show()
    
    # Plot Scaling Factors
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(scaling_factors) + 1), scaling_factors, marker='o', linestyle='-')
    plt.title('Scaling Factors Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Scaling Factor')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_viterbi_path(observations_seq, best_path, states=None, observations=None):
    """
    Plots the Viterbi path alongside the observations.
    
    Args:
        observations_seq (list): List of observation indices.
        best_path (list): List of state indices representing the Viterbi path.
        states (list, optional): List of state names. Defaults to ['State 0', 'State 1', ...].
        observations (list, optional): List of observation names. Defaults to ['Obs 0', 'Obs 1', ...].
    """
    if states is None:
        num_states = max(best_path) + 1
        states = [f"State {i}" for i in range(num_states)]
    if observations is None:
        num_obs = max(observations_seq) + 1
        observations = [f"Obs {i}" for i in range(num_obs)]
    
    obs_labels = [observations[obs] for obs in observations_seq]
    state_labels = [states[state] for state in best_path]

    plt.figure(figsize=(15, 4))
    plt.plot(range(len(obs_labels)), best_path, marker='o', color='blue', label='Viterbi Path')
    plt.xticks(range(len(obs_labels)), obs_labels)
    plt.yticks(range(len(states)), states)
    plt.xlabel('Time Step')
    plt.ylabel('States')
    plt.title('Viterbi Path')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_log_likelihood(log_likelihoods):
    """
    Plots the log-likelihood progression over training iterations.
    
    Args:
        log_likelihoods (list): List of log-likelihood values per iteration.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(log_likelihoods) + 1), log_likelihoods, marker='o', linestyle='-', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood Progression during Baum-Welch Training')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
