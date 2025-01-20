# HMM
This is First-Order Hidden Markov Model (HMM), implemented in Python and includes visualization tools.

This project provides a Python implementation of a First-Order Hidden Markov Model (HMM) along with utilities for visualization and training. The implementation supports key algorithms such as Forward, Backward, Viterbi, and Baum-Welch, and includes tools to visualize model parameters, probabilities, and training progression. It only has one sequence but can be easily revised to handle the multiple

Features

Hidden Markov Model (HMM) Implementation:

Configurable transition, emission, and initial probability matrices.

Scalable for various hidden states and observation symbols.

Algorithms:

Forward Algorithm: Computes scaled probabilities of observation sequences.

Backward Algorithm: Computes backward probabilities with scaling.

Viterbi Algorithm: Decodes the most probable hidden state sequence.

Baum-Welch Algorithm: Trains the HMM parameters using observation sequences.

Synthetic Data Generation:

Simulates observation and hidden state sequences for testing and experimentation.

Visualizations:

Heatmaps for Forward, Backward, Transition, and Emission matrices.

Line plots for Log-Likelihood progression and Scaling factors during training.

Viterbi decoded path visualization.

File Structure

HMM_Project/
├── hmm.py           # Implementation of the HMM class
├── visualize.py     # Visualization utilities for matrices and training progress
├── test_hmm.py      # Example script demonstrating HMM functionality and visualizations
Installation

Clone the repository:

git clone https://github.com/Mio-Hasumi/HMM 
cd hmm-project

Install dependencies:

pip install numpy
pip install matplotlib
pip install seaborn

Usage

Run the Sample Test Script/Use

The test_hmm.py script demonstrates the following:

Generating a synthetic observation and hidden state sequence.

Calculating Forward and Backward probabilities.

Computing the log-probability of the observation sequence.

Decoding the most probable hidden state path using Viterbi.

Training the HMM with Baum-Welch.

Visualizing model parameters and results.
