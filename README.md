# HMM
This is First-Order Hidden Markov Model (HMM), implemented in Python and includes visualization tools.

This project provides a Python implementation of a First-Order Hidden Markov Model (HMM) along with utilities for visualization and training. The implementation supports key algorithms such as Forward, Backward, Viterbi, and Baum-Welch, and includes tools to visualize model parameters, probabilities, and training progression.

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

git clone https://github.com/your-username/hmm-project.git
cd hmm-project

Install dependencies:

pip install numpy
pip install matplotlib
pip install seaborn

Usage

Run the Test Script

The test_hmm.py script demonstrates the following:

Generating a synthetic observation and hidden state sequence.

Calculating Forward and Backward probabilities.

Computing the log-probability of the observation sequence.

Decoding the most probable hidden state path using Viterbi.

Training the HMM with Baum-Welch.

Visualizing model parameters and results.

Run the script:

python test_hmm.py

Key Outputs

Console Outputs:

Generated observation and hidden state sequences.

Forward/Backward probability matrices and scaling factors.

Log probabilities of observation sequences.

Viterbi decoded paths.

Log-likelihood progression during Baum-Welch training.

Visualizations:

Forward and Backward probability matrices (heatmaps).

Transition and Emission probability matrices (heatmaps).

Log-likelihood progression during Baum-Welch training (line plot).

Viterbi decoded path (state path vs. observations).

Example Output

Console Output

Generated Observations : [1 2 1 0 0 2 1 1 1 0]
Generated Hidden States: [1 0 0 1 1 0 0 0 0 1]

--- Forward Algorithm ---
Forward Probability Matrix:
 [[0.6667 0.8824 0.7255 ...]
 [0.3333 0.1176 0.2745 ...]]
Scaling Factors: [0.36 0.34 0.366 ...]

--- Backward Algorithm ---
Backward Probability Matrix:
 [[3.095 2.914 ...]
 [2.142 3.138 ...]]

--- Observation Probability ---
Log Probability of the sequence = -10.8837
...

Visualizations

Transition and Emission Matrices



Log-Likelihood Progression



Dependencies

Python 3.x

Numpy: Efficient array operations

Matplotlib: Visualization

Seaborn: Enhanced heatmap and plotting aesthetics

Install via:

pip install -r requirements.txt

License

This project is licensed under the MIT License. See the LICENSE file for details.

Future Enhancements

Support for higher-order HMMs.

Integration of continuous observation spaces (e.g., Gaussian emissions).

Batch training on multiple sequences.

Advanced visualization tools with interactivity.


