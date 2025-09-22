import os
import subprocess
import re
import numpy as np

# Parameters for the main script
params = {
    'device': '3',
    'hidden_dim': 128,
    'lr': 5e-4,
    'budget': 5,
    'max_episodes': 10000,
    'max_epsilon': 1.0,
    'min_epsilon': 0.01,
    'epsilon_decay': 1/1600,
    'batch_size': 32,
    'max_grad_norm': 1.0,
    'update_target_iters': 1000,
    'buffer_size': 5000,
    'beta': 0.4,
    'alpha': 0.6,
    'verbose': True
}

# List of seeds to run the script with
seeds = [0, 1, 2, 3, 4]

eigenvalues = []
losses = []
neighbors = []

# Check if the DQN.py file exists
script_path = 'DQN.py'
if not os.path.exists(script_path):
    print(f"Error: {script_path} not found.")
    exit(1)

# Run the main script multiple times with different seeds
for seed in seeds:
    args = [f'--{key}={value}' for key, value in params.items() if key != 'verbose']
    if params['verbose']:
        args.append('--verbose')
    args.append(f'--seed={seed}')
    command = ['python', script_path] + args
    print(f"Running command: {' '.join(command)}")  # Debug print

    result = subprocess.run(command, capture_output=True, text=True)
    
    # Capture and print the output and errors for debugging
    output = result.stdout
    error = result.stderr
    print(f"Output for seed {seed}:\n{output}")
    if error:
        print(f"Error for seed {seed}:\n{error}")

    # Extract the final eigenvalue, loss, and neighbors from the output
    eigenvalue_match = re.search(r"Final eigenvalue: \[([\d.]+)\]", output)
    loss_match = re.search(r"Final loss: ([\d.]+)", output)
    neighbor_match = re.search(r"Final neighbors: ([\d.]+)", output)

    if eigenvalue_match and loss_match and neighbor_match:
        eigenvalue = float(eigenvalue_match.group(1))
        loss = float(loss_match.group(1))
        neighbor = float(neighbor_match.group(1))
        
        eigenvalues.append(eigenvalue)
        losses.append(loss)
        neighbors.append(neighbor)
    else:
        print(f"Error: Output did not match expected format for seed {seed}")

# Compute the mean and standard deviation if there are valid results
if eigenvalues and losses and neighbors:
    mean_eigenvalue = np.mean(eigenvalues)
    std_eigenvalue = np.std(eigenvalues)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    mean_neighbors = np.mean(neighbors)
    std_neighbors = np.std(neighbors)

    print(f"Mean eigenvalue: {mean_eigenvalue:.4f} ± {std_eigenvalue:.4f}")
    print(f"Mean loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"Mean neighbors: {mean_neighbors:.4f} ± {std_neighbors:.4f}")
else:
    print("Error: No valid results were obtained.")
