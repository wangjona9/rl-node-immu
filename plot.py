import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('shield_iter_shield_value.csv')

# Extract the ShieldValue column
shield_values = df['ShieldValue']

# Plotting the distribution of ShieldValue
plt.figure(figsize=(10, 6))
plt.hist(shield_values, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of ShieldValue')
plt.xlabel('ShieldValue')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('shield_value_distribution.png')
plt.close()


# Read the CSV file
df = pd.read_csv('SV_iter_shield_value.csv')

# Extract the ShieldValue column
shield_values = df['ShieldValue']

# Plotting the distribution of ShieldValue
plt.figure(figsize=(10, 6))
plt.hist(shield_values, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of SV')
plt.xlabel('ShieldValue')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('SV_distribution.png')
plt.close()