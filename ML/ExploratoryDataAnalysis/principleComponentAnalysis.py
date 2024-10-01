import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Step 1: Load and preprocess the data
def load_and_preprocess(file_path):
    # Load CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Drop non-numeric columns (timestamp and Target label)
    data = data.drop(
        columns=["yyyy-mm-dd timestamp", "Target"]
    )  # Adjust the column names based on actual CSV

    # Ensure all the relevant columns are numeric
    data = data.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values after conversion
    data = data.dropna()

    return data


# Step 2: Standardize the data (PCA requires data to be scaled)
def standardize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


# Step 3: Perform PCA
def perform_pca(scaled_data, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by components: {explained_variance}")

    return pca_result, explained_variance


# Step 4: Visualize the PCA results
def plot_pca(pca_result):
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c="blue", edgecolor="k", s=50)
    plt.title("PCA of Sensor Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


# Main execution
file_path = "ML/WineCSVs/Train/SixWinesData/combined_cleaned_r1_3009_data.csv"  # Replace with the path to your file

# Load and preprocess data
data = load_and_preprocess(file_path)

# Standardize the data
scaled_data = standardize_data(data)

# Perform PCA (reduce to 2 components for easy visualization)
pca_result, explained_variance = perform_pca(scaled_data, n_components=2)

# Plot PCA results
plot_pca(pca_result)
