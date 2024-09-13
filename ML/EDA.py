import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def printData(data):
    print(data.columns)


def perform_eda(data):
    # Check for missing values
    print("Missing values in each column:")
    print(data.isnull().sum())
    print("\n")

    # Descriptive statistics
    print("Descriptive Statistics:")
    print(data.describe())
    print("\n")

    # Correlation matrix
    corr_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.show()

    # Scatter plot example: MQ3 vs MQ8 colored by Target
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="MQ3", y="MQ8", hue="Target", data=data, palette="Set1")
    plt.title("Scatter Plot of MQ3 vs MQ8")
    plt.show()

    # Pairplot with hue
    sns.pairplot(data, hue="Target", palette="Set1")
    plt.title("Pairplot of Gas Sensors with Wine Types")
    plt.show()

    # Box plot for each feature by Target
    features = ["MQ3", "MQ135", "MQ8", "MQ5", "MQ7", "mq4", "mq6", "MQ2", "MQ9"]
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features):
        plt.subplot(3, 3, i + 1)
        sns.boxplot(x="Target", y=feature, data=data, palette="Set1")
        plt.title(f"Box Plot of {feature} by Wine Type")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Histograms for each feature
    plt.figure(figsize=(14, 10))
    data[features].hist(bins=30, edgecolor="k", figsize=(14, 10))
    plt.suptitle("Histograms of Sensor Readings")
    plt.show()

    # Correlation with Target
    target_corr = data.corr()["Target"].sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    target_corr.drop("Target").plot(kind="bar")
    plt.title("Correlation of Features with Target")
    plt.show()

    # Optional: Dimensionality Reduction with PCA
    scaled_features = StandardScaler().fit_transform(data[features])
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
    pca_df["Target"] = data["Target"]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="PC1", y="PC2", hue="Target", data=pca_df, palette="Set1")
    plt.title("PCA of Gas Sensor Data")
    plt.show()


# Load the dataset
data = pd.read_csv("ML/MQSensorData_updated.csv", header=0)

# Perform EDA
perform_eda(data)
