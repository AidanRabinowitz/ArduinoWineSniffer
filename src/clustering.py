import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt


class ClusteringTechniques:
    def __init__(self, df):
        """
        Initialize the class with the dataset, dropping timestamp but separating target.
        :param df: The input DataFrame for clustering.
        """
        self.df = df
        self.features = self.df.drop(columns=['Target']).values
        # Keeping the Target column for evaluation
        self.target = self.df['Target']

    def kmeans(self, n_clusters=3, random_state=42):
        """
        Apply K-Means clustering.
        :param n_clusters: Number of clusters to form.
        :param random_state: Random state for reproducibility.
        :return: K-Means labels for each point.
        """
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans_labels = kmeans_model.fit_predict(self.features)
        return kmeans_labels

    def dbscan(self, eps=0.5, min_samples=5):
        """
        Apply DBSCAN clustering.
        :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
        :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        :return: DBSCAN labels for each point.
        """
        dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan_model.fit_predict(self.features)
        return dbscan_labels

    def agglomerative(self, n_clusters=3, linkage='ward'):
        """
        Apply Agglomerative (Hierarchical) clustering.
        :param n_clusters: The number of clusters to find.
        :param linkage: Linkage criterion to use ('ward', 'complete', 'average', 'single').
        :return: Agglomerative clustering labels for each point.
        """
        agglomerative_model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage)
        agglomerative_labels = agglomerative_model.fit_predict(self.features)
        return agglomerative_labels

    def visualize(self, labels, title="Clustering Results"):
        """
        Visualize the clustering result using a 2D scatter plot.
        :param labels: Cluster labels for each data point.
        :param title: The title of the plot.
        """
        plt.scatter(self.features[:, 0],
                    self.features[:, 1], c=labels, cmap='viridis')
        plt.title(title)
        plt.show()

    def evaluate_with_target(self, labels):
        """
        Compare clustering results with the actual Target column (wine labels).
        :param labels: Cluster labels produced by a clustering algorithm.
        :return: A DataFrame showing the predicted cluster vs the actual wine label.
        """
        results = pd.DataFrame({'Cluster': labels, 'Wine': self.target})
        return results

# Example usage with a DataFrame:
# df = pd.read_csv("your_dataset.csv")  # Assuming you have the DataFrame
# clustering = Cl
