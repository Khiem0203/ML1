import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def explore_and_cluster(data):
    plt.figure(figsize=(8, 4))
    sns.histplot(data['Age'], bins=30, kde=True)
    plt.title("Distribution of User Age")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data['Rating'])
    plt.title("Boxplot of Book Ratings")
    plt.tight_layout()
    plt.show()

    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['Age', 'Rating']])
    sns.scatterplot(data=data, x='Age', y='Rating', hue='Cluster', palette='tab10')
    plt.title("User Clustering by Age & Rating")
    plt.tight_layout()
    plt.show()

    return data

if __name__ == "__main__":
    data = pd.read_csv("data_merged.csv")
    clustered_data = explore_and_cluster(data)
