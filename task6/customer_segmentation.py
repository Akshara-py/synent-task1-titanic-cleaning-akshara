import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

print("Loading Dataset...")

# Read dataset
file_path = os.path.join(
    os.path.dirname(__file__),
    "Mall_Customers.csv"
)

df = pd.read_csv(file_path)

print("\nFirst 5 Rows:")
print(df.head())

# Select features
X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]]

# Elbow Method

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)

    wcss.append(
        kmeans.inertia_
    )

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.savefig("elbow_method.png")
plt.close()

# Final KMeans Model

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

# Cluster Visualization

plt.figure(figsize=(8,6))

for cluster in range(5):

    cluster_data = df[
        df["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        label=f"Cluster {cluster}"
    )

# Centroids

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=250,
    marker="X",
    label="Centroids"
)

plt.title(
    "Customer Segmentation using K-Means"
)

plt.xlabel(
    "Annual Income (k$)"
)

plt.ylabel(
    "Spending Score (1-100)"
)

plt.legend()

plt.savefig(
    "customer_segments.png"
)

plt.close()

# Save Output

df.to_csv(
    "customer_clusters.csv",
    index=False
)

print("\nCustomer Segmentation Completed Successfully!")