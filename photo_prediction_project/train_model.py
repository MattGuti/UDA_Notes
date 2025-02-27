import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load extracted features
features = pd.read_csv("features.csv")

# Unsupervised Clustering (K-Means)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(features)
features['Cluster'] = clusters

# Visualize clusters
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=clusters, cmap='viridis')
plt.title("Clustering of Images")
plt.show()

# Create labels (Assume first 50 images are featured)
labels = np.array([1 if i < 50 else 0 for i in range(len(features))])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features.drop(columns=['Cluster']), labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')