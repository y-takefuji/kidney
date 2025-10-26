import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import MeanShift, estimate_bandwidth
import xgboost as xgb
import time

# Function to map cluster labels to binary predictions
def map_clusters_to_binary(cluster_labels, y_true):
    """
    Maps each cluster to the majority class within that cluster.
    
    Parameters:
    - cluster_labels: Cluster assignments from Mean-Shift
    - y_true: True labels used to determine majority class in each cluster
    
    Returns:
    - y_pred: Binary predictions based on majority class in each cluster
    """
    # Find unique cluster labels
    unique_clusters = np.unique(cluster_labels)
    
    # For each cluster, find the majority class
    cluster_to_class = {}
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        if np.sum(mask) > 0:
            # Find majority class in this cluster
            cluster_majority = np.argmax(np.bincount(y_true[mask]))
            cluster_to_class[cluster] = cluster_majority
            
            # Print information about this cluster
            cluster_size = np.sum(mask)
            class_0_count = np.sum(y_true[mask] == 0)
            class_1_count = np.sum(y_true[mask] == 1)
            majority_pct = max(class_0_count, class_1_count) / cluster_size * 100
            
    # Map each point's cluster to its predicted class
    y_pred = np.zeros_like(cluster_labels)
    for i, cluster in enumerate(cluster_labels):
        y_pred[i] = cluster_to_class[cluster]
    
    return y_pred

# Record start time
start_time = time.time()

# Load the data
print("Loading data...")
data = pd.read_csv('Chronic_Kidney_Disease_data.csv')

# Drop 'DoctorInCharge' and 'PatientID' columns if they exist
data = data.drop(['DoctorInCharge', 'PatientID'], axis=1)

# Separate features and target
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Print class distribution
class_counts = y.value_counts()
print("\nClass distribution in Diagnosis:")
print(f"Negative (0): {class_counts.get(0, 0)} samples")
print(f"Positive (1): {class_counts.get(1, 0)} samples")
print(f"Total: {len(y)} samples")
print(f"Class imbalance: {class_counts.get(1, 0)/len(y):.2f} (proportion of positive cases)")

print(f"\nDataset shape: {X.shape}")

# Convert to numpy arrays
X_array = X.values if isinstance(X, pd.DataFrame) else X
y_array = y.values if isinstance(y, pd.Series) else y

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 1. Random Forest with 5-fold CV
print("\n1. Running Random Forest with 5-fold CV...")
rf_model = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')
print(f"Random Forest 5-fold CV Accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
print(f"Individual fold scores: {[f'{score:.4f}' for score in rf_scores]}")

# 2. XGBoost with 5-fold CV
print("\n2. Running XGBoost with 5-fold CV...")
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_scores = cross_val_score(xgb_model, X, y, cv=kf, scoring='accuracy')
print(f"XGBoost 5-fold CV Accuracy: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
print(f"Individual fold scores: {[f'{score:.4f}' for score in xgb_scores]}")

# 3. Mean-Shift with 5-fold CV
print("\n3. Running Mean-Shift with 5-fold CV...")
meanshift_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_array)):
    X_test = X_array[test_idx]
    y_test = y_array[test_idx]
    
    print(f"\n--- Fold {fold_idx+1} ---")
    print(f"Test set: {len(y_test)} samples")
    print(f"Class distribution in test set: {np.sum(y_test==0)} class 0, {np.sum(y_test==1)} class 1")
    
    # Calculate bandwidth based on data variance
    bandwidth = np.mean(np.std(X_test, axis=0)) * 0.5
    print(f"Using bandwidth: {bandwidth:.4f}")
    
    # Apply Mean-Shift clustering
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    cluster_labels = meanshift.fit_predict(X_test)
    
    # Count clusters
    n_clusters = len(np.unique(cluster_labels))
    print(f"Number of clusters found: {n_clusters}")
    
    # Map clusters to binary predictions
    y_pred = map_clusters_to_binary(cluster_labels, y_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    meanshift_scores.append(acc)
    
    print(f"Fold {fold_idx+1} accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Calculate mean and std of accuracies
mean_acc = np.mean(meanshift_scores)
std_acc = np.std(meanshift_scores)
print(f"\nMean-Shift 5-fold CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Individual fold scores: {[f'{score:.4f}' for score in meanshift_scores]}")

# 4. Mean-Shift on full dataset
print("\n4. Running Mean-Shift on full dataset...")

# Calculate bandwidth for full dataset
bandwidth_full = np.mean(np.std(X_array, axis=0)) * 0.5
print(f"Using bandwidth: {bandwidth_full:.4f}")

# Apply Mean-Shift clustering on full dataset
meanshift_full = MeanShift(bandwidth=bandwidth_full, bin_seeding=True)
cluster_labels_full = meanshift_full.fit_predict(X_array)

# Count clusters in full dataset
n_clusters_full = len(np.unique(cluster_labels_full))
print(f"Number of clusters found: {n_clusters_full}")

# Map clusters to binary predictions for full dataset
y_pred_full = map_clusters_to_binary(cluster_labels_full, y_array)

# Calculate final accuracy
final_acc = accuracy_score(y_array, y_pred_full)
print(f"Full dataset accuracy: {final_acc:.4f}")

# Print confusion matrix and classification report
print("\nFull Dataset Confusion Matrix:")
print(confusion_matrix(y_array, y_pred_full))
print("\nClassification Report:")
print(classification_report(y_array, y_pred_full))

# Show cluster distribution
#cluster_counts = pd.Series(cluster_labels_full).value_counts().sort_index()
#print("\nCluster distribution:")
#for cluster, count in cluster_counts.items():
#    print(f"Cluster {cluster}: {count} samples")

# Results Summary
print("\n--- Results Summary ---")
print(f"Random Forest:   {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
print(f"XGBoost:        {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")
print(f"Mean-Shift CV:  {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Mean-Shift Full: {final_acc:.4f}")

# Print execution time
execution_time = time.time() - start_time
print(f"\nExecution time: {execution_time:.2f} seconds")
