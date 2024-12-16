import pandas as pd
import numpy as np
import os
import joblib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load CSV files
def load_csv(file_path):
    logging.info(f"Loading data from {file_path}...")
    return pd.read_csv(file_path)

# Function for feature scaling
def scale_features(features, method):
    logging.info(f"Scaling features using {method} scaling...")
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaling method. Choose 'standard', 'minmax', or 'robust'.")
    return scaler.fit_transform(features), scaler

# Function for PCA
def apply_pca(features, n_components=2):
    logging.info(f"Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    explained_variance = pca.explained_variance_ratio_
    return reduced_features, explained_variance, pca

# Function to train clustering models
def train_clustering(features, n_clusters, method):
    logging.info(f"Training {method.upper()} clustering with {n_clusters} clusters...")
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', max_iter=500, n_init=20)
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    else:
        raise ValueError("Invalid clustering method. Choose 'kmeans' or 'gmm'.")
    labels = model.fit_predict(features)
    return model, labels

# Function to evaluate clustering performance
def evaluate_clustering(features, labels):
    logging.info("Evaluating clustering performance...")
    return {
        "Silhouette Score": silhouette_score(features, labels),
        "Calinski-Harabasz Score": calinski_harabasz_score(features, labels),
        "Davies-Bouldin Score": davies_bouldin_score(features, labels),
    }

# Function to analyze two conditions and save the best model
def analyze_two_conditions(file_path1, file_path2, output_folder="Results", n_clusters=2):
    # Load data
    data1 = load_csv(file_path1)
    data2 = load_csv(file_path2)
    
    # Combine data and assign labels
    logging.info("Combining datasets and assigning condition labels...")
    data1['Condition'] = 0  # Label for the first condition
    data2['Condition'] = 1  # Label for the second condition
    combined_data = pd.concat([data1, data2], ignore_index=True)
    true_labels = combined_data['Condition'].values
    features = combined_data.drop(columns=['Condition'])
    
    # Define scaling methods and clustering methods
    scaling_methods = ['standard', 'minmax', 'robust']
    clustering_methods = ['kmeans', 'gmm']
    
    # Prepare results folder
    os.makedirs(output_folder, exist_ok=True)
    logging.info(f"Results will be saved to {output_folder}")
    
    best_model = None
    best_scaler = None
    best_metrics = None
    best_score = -np.inf  # To track the highest silhouette score
    best_model_details = {}
    
    for scaling_method in scaling_methods:
        scaled_features, scaler = scale_features(features, method=scaling_method)
        reduced_features, explained_variance, pca = apply_pca(scaled_features, n_components=2)
        logging.info(f"PCA Explained Variance for {scaling_method}: {explained_variance}")

        for clustering_method in clustering_methods:
            model, labels = train_clustering(reduced_features, n_clusters, method=clustering_method)
            
            metrics = evaluate_clustering(reduced_features, labels)
            metrics['Scaling Method'] = scaling_method
            metrics['Clustering Method'] = clustering_method
            metrics['Explained Variance'] = explained_variance.tolist()
            
            logging.info(f"Metrics for {scaling_method} with {clustering_method}: {metrics}")
            
            # Update best model if current model has a better Silhouette Score
            if metrics['Silhouette Score'] > best_score:
                best_score = metrics['Silhouette Score']
                best_model = model
                best_scaler = scaler
                best_metrics = metrics
                best_model_details = {
                    "Scaling Method": scaling_method,
                    "Clustering Method": clustering_method,
                    "Explained Variance": explained_variance.tolist()
                }
    
    # Save the best model and scaler
    model_file = os.path.join(output_folder, "best_model.pkl")
    scaler_file = os.path.join(output_folder, "best_scaler.pkl")
    joblib.dump(best_model, model_file)
    joblib.dump(best_scaler, scaler_file)
    
    # Save metrics and details for the best model
    best_metrics_file = os.path.join(output_folder, "best_model_metrics.csv")
    pd.DataFrame([best_metrics]).to_csv(best_metrics_file, index=False)
    
    logging.info(f"Best model saved to {model_file}")
    logging.info(f"Best scaler saved to {scaler_file}")
    logging.info(f"Best model metrics saved to {best_metrics_file}")
    
    # Display results
    print("\n=== Best Model Details ===")
    print(f"Scaling Method: {best_model_details['Scaling Method']}")
    print(f"Clustering Method: {best_model_details['Clustering Method']}")
    print(f"Silhouette Score: {best_metrics['Silhouette Score']}")
    print(f"Calinski-Harabasz Score: {best_metrics['Calinski-Harabasz Score']}")
    print(f"Davies-Bouldin Score: {best_metrics['Davies-Bouldin Score']}")
    print(f"Explained Variance: {best_model_details['Explained Variance']}")
    
    return model_file, scaler_file, best_metrics_file

# Main execution
if __name__ == "__main__":
    # Example file paths
    file_path1 = r"C:\Users\siris\Projects\Machine Fault Detection and Monitoring System\MFDMS\normal_time_domain_features_reshaped.csv"
    file_path2 = r"C:\Users\siris\Projects\Machine Fault Detection and Monitoring System\MFDMS\unbalance_time_domain_features_reshaped.csv"
    output_folder = "Results"
    
    model_file, scaler_file, metrics_file = analyze_two_conditions(file_path1, file_path2, output_folder=output_folder)
    
    print(f"\nBest model saved to: {model_file}")
    print(f"Best scaler saved to: {scaler_file}")
    print(f"Best model metrics saved to: {metrics_file}")
