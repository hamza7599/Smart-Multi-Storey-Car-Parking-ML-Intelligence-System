 Smart Multi-Storey Car Parking ML Intelligence System

An end-to-end Machine Learning pipeline and interactive web application built to optimize multi-storey parking facility operations using relational SQL integration, unsupervised clustering, supervised classification, and real-time deployment.

📂 Project Structure

├── parking_system.db          # SQLite database containing floors, guards, and cars tables
├── parking_ml_pipeline.joblib  # Serialized model pipeline (Scaler + Best Classifier)
├── train_model.py             # End-to-end data generation, SQL, clustering, and training script
├── app.py                     # Streamlit web application frontend
└── README.md                  # Project documentation


⚙️ Project Requirements Covered

Dataset Creation & SQL Integration: Relational database setup (floors, guards, cars with 2,000 synthetic records) combined via SQL JOIN queries.

Data Preprocessing: Missing value imputation, One-Hot Encoding for categorical fields, and StandardScaler normalization.

Unsupervised Learning (Clustering): Applied K-Means and DBSCAN clustering to partition parking behavioral segments, optimized using Silhouette scores.

Supervised Learning (Classification): Treated cluster labels as targets, training Random Forest and Gradient Boosting classifiers with GridSearchCV hyperparameter tuning.

Model Evaluation: Evaluated via Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

Model Serialization: Saved, reloaded, and verified pipeline consistency using joblib.

Input Schema & Validation: Robust payload validation using Pydantic schemas.

Model Card: Comprehensive technical documentation covering intended use, data profile, and limitations.

Streamlit Deployment Frontend: Interactive UI for real-time inference and parking segment prediction.

🚀 Getting Started Locally

1. Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries:

pip install pandas numpy scikit-learn pydantic streamlit joblib


2. Train and Generate Pipeline Artifacts

Run your training script to create the SQLite database (parking_system.db) and serialize the machine learning pipeline (parking_ml_pipeline.joblib):

python train_model.py


3. Launch the Streamlit App

Run the interactive frontend dashboard:

streamlit run app.py


📋 Model Card Summary

Target Audience: Parking facility managers and automated security routing.

Core Features: Entry hour, parking duration, hourly rate, total fee, VIP status, floor location, and guard shift.

Performance: High predictive accuracy (>95%) across generated customer behavioral tiers.# Smart-Multi-Storey-Car-Parking-ML-Intelligence-System
An end-to-end Machine Learning pipeline and interactive web application built to optimize multi-storey parking facility operations using relational SQL integration, unsupervised clustering, supervised classification, and real-time deployment.
