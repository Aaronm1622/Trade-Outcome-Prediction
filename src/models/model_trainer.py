# src/models/model_trainer.py

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

class ModelTrainer:
    """Class to handle model training and evaluation"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []  # To be defined based on features used
        self.target_column = 'excess_return'  # Adjust if different
        self.models = {}
    
    def prepare_data(self, df):
        """Prepare data for modeling with diagnostics"""
        logging.info("\nData Preparation Diagnostics:")
        
        # Initial state
        initial_samples = len(df)
        logging.info(f"\nInitial number of samples: {initial_samples}")
        
        # Define feature columns
        # List all columns that are features
        self.feature_columns = [
            'party_code', 'chamber_code', 'state_code', 'is_purchase',
            'momentum_7d', 'trade_size',
            'trade_size_x_is_purchase', 'party_x_chamber', 'state_x_is_purchase',
            'momentum_7d_x_trade_size'
        ]
        
        # Check required columns
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            missing_cols_str = ', '.join(missing_cols)
            logging.info(f"\nMissing required columns: {missing_cols_str}")
            raise ValueError(f"Missing required columns: {missing_cols_str}")
        
        # Select features and target
        X = df[self.feature_columns].copy()
        y = (df['excess_return'] > 0).astype(int)
        
        # Check for missing values in features
        null_counts = X.isnull().sum()
        if null_counts.any():
            logging.info("\nNull values in features:")
            for col, count in null_counts[null_counts > 0].items():
                logging.info(f"{col}: {count} nulls ({(count/len(X))*100:.2f}%)")
        
        # Check for missing values in target
        target_nulls = y.isnull().sum()
        if target_nulls > 0:
            logging.info(f"\nNull values in target: {target_nulls} ({(target_nulls/len(y))*100:.2f}%)")
        
        # Remove rows with any missing values
        complete_cases = X.notna().all(axis=1) & y.notna()
        X = X[complete_cases]
        y = y[complete_cases]
        
        # Final state
        final_samples = len(X)
        logging.info(f"\nFinal number of samples: {final_samples}")
        logging.info(f"Dropped samples: {initial_samples - final_samples} ({((initial_samples - final_samples)/initial_samples)*100:.2f}%)")
        
        # Class distribution
        class_dist = y.value_counts(normalize=True) * 100
        logging.info("\nClass distribution:")
        logging.info(f"Positive class (profitable trades): {class_dist.get(1, 0):.2f}%")
        logging.info(f"Negative class (unprofitable trades): {class_dist.get(0, 0):.2f}%")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_baseline(self, X_train, y_train):
        """Train a baseline Logistic Regression model with hyperparameter tuning"""
        logging.info("Training Baseline Logistic Regression Model with Hyperparameter Tuning...")
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
        lr = LogisticRegression(max_iter=1000, random_state=42)
        grid_search = GridSearchCV(
            estimator=lr,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_lr = grid_search.best_estimator_
        logging.info(f"Best Logistic Regression Parameters: {grid_search.best_params_}")
        self.models['Baseline Logistic Regression'] = best_lr
        return best_lr
    
    def train_random_forest(self, X_train, y_train):
        """Train a Random Forest model with hyperparameter tuning"""
        logging.info("Training Random Forest Model with Hyperparameter Tuning...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        logging.info(f"Best Random Forest Parameters: {grid_search.best_params_}")
        self.models['Random Forest'] = best_rf
        return best_rf
    
    def train_xgboost(self, X_train, y_train):
        """Train an XGBoost model with hyperparameter tuning"""
        logging.info("Training XGBoost Model with Hyperparameter Tuning...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        xgb_model = xgb.XGBClassifier(
            eval_metric='logloss',
            #use_label_encoder=False,
            random_state=42
        )
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_xgb = grid_search.best_estimator_
        logging.info(f"Best XGBoost Parameters: {grid_search.best_params_}")
        self.models['XGBoost'] = best_xgb
        return best_xgb
    
    def get_feature_importance(self):
        """Retrieve feature importances from tree-based models"""
        feature_importance = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                feature_importance[name] = model.feature_importances_
        return feature_importance

    def train_voting_classifier(self, X_train, y_train):
        """Train a Voting Classifier combining Logistic Regression, Random Forest, and XGBoost"""
        logging.info("Training Voting Classifier...")
        
        # Ensure that base models have been trained
        if not all(model_name in self.models for model_name in ['Baseline Logistic Regression', 'Random Forest', 'XGBoost']):
            logging.error("Base models are not all trained. Please train Logistic Regression, Random Forest, and XGBoost first.")
            raise ValueError("Base models not trained.")
        
        # Retrieve the trained base models
        lr = self.models['Baseline Logistic Regression']
        rf = self.models['Random Forest']
        xgb_model = self.models['XGBoost']
        
        # Initialize Voting Classifier with soft voting
        voting_clf = VotingClassifier(
            estimators=[
                ('lr', lr),
                ('rf', rf),
                ('xgb', xgb_model)
            ],
            voting='soft',
            n_jobs=-1  # Utilize all available cores
        )
        
        # Fit the Voting Classifier
        voting_clf.fit(X_train, y_train)
        self.models['Voting Classifier'] = voting_clf
        logging.info("Voting Classifier trained successfully.")
        return voting_clf