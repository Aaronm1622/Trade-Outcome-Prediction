# main.py

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from src.data.data_loader import DataLoader, DataDiagnostics
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.visualization.visualizer import Visualizer  # Ensure this class is defined
from src.utils.config import Config  # Ensure this class is defined
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'output',
        'output/plots',
        'output/models',
        'output/results'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logging.info("Directory setup complete")

def run_analysis():
    """Main analysis pipeline"""
    try:
        # Load configuration
        logging.info("Loading configuration...")
        config = Config.load_config()
        
        # Initialize data loader
        logging.info("Initializing data loader...")
        data_loader = DataLoader(
            congress_excel_path=config['congress_excel_path'],
            stock_data_path=config['stock_data_path'],
            spy_data_path=config['spy_data_path']
        )
        
        # Initialize diagnostics
        diagnostics = DataDiagnostics()
        
        # Load congressional data
        logging.info("Loading congressional trading data...")
        congress_data = data_loader.load_congress_data()
        
        # First diagnostic check
        initial_data = congress_data.copy()
        diagnostics.print_filtering_summary(
            initial_data, 
            congress_data, 
            "Congressional Data Loading"
        )
        
        # Load stock data
        logging.info("Loading stock market data...")
        stock_data = data_loader.load_stock_data()
        
        # Company matching analysis
        diagnostics.analyze_market_data_coverage(congress_data, stock_data)
        
        # Load SPY data
        logging.info("Loading SPY data...")
        spy_data = data_loader.load_spy_data()
        
        # Feature engineering
        logging.info("Engineering features...")
        feature_engineer = FeatureEngineer()
        processed_data = feature_engineer.create_time_features(congress_data)
        
        # Feature engineering diagnostic
        diagnostics.print_filtering_summary(
            congress_data,
            processed_data,
            "Feature Engineering"
        )
        
        # Create market features
        market_features = feature_engineer.create_market_features(
            processed_data, 
            stock_data
        )
        
        # Combine all features
        final_data = processed_data.copy()
        for col in market_features.columns:
            final_data[col] = market_features[col]
        
        # Create interaction features
        final_data = feature_engineer.create_interaction_features(final_data)
        
        # Feature engineering diagnostic after interactions
        diagnostics.print_filtering_summary(
            processed_data,
            final_data,
            "Interaction Feature Engineering"
        )
        
        # Initialize model trainer
        model_trainer = ModelTrainer()
        
        # Prepare data for modeling
        X, y = model_trainer.prepare_data(final_data)
        
        # Split data into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train baseline and other models
        logging.info("Training models...")
        baseline_model = model_trainer.train_baseline(X_train, y_train)
        rf_model = model_trainer.train_random_forest(X_train, y_train)
        xgb_model = model_trainer.train_xgboost(X_train, y_train)
        
        # Train Voting Classifier
        voting_model = model_trainer.train_voting_classifier(X_train, y_train)
        
        # Create visualizations
        logging.info("Creating visualizations...")
        visualizer = Visualizer()
        
        # Get and plot feature importance
        feature_importance = model_trainer.get_feature_importance()
        logging.info("Feature Importance:")
        for model_name, importances in feature_importance.items():
            logging.info(f"\n{model_name} Feature Importances:")
            logging.info(importances)
            
            # Create a DataFrame for feature importances
            importance_df = pd.DataFrame({
                'feature': model_trainer.feature_columns,
                'importance': importances
            })
            
            # Plot feature importances
            importance_plot = visualizer.plot_feature_importance(importance_df, model_name, top_n=10)
            importance_plot.savefig(f'output/plots/feature_importance_{model_name.replace(" ", "_").lower()}.png')
            plt.close()
        
        # Evaluate models on test set
        logging.info("Evaluating models on test set...")
        models_to_evaluate = model_trainer.models  # This includes Voting Classifier now
        
        for model_name, model in models_to_evaluate.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1]
            
            # Accuracy
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            logging.info(f"\n{model_name} Train Accuracy: {train_acc:.3f}")
            logging.info(f"{model_name} Test Accuracy: {test_acc:.3f}")
            
            # Classification Report
            report = classification_report(y_test, y_pred)
            logging.info(f"\nClassification Report for {model_name}:\n{report}")
            
            # ROC-AUC Score
            roc_auc = roc_auc_score(y_test, y_proba)
            logging.info(f"ROC-AUC Score for {model_name}: {roc_auc:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            confusion_plot = visualizer.plot_confusion_matrix(cm, model_name)
            confusion_plot.savefig(f'output/plots/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
            plt.close()
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_plot = visualizer.plot_roc_curve(fpr, tpr, roc_auc, model_name)
            roc_plot.savefig(f'output/plots/roc_curve_{model_name.replace(" ", "_").lower()}.png')
            plt.close()
        
        # Save detailed results
        results_df = pd.DataFrame([{
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'models_evaluated': list(models_to_evaluate.keys())
        }])
        results_df.to_csv('output/results/model_performance_summary.csv', index=False)
        
        # (Optional) Create and save additional plots, e.g., trading analysis
        # trading_plot = visualizer.plot_trading_analysis(processed_data)
        # trading_plot.savefig('output/plots/trading_analysis.png')
        
        # Calculate and store results
        results = {
            'baseline_train_accuracy': baseline_model.score(X_train, y_train),
            'baseline_test_accuracy': baseline_model.score(X_test, y_test),
            'rf_train_accuracy': rf_model.score(X_train, y_train),
            'rf_test_accuracy': rf_model.score(X_test, y_test),
            'xgb_train_accuracy': xgb_model.score(X_train, y_train),
            'xgb_test_accuracy': xgb_model.score(X_test, y_test),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_importance': {model: list(importances) for model, importances in feature_importance.items()}
        }
        
        # Save detailed results
        results_df = pd.DataFrame([{
            'baseline_train_accuracy': results['baseline_train_accuracy'],
            'baseline_test_accuracy': results['baseline_test_accuracy'],
            'rf_train_accuracy': results['rf_train_accuracy'],
            'rf_test_accuracy': results['rf_test_accuracy'],
            'xgb_train_accuracy': results['xgb_train_accuracy'],
            'xgb_test_accuracy': results['xgb_test_accuracy'],
            'timestamp': results['timestamp'],
            'n_samples': results['n_samples'],
            'n_features': results['n_features']
        }])
        results_df.to_csv('output/results/model_performance.csv', index=False)
        
        # Save feature importance separately
        for model_name, importances in feature_importance.items():
            fi_df = pd.DataFrame({
                'feature': model_trainer.feature_columns,
                'importance': importances
            }).sort_values(by='importance', ascending=False)
            fi_df.to_csv(f'output/results/feature_importance_{model_name.replace(" ", "_").lower()}.csv', index=False)
        
        logging.info("Analysis complete! Results saved in output directory.")
        return results
            
    except Exception as e:
        logging.error(f"Error in analysis: {str(e)}")
        logging.error("Pipeline failed")
        raise

def main():
    """Main execution function"""
    try:
        # Setup directories
        logging.info("Starting analysis pipeline...")
        setup_directories()
        
        # Run analysis
        results = run_analysis()
        
        # Print summary
        print("\nAnalysis Summary:")
        print("----------------")
        print(f"Number of samples: {results['n_samples']}")
        print(f"Number of features: {results['n_features']}")
        print(f"Baseline Model Train Accuracy: {results['baseline_train_accuracy']:.3f}")
        print(f"Baseline Model Test Accuracy: {results['baseline_test_accuracy']:.3f}")
        print(f"Random Forest Train Accuracy: {results['rf_train_accuracy']:.3f}")
        print(f"Random Forest Test Accuracy: {results['rf_test_accuracy']:.3f}")
        print(f"XGBoost Train Accuracy: {results['xgb_train_accuracy']:.3f}")
        print(f"XGBoost Test Accuracy: {results['xgb_test_accuracy']:.3f}")
        print(f"\nResults saved in output directory")
        print("Check output/plots for visualizations")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
