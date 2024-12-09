# src/visualization/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

class Visualizer:
    """Class to handle all visualization tasks"""
    
    @staticmethod
    def plot_feature_importance(importance_df, model_name, top_n=10):
        """
        Plot feature importance
        
        Parameters:
        importance_df (pd.DataFrame): DataFrame with columns 'feature' and 'importance'
        model_name (str): Name of the model for plot titling
        top_n (int): Number of top features to display
        
        Returns:
        matplotlib.figure.Figure: The created plot
        """
        plt.figure(figsize=(12, 8))
        
        # Sort by importance and select top_n
        importance_df = importance_df.sort_values('importance', ascending=True).tail(top_n)
        
        # Create horizontal bar plot
        sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
        
        plt.title(f'Top {top_n} Feature Importances for {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_confusion_matrix(cm, model_name):
        """
        Plot confusion matrix as a heatmap.
        
        Parameters:
        cm (array-like): Confusion matrix
        model_name (str): Name of the model for plot titling
        
        Returns:
        matplotlib.figure.Figure: The created plot
        """
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_roc_curve(fpr, tpr, roc_auc, model_name):
        """
        Plot ROC curve.
        
        Parameters:
        fpr (array-like): False positive rates
        tpr (array-like): True positive rates
        roc_auc (float): Area Under the Curve
        model_name (str): Name of the model for plot titling
        
        Returns:
        matplotlib.figure.Figure: The created plot
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic for {model_name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_trading_analysis(df):
        """Create trading analysis plots"""
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)
        
        # Plot 1: Trading volume by company
        ax1 = fig.add_subplot(gs[0, 0])
        company_counts = df['matched_company'].value_counts()
        sns.barplot(x=company_counts.values, y=company_counts.index, ax=ax1)
        ax1.set_title('Number of Trades by Company')
        ax1.set_xlabel('Number of Trades')
        
        # Plot 2: Average excess returns by company
        ax2 = fig.add_subplot(gs[0, 1])
        avg_returns = df.groupby('matched_company')['excess_return'].mean().sort_values()
        sns.barplot(x=avg_returns.values, y=avg_returns.index, ax=ax2)
        ax2.set_title('Average Excess Returns by Company')
        ax2.set_xlabel('Average Excess Return')
        
        # Plot 3: Trading volume over time
        ax3 = fig.add_subplot(gs[1, :])
        monthly_trades = df.resample('M', on='Date').size()
        monthly_trades.plot(ax=ax3)
        ax3.set_title('Trading Volume Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Trades')
        
        # Plot 4: Return distribution
        ax4 = fig.add_subplot(gs[2, 0])
        sns.histplot(data=df, x='excess_return', ax=ax4)
        ax4.set_title('Distribution of Excess Returns')
        ax4.set_xlabel('Excess Return')
        
        # Plot 5: Returns by party
        ax5 = fig.add_subplot(gs[2, 1])
        sns.boxplot(data=df, x='Party', y='excess_return', ax=ax5)
        ax5.set_title('Excess Returns by Party')
        
        plt.tight_layout()
        return fig
