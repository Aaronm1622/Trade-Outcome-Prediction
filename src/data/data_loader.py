# src/data/data_loader.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
from src.features.feature_engineering import FeatureEngineer  # Ensure correct import

class DataLoader:
    """Class to handle all data loading operations"""
    def __init__(self, congress_excel_path, stock_data_path, spy_data_path):
        self.congress_excel_path = congress_excel_path
        self.stock_data_path = stock_data_path
        self.spy_data_path = spy_data_path
        
        # Enhanced company variations mapping
        self.company_variations = {
            'Amazon': ['AMAZON', 'AMZN', 'AMAZON.COM', 'AMAZON COM INC', 'AMAZON SERVICES'],
            'Apple': ['APPLE', 'AAPL', 'APPLE INC', 'APPLE COMPUTERS', 'APPLE COMPUTER INC'],
            'Facebook': ['FACEBOOK', 'FB', 'META', 'META PLATFORMS', 'FACEBOOK INC', 'META PLATFORMS INC'],
            'Google': ['GOOGLE', 'GOOGL', 'GOOG', 'ALPHABET', 'ALPHABET INC', 'GOOGLE INC', 'ALPHABET CLASS A'],
            'Microsoft': ['MICROSOFT', 'MSFT', 'MICROSOFT CORP', 'MICROSOFT CORPORATION'],
            'Netflix': ['NETFLIX', 'NFLX', 'NETFLIX INC', 'NETFLIX COM INC'],
            'Tesla': ['TESLA', 'TSLA', 'TESLA MOTORS', 'TESLA INC', 'TESLA AUTOMOTIVE'],
            'Walmart': ['WALMART', 'WMT', 'WAL-MART', 'WALMART INC', 'WAL MART STORES INC']
        }
        
        # Create reverse mapping for quick lookups
        self.variations_lookup = {}
        for standard_name, variations in self.company_variations.items():
            for variation in variations:
                self.variations_lookup[variation] = standard_name

    def load_congress_data(self):
        """Load and preprocess congressional trading data with enhanced matching"""
        logging.info("Loading congressional trading data...")
        
        # Load the Excel file 
        df = pd.read_excel(self.congress_excel_path)
        
        # Convert Traded column to Date early
        df['Date'] = pd.to_datetime(df['Traded'])
        
        # Filter for 2015-2021 date range
        mask = (df['Date'] >= '2015-01-01') & (df['Date'] <= '2021-12-31')
        df = df[mask]
        
        # Add matched company column
        df['matched_company'] = df['Company'].apply(self._match_company)
        
        # Analyze matching results
        self.analyze_company_matches(df)
        
        # Filter to only include matched companies 
        df_filtered = df[df['matched_company'].notna()].copy()
        
        stock_data = self.load_stock_data()
        spy_data = self.load_spy_data()

        df_filtered = self.calculate_excess_returns(df_filtered, stock_data, spy_data)

        # Initialize FeatureEngineer
        feature_engineer = FeatureEngineer()
        
        # Create Time Features (includes 'trade_size')
        df_filtered = feature_engineer.create_time_features(df_filtered)
        logging.info(f"Columns after time feature engineering: {df_filtered.columns.tolist()}")
        
        # Create Market Features
        market_features_df = feature_engineer.create_market_features(df_filtered, stock_data)
        df_filtered = pd.concat([df_filtered, market_features_df], axis=1)
        logging.info(f"Columns after market feature engineering: {df_filtered.columns.tolist()}")
        
        # Create Interaction Features
        df_filtered = feature_engineer.create_interaction_features(df_filtered)
        logging.info(f"Columns after interaction feature engineering: {df_filtered.columns.tolist()}")
        
        # Sort by date
        df_filtered = df_filtered.sort_values('Date')
        
        logging.info(f"Loaded {len(df_filtered)} congressional trades")
        logging.info(f"Date range: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")
        logging.info(f"Number of unique companies: {df_filtered['matched_company'].nunique()}")
        
        return df_filtered

    def _match_company(self, company_name):
        """Enhanced company name matching"""
        if pd.isna(company_name):
            return None
            
        # Clean and standardize the input name
        company_upper = company_name.upper().strip()
        company_upper = company_upper.replace('.', ' ').replace(',', ' ')
        company_upper = ' '.join(company_upper.split())  # Normalize whitespace
        
        # Try exact match first
        if company_upper in self.variations_lookup:
            return self.variations_lookup[company_upper]
        
        # Try partial matches
        for variation, standard_name in self.variations_lookup.items():
            if variation in company_upper:
                return standard_name
                
        return None

    def analyze_company_matches(self, df):
        """Analyze company name matching results"""
        logging.info("\nCompany Matching Analysis:")
        
        # Original company names - handle NaN values
        valid_companies = df['Company'].dropna().unique()
        unique_companies = sorted(str(company) for company in valid_companies)
        logging.info(f"\nTotal unique company names in data: {len(unique_companies)}")
        
        # Matching results
        matched = df[df['matched_company'].notna()]
        unmatched = df[df['matched_company'].isna()]
        
        # Basic statistics
        matched_pct = len(matched)/len(df)*100
        unmatched_pct = len(unmatched)/len(df)*100
        
        logging.info(f"\nMatching Summary:")
        logging.info(f"Total trades: {len(df)}")
        logging.info(f"Matched trades: {len(matched)} ({matched_pct:.2f}%)")
        logging.info(f"Unmatched trades: {len(unmatched)} ({unmatched_pct:.2f}%)")
        
        # Matches by company with value information
        logging.info("\nMatches by company:")
        for company in sorted(matched['matched_company'].unique()):
            company_data = matched[matched['matched_company'] == company]
            n_trades = len(company_data)
            pct = n_trades/len(df)*100
            date_range = f"{company_data['Date'].min():%Y-%m-%d} to {company_data['Date'].max():%Y-%m-%d}"
            logging.info(f"{company}: {n_trades} trades ({pct:.2f}%) - {date_range}")
        
        # Most common unmatched companies
        if len(unmatched) > 0:
            logging.info("\nTop 20 unmatched companies by trade count:")
            unmatched_counts = unmatched['Company'].value_counts().head(20)
            for company, count in unmatched_counts.items():
                pct = count/len(df)*100
                logging.info(f"  {company}: {count} trades ({pct:.2f}%)")

    def load_stock_data(self):
        """Load stock price data for matched companies"""
        logging.info("Loading stock market data...")
        stock_data = {}
        
        # Get list of standard company names
        standard_names = set(self.company_variations.keys())
        
        for file in os.listdir(self.stock_data_path):
            if file.endswith('.csv') and file != 'spy.csv':
                company = file.split('.')[0]
                if company in standard_names:
                    file_path = os.path.join(self.stock_data_path, file)
                    stock_data[company] = pd.read_csv(file_path)
                    stock_data[company]['Date'] = pd.to_datetime(stock_data[company]['Date'])
                    logging.info(f"Loaded {company} data: {len(stock_data[company])} rows")
        
        return stock_data

    def load_spy_data(self):
        """Load SPY benchmark data"""
        logging.info("Loading SPY data...")
        spy_data = pd.read_csv(self.spy_data_path)
        spy_data['Date'] = pd.to_datetime(spy_data['Date'])
        logging.info(f"Loaded SPY data: {len(spy_data)} rows")
        return spy_data
    
    def calculate_excess_returns(self, df, stock_data, spy_data, window_days=30, use_log_returns=True):
        """Calculate excess returns for each trade with enhanced methodologies"""
        df = df.copy()
        excess_returns = []
        
        # Precompute SPY returns to optimize performance
        spy_data = spy_data.set_index('Date').sort_index()
        spy_returns = spy_data['Close'].pct_change(window_days) * 100  # Simple return
        if use_log_returns:
            spy_returns = np.log(spy_data['Close'] / spy_data['Close'].shift(window_days)) * 100  # Log return
        
        for _, trade in df.iterrows():
            trade_date = trade['Date']
            company = trade['matched_company']
            
            if company not in stock_data:
                excess_returns.append(np.nan)
                continue
            
            # Get stock prices
            stock_prices = stock_data[company].set_index('Date').sort_index()
            if trade_date not in stock_prices.index:
                # Use the closest previous trading day
                stock_start_price = stock_prices['Close'].asof(trade_date)
                if pd.isna(stock_start_price):
                    excess_returns.append(np.nan)
                    continue
            else:
                stock_start_price = stock_prices.loc[trade_date, 'Close']
            
            # Calculate end date as window_days trading days after trade_date
            stock_dates = stock_prices.index
            try:
                trade_idx = stock_dates.get_loc(trade_date)
                end_idx = trade_idx + window_days
                if end_idx >= len(stock_prices):
                    excess_returns.append(np.nan)
                    continue
                stock_end_price = stock_prices.iloc[end_idx]['Close']
            except KeyError:
                excess_returns.append(np.nan)
                continue
            
            # Calculate stock return
            if use_log_returns:
                stock_return = (np.log(stock_end_price / stock_start_price)) * 100
            else:
                stock_return = ((stock_end_price / stock_start_price) - 1) * 100
            
            # Get SPY return
            if trade_date not in spy_returns.index:
                spy_return = spy_returns.asof(trade_date)
            else:
                spy_return = spy_returns.loc[trade_date]
            
            if pd.isna(stock_return) or pd.isna(spy_return):
                excess_returns.append(np.nan)
            else:
                excess_return = stock_return - spy_return
                excess_returns.append(excess_return)
        
        df['excess_return'] = excess_returns
        
        # Log statistics
        valid_returns = df['excess_return'].dropna()
        logging.info(f"\nExcess Returns Summary:")
        logging.info(f"Valid returns calculated: {len(valid_returns)} out of {len(df)}")
        logging.info(f"Mean excess return: {valid_returns.mean():.2f}%")
        logging.info(f"Median excess return: {valid_returns.median():.2f}%")
        
        return df
    
class DataDiagnostics:
    """Class to track and analyze data filtering steps"""
    
    @staticmethod
    def print_filtering_summary(original_df, filtered_df, step_name):
        """Print summary of data filtering step"""
        initial_count = len(original_df)
        final_count = len(filtered_df)
        dropped_count = initial_count - final_count
        
        logging.info(f"\n{step_name} Summary:")
        logging.info(f"Initial samples: {initial_count}")
        logging.info(f"Final samples: {final_count}")
        logging.info(f"Dropped samples: {dropped_count} ({(dropped_count/initial_count)*100:.2f}%)")
        
        if dropped_count > 0:
            # Analyze which rows were dropped
            if 'Date' in original_df.columns:
                date_range = original_df['Date'].agg(['min', 'max'])
                logging.info(f"Original date range: {date_range['min']} to {date_range['max']}")
            
            # Check for null values in each column
            null_counts = filtered_df.isnull().sum()
            if null_counts.any():
                logging.info("\nNull values by column:")
                for col, count in null_counts[null_counts > 0].items():
                    logging.info(f"{col}: {count} nulls")

    @staticmethod
    def analyze_market_data_coverage(congress_trades, stock_data):
        """Analyze stock data coverage for congressional trades"""
        trade_dates = congress_trades['Date'].unique()
        
        coverage_stats = {}
        for company, data in stock_data.items():
            data_dates = data['Date'].unique()
            covered_trades = sum(trade_date in data_dates for trade_date in trade_dates)
            coverage_stats[company] = {
                'total_trades': len(trade_dates),
                'covered_trades': covered_trades,
                'coverage_percentage': (covered_trades / len(trade_dates)) * 100
            }
        
        logging.info("\nStock Data Coverage Analysis:")
        for company, stats in coverage_stats.items():
            logging.info(f"\n{company}:")
            logging.info(f"Total trades: {stats['total_trades']}")
            logging.info(f"Covered trades: {stats['covered_trades']}")
            logging.info(f"Coverage: {stats['coverage_percentage']:.2f}%")

# Update ModelTrainer class prepare_data method

def prepare_data(self, df):
    """Prepare data for modeling with diagnostics"""
    logging.info("\nData Preparation Diagnostics:")
    
    # Initial state
    initial_samples = len(df)
    logging.info(f"\nInitial number of samples: {initial_samples}")
    
    # Check required columns
    missing_cols = [col for col in self.feature_columns if col not in df.columns]
    if missing_cols:
        missing_cols_str = ', '.join(missing_cols)
        logging.info(f"\nMissing required columns: {missing_cols_str}")
        raise ValueError(f"Missing required columns: {missing_cols_str}")
    
    # Select features and target
    X = df[self.feature_columns].copy()
    y = (df[self.target_column] > 0).astype(int)
    
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
    logging.info(f"Positive class (profitable trades): {class_dist[1]:.2f}%")
    logging.info(f"Negative class (unprofitable trades): {class_dist[0]:.2f}%")
    
    # Scale features
    X_scaled = self.scaler.fit_transform(X)
    
    return X_scaled, y    