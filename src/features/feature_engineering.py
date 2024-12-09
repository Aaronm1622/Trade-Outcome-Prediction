import pandas as pd
import numpy as np
import logging
import re

class FeatureEngineer:
    @staticmethod
    def create_time_features(df):
        processed_df = df.copy()
        
        # Basic categorical features
        processed_df['party_code'] = pd.Categorical(processed_df['Party']).codes
        processed_df['chamber_code'] = pd.Categorical(processed_df['Chamber']).codes
        processed_df['state_code'] = pd.Categorical(processed_df['State']).codes
        processed_df['is_purchase'] = processed_df['Transaction'].str.lower().str.contains('purchase').astype(int)
        
        # Process 'Trade_Size_USD' to create 'trade_size'
        if 'Trade_Size_USD' in processed_df.columns:
            processed_df['trade_size'] = processed_df['Trade_Size_USD'].apply(FeatureEngineer._process_trade_size)
            logging.info("Created 'trade_size' from 'Trade_Size_USD'")
        else:
            logging.warning("'Trade_Size_USD' column is missing. 'trade_size' cannot be created.")
            processed_df['trade_size'] = np.nan  # or handle as appropriate
        
        # Impute missing 'trade_size' with median
        if 'trade_size' in processed_df.columns:
            median_trade_size = processed_df['trade_size'].median()
            processed_df['trade_size'].fillna(median_trade_size, inplace=True)
            logging.info(f"Imputed missing 'trade_size' values with median: {median_trade_size}")
        
        logging.info("\nFeature Creation Summary:")
        logging.info("Categorical features created: party_code, chamber_code, state_code, is_purchase")
        if 'trade_size' in processed_df.columns:
            logging.info("Feature 'trade_size' created.")
        
        return processed_df
    
    @staticmethod
    def _process_trade_size(trade_size_str):
        """
        Convert 'Trade_Size_USD' string to numerical midpoint.
        Examples:
            '$1,001 - $15,000' -> 8000.5
            '$15,001 - $50,000' -> 32500.5
            '$50,001+' -> 50001  # or another logic for open-ended ranges
        """
        if pd.isna(trade_size_str):
            return np.nan
        
        # Remove dollar signs and commas
        trade_size_str = trade_size_str.replace('$', '').replace(',', '')
        
        # Use regex to extract numbers
        range_match = re.match(r'(\d+)\s*-\s*(\d+)', trade_size_str)
        if range_match:
            lower = float(range_match.group(1))
            upper = float(range_match.group(2))
            midpoint = (lower + upper) / 2
            return midpoint
        else:
            # Handle open-ended ranges like '$50,001+'
            open_ended_match = re.match(r'(\d+)\+', trade_size_str)
            if open_ended_match:
                lower = float(open_ended_match.group(1))
                # Define a logic for upper bound, e.g., assume a certain multiplier
                # Here, we'll add a fixed amount, say $50,000
                upper = lower + 50000
                midpoint = (lower + upper) / 2
                return midpoint
            else:
                # If format is unexpected, log a warning and return NaN
                logging.warning(f"Unexpected trade size format: '{trade_size_str}'. Setting as NaN.")
                return np.nan
    
    @staticmethod
    def create_market_features(df, stock_data, lookback_days=7):
        logging.info(f"\nMarket Features Creation:")
        logging.info(f"Starting with {len(df)} trades")
        
        market_features = []
        
        for idx, trade in df.iterrows():
            company = trade['matched_company']
            trade_date = trade['Date']
            
            if company not in stock_data:
                market_features.append({'momentum_7d': np.nan})
                continue
                
            stock_prices = stock_data[company]
            closest_trade_date = stock_prices[stock_prices['Date'] <= trade_date]['Date'].max()
            
            if pd.isnull(closest_trade_date):
                market_features.append({'momentum_7d': np.nan})
                continue
            
            trade_idx = stock_prices[stock_prices['Date'] == closest_trade_date].index[0]
            if trade_idx < lookback_days:
                market_features.append({'momentum_7d': np.nan})
                continue
                
            lookback_data = stock_prices.iloc[trade_idx - lookback_days : trade_idx + 1]
            
            if len(lookback_data) >= lookback_days:
                momentum = (lookback_data['Close'].iloc[-1] / lookback_data['Close'].iloc[0] - 1)
                market_features.append({'momentum_7d': momentum})
            else:
                market_features.append({'momentum_7d': np.nan})
        
        market_features_df = pd.DataFrame(market_features, index=df.index)
        
        valid_samples = market_features_df['momentum_7d'].notna().sum()
        logging.info(f"Valid momentum samples: {valid_samples} out of {len(df)}")
        
        return market_features_df
    
    @staticmethod
    def create_interaction_features(df):
        """
        Create interaction features by combining existing features.
        """
        logging.info("Creating interaction features...")
        
        # Check if 'trade_size' exists
        if 'trade_size' not in df.columns:
            logging.error("'trade_size' column is missing. Cannot create interaction features.")
            raise KeyError("'trade_size' column is missing.")
        
        # Optionally handle missing 'trade_size' values
        if df['trade_size'].isnull().any():
            logging.warning("'trade_size' contains NaN values. Filling with 0.")
            df['trade_size'].fillna(0, inplace=True)
        
        # Interaction between trade_size and is_purchase
        df['trade_size_x_is_purchase'] = df['trade_size'] * df['is_purchase']
        
        # Interaction between party_code and chamber_code
        df['party_x_chamber'] = df['party_code'] * df['chamber_code']
        
        # Interaction between state_code and is_purchase
        df['state_x_is_purchase'] = df['state_code'] * df['is_purchase']
        
        # Interaction between momentum_7d and trade_size
        df['momentum_7d_x_trade_size'] = df['momentum_7d'] * df['trade_size']
        
        logging.info("Interaction features created: trade_size_x_is_purchase, party_x_chamber, state_x_is_purchase, momentum_7d_x_trade_size")
        
        return df
