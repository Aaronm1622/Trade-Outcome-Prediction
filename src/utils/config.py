class Config:
    """Configuration management"""
    @staticmethod
    def load_config():
        """Load configuration from file or environment"""
        return {
            'congress_excel_path': 'Data/Congressmen/congress-trading-all.xlsx',
            'stock_data_path': 'Data/archive',
            'spy_data_path': 'Data/archive/spy.csv'
        }