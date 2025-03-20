import pandas as pd
from typing import Optional, Tuple, List, Any

class Market:
    """Base Market class that defines the interface for market operations"""
    
    def __init__(self, opt_df: pd.DataFrame, mechanism_solver: Optional[Any] = None, input_format: Optional[str] = None):
        """Initialize the Market class with option data and mechanism solver"""
        pass

    def check_match(self, orders: pd.DataFrame, offset: bool = True) -> Tuple[bool, float]:
        """Check if orders can be matched"""
        pass

    def apply_mechanism(self, orders: pd.DataFrame, offset: bool = True) -> Tuple[bool, float]:
        """Apply mechanism solver to market data"""
        pass

    def epsilon_priceQuote(self, option_to_quote: pd.DataFrame, orders_in_market: Optional[pd.DataFrame] = None, offset: bool = True) -> float:
        """Quote price for option with epsilon amount"""
        pass

    def priceQuote(self, option_to_quote: pd.DataFrame, orders_in_market: Optional[pd.DataFrame] = None, liquidity: Optional[pd.Series] = None, offset: bool = True) -> float:
        """Generate price of given input order w.r.t orders in the market"""
        pass

    def frontierGeneration(self, orders: Optional[pd.DataFrame] = None, epsilon: bool = False) -> pd.DataFrame:
        """Generate frontier of options with epsilon price quote and constraints"""
        pass

    def epsilon_frontierGeneration(self, orders: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Generate frontier of options with epsilon price quote"""
        pass

    def get_market_data_order_format(self) -> pd.DataFrame:
        """Get market data in order format"""
        pass

    def get_strikes(self) -> List[float]:
        """Get list of strike prices"""
        pass 