"""
Supabase Database Adapter for Cyber News Classifier
────────────────────────────────────────────────────────────────────────────────
Connects to Supabase and fetches data from cyber_news table.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()


class SupabaseAdapter:
    """Adapter for fetching cyber_news data from Supabase."""
    
    def __init__(self, url=None, key=None):
        self.url = url or os.environ.get('SUPABASE_URL')
        self.key = key or os.environ.get('SUPABASE_KEY')
        
        if not self.url or not self.key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set in environment variables or passed explicitly."
            )
        
        self.client = create_client(self.url, self.key)
    
    def fetch_cyber_news(self, limit=None, filters=None):
        """
        Fetch data from cyber_news table.
        
        Args:
            limit: Maximum number of rows to fetch (None = all)
            filters: Dict of column:value filters (e.g., {'country': 'Malaysia'})
        
        Returns:
            DataFrame with cyber news data
        """
        query = self.client.table('cyber_news').select('*')
        
        # Apply filters if provided
        if filters:
            for column, value in filters.items():
                query = query.eq(column, value)
        
        # Apply limit if provided
        if limit:
            query = query.limit(limit)
        
        response = query.execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            return df
        return pd.DataFrame()
    
    def fetch_table(self, table_name):
        """Generic table fetcher (for compatibility with page_ai_classifier)."""
        response = self.client.table(table_name).select('*').execute()
        if response.data:
            return pd.DataFrame(response.data)
        return pd.DataFrame()


def get_supabase_data(table_name):
    """
    Factory function that returns a callable for fetching data.
    This matches the expected signature for page_ai_classifier.
    """
    adapter = SupabaseAdapter()
    
    def fetch(table_name):
        if table_name == 'global_news':
            # Map 'global_news' to your actual 'cyber_news' table
            return adapter.fetch_cyber_news()
        return adapter.fetch_table(table_name)
    
    return fetch
