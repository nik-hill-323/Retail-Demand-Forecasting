"""
Data Generator for Retail Demand Forecasting
Generates synthetic sales data with realistic patterns including:
- Seasonality
- Trends
- Store-specific variations
- Economic indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

class RetailDataGenerator:
    def __init__(self, n_stores=10, n_products=50, n_days=730):
        """
        Initialize the data generator

        Parameters:
        -----------
        n_stores : int
            Number of retail stores
        n_products : int
            Number of products
        n_days : int
            Number of days of historical data
        """
        self.n_stores = n_stores
        self.n_products = n_products
        self.n_days = n_days
        self.start_date = datetime.now() - timedelta(days=n_days)

    def generate_economic_indicators(self):
        """Generate external economic indicators"""
        dates = pd.date_range(start=self.start_date, periods=self.n_days, freq='D')

        # Generate realistic economic indicators
        base_cpi = 100
        cpi_trend = np.linspace(0, 10, self.n_days)
        cpi_noise = np.random.normal(0, 0.5, self.n_days)
        cpi = base_cpi + cpi_trend + cpi_noise

        # Unemployment rate
        unemployment = 5 + np.random.normal(0, 0.3, self.n_days)
        unemployment = np.clip(unemployment, 3, 8)

        # Consumer confidence index
        confidence = 70 + np.random.normal(0, 5, self.n_days)

        economic_data = pd.DataFrame({
            'date': dates,
            'cpi': cpi,
            'unemployment_rate': unemployment,
            'consumer_confidence': confidence
        })

        return economic_data

    def generate_sales_data(self, economic_data):
        """Generate realistic sales data with patterns"""
        data = []

        product_base_prices = np.random.uniform(10, 200, self.n_products)
        store_traffic_factors = np.random.uniform(0.7, 1.3, self.n_stores)

        for day in range(self.n_days):
            current_date = self.start_date + timedelta(days=day)

            # Day of week effect
            day_of_week = current_date.weekday()
            weekend_boost = 1.3 if day_of_week >= 5 else 1.0

            # Seasonal effect
            month = current_date.month
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * month / 12)

            # Holiday effect (simplified)
            holiday_boost = 1.5 if month in [11, 12] else 1.0

            # Economic impact
            econ_impact = 1 + (economic_data.loc[day, 'consumer_confidence'] - 70) / 100

            for store_id in range(self.n_stores):
                for product_id in range(self.n_products):
                    # Base demand
                    base_demand = np.random.poisson(20)

                    # Apply all factors
                    demand = base_demand * weekend_boost * seasonal_factor * holiday_boost * econ_impact * store_traffic_factors[store_id]
                    demand = int(demand)

                    # Price with variation
                    base_price = product_base_prices[product_id]
                    price_variation = np.random.uniform(0.9, 1.1)
                    price = base_price * price_variation

                    # Competitor price
                    competitor_price = price * np.random.uniform(0.85, 1.15)

                    # Stock status
                    stockout = 1 if np.random.random() < 0.05 else 0
                    if stockout:
                        demand = 0

                    revenue = demand * price

                    data.append({
                        'date': current_date,
                        'store_id': f'STORE_{store_id:03d}',
                        'product_id': f'PROD_{product_id:03d}',
                        'demand': demand,
                        'price': round(price, 2),
                        'competitor_price': round(competitor_price, 2),
                        'revenue': round(revenue, 2),
                        'stockout': stockout,
                        'day_of_week': day_of_week,
                        'month': month,
                        'is_weekend': int(day_of_week >= 5),
                        'is_holiday_season': int(month in [11, 12])
                    })

        sales_df = pd.DataFrame(data)

        # Merge with economic indicators
        sales_df = sales_df.merge(economic_data, on='date', how='left')

        return sales_df

    def generate_and_save(self):
        """Generate all data and save to files"""
        print("Generating economic indicators...")
        economic_data = self.generate_economic_indicators()

        print("Generating sales data...")
        sales_data = self.generate_sales_data(economic_data)

        # Save data
        output_dir = '../data/raw'
        os.makedirs(output_dir, exist_ok=True)

        sales_data.to_csv(f'{output_dir}/sales_data.csv', index=False)
        economic_data.to_csv(f'{output_dir}/economic_indicators.csv', index=False)

        print(f"\nData generation complete!")
        print(f"Total records: {len(sales_data):,}")
        print(f"Date range: {sales_data['date'].min()} to {sales_data['date'].max()}")
        print(f"Stores: {self.n_stores}")
        print(f"Products: {self.n_products}")
        print(f"\nFiles saved to: {output_dir}/")

        # Display summary statistics
        print("\n=== Sales Data Summary ===")
        print(sales_data.describe())

        return sales_data, economic_data


if __name__ == "__main__":
    generator = RetailDataGenerator(n_stores=10, n_products=50, n_days=730)
    sales_data, economic_data = generator.generate_and_save()
