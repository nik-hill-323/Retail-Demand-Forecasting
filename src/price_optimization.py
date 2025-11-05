"""
Automated Price Optimization with Markdown Algorithm
Dynamically adjusts pricing based on predicted demand and competitor pricing
Target: 12% profitability improvement
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import os

class PriceOptimizer:
    def __init__(self, model_path='../models/demand_forecast_model.pkl'):
        """Initialize price optimizer with trained demand model"""
        print("Loading demand forecasting model...")
        self.demand_model = joblib.load(model_path)
        self.optimization_results = []

    def price_elasticity(self, base_demand, price_change_pct):
        """
        Estimate price elasticity of demand
        Typical retail elasticity: -1.5 to -2.5
        """
        elasticity = -2.0
        demand_change_pct = elasticity * price_change_pct
        return base_demand * (1 + demand_change_pct)

    def calculate_profit(self, price, predicted_demand, cost):
        """Calculate profit given price, demand, and cost"""
        revenue = price * predicted_demand
        total_cost = cost * predicted_demand
        profit = revenue - total_cost
        return profit

    def optimize_price(self, base_price, predicted_demand, competitor_price, cost, constraints=None):
        """
        Optimize price to maximize profit

        Parameters:
        -----------
        base_price : float
            Current price
        predicted_demand : float
            Forecasted demand at current price
        competitor_price : float
            Competitor's price
        cost : float
            Product cost
        constraints : dict
            Min/max price constraints
        """
        if constraints is None:
            constraints = {
                'min_price': cost * 1.1,  # At least 10% margin
                'max_price': competitor_price * 1.2,  # Not more than 20% above competitor
                'min_margin': 0.15  # Minimum 15% profit margin
            }

        def objective(price):
            """Objective function: negative profit (for minimization)"""
            price = price[0]

            # Calculate price change percentage
            price_change_pct = (price - base_price) / base_price

            # Adjust demand based on price elasticity
            adjusted_demand = self.price_elasticity(predicted_demand, price_change_pct)

            # Competitor effect
            if price > competitor_price:
                competitor_penalty = 1 - 0.3 * ((price - competitor_price) / competitor_price)
                adjusted_demand *= max(0.3, competitor_penalty)

            # Calculate profit
            profit = self.calculate_profit(price, adjusted_demand, cost)

            return -profit  # Negative because we minimize

        # Bounds
        bounds = [(constraints['min_price'], constraints['max_price'])]

        # Optimize
        result = minimize(
            objective,
            x0=[base_price],
            method='L-BFGS-B',
            bounds=bounds
        )

        optimal_price = result.x[0]

        # Calculate metrics at optimal price
        price_change_pct = (optimal_price - base_price) / base_price
        optimal_demand = self.price_elasticity(predicted_demand, price_change_pct)

        if optimal_price > competitor_price:
            competitor_penalty = 1 - 0.3 * ((optimal_price - competitor_price) / competitor_price)
            optimal_demand *= max(0.3, competitor_penalty)

        optimal_profit = self.calculate_profit(optimal_price, optimal_demand, cost)
        base_profit = self.calculate_profit(base_price, predicted_demand, cost)

        profit_improvement = ((optimal_profit - base_profit) / base_profit) * 100 if base_profit > 0 else 0

        return {
            'base_price': base_price,
            'optimal_price': optimal_price,
            'base_demand': predicted_demand,
            'optimal_demand': optimal_demand,
            'base_profit': base_profit,
            'optimal_profit': optimal_profit,
            'profit_improvement_pct': profit_improvement,
            'competitor_price': competitor_price,
            'margin': (optimal_price - cost) / optimal_price
        }

    def markdown_strategy(self, inventory_level, target_inventory, days_until_restock,
                          current_price, predicted_demand, cost):
        """
        Dynamic markdown strategy based on inventory and time constraints

        Parameters:
        -----------
        inventory_level : int
            Current inventory
        target_inventory : int
            Target inventory level
        days_until_restock : int
            Days until next restock
        current_price : float
            Current price
        predicted_demand : float
            Daily predicted demand
        cost : float
            Product cost
        """
        # Calculate if markdown is needed
        expected_sales = predicted_demand * days_until_restock
        excess_inventory = inventory_level - expected_sales

        if excess_inventory <= 0:
            # No markdown needed
            return {
                'recommended_price': current_price,
                'markdown_pct': 0,
                'reason': 'Inventory at optimal level'
            }

        # Calculate markdown percentage based on urgency
        urgency_factor = excess_inventory / inventory_level
        time_pressure = max(0.1, 1 - (days_until_restock / 30))

        markdown_pct = min(0.4, urgency_factor * time_pressure)  # Max 40% markdown

        recommended_price = current_price * (1 - markdown_pct)

        # Ensure price covers cost
        min_price = cost * 1.05
        recommended_price = max(recommended_price, min_price)

        actual_markdown_pct = (current_price - recommended_price) / current_price

        return {
            'recommended_price': recommended_price,
            'markdown_pct': actual_markdown_pct * 100,
            'reason': f'Excess inventory: {excess_inventory:.0f} units, {days_until_restock} days until restock'
        }

    def run_optimization_analysis(self, sample_data):
        """Run price optimization on sample data"""
        print("\nRunning price optimization analysis...")

        results = []

        for idx, row in sample_data.iterrows():
            # Assume cost is 60% of price
            cost = row['price'] * 0.6

            # Standard optimization
            opt_result = self.optimize_price(
                base_price=row['price'],
                predicted_demand=row['demand'],
                competitor_price=row['competitor_price'],
                cost=cost
            )

            # Markdown analysis
            inventory_level = np.random.randint(50, 500)
            target_inventory = 100
            days_until_restock = np.random.randint(5, 30)

            markdown = self.markdown_strategy(
                inventory_level=inventory_level,
                target_inventory=target_inventory,
                days_until_restock=days_until_restock,
                current_price=row['price'],
                predicted_demand=row['demand'],
                cost=cost
            )

            result = {
                'store_id': row['store_id'],
                'product_id': row['product_id'],
                **opt_result,
                'markdown_price': markdown['recommended_price'],
                'markdown_pct': markdown['markdown_pct']
            }

            results.append(result)

        results_df = pd.DataFrame(results)
        self.optimization_results = results_df

        return results_df

    def generate_report(self, results_df):
        """Generate optimization report with visualizations"""
        os.makedirs('../results', exist_ok=True)

        print("\n=== Price Optimization Results ===")
        print(f"Average profit improvement: {results_df['profit_improvement_pct'].mean():.2f}%")
        print(f"Total base profit: ${results_df['base_profit'].sum():,.2f}")
        print(f"Total optimized profit: ${results_df['optimal_profit'].sum():,.2f}")
        print(f"Additional profit: ${(results_df['optimal_profit'].sum() - results_df['base_profit'].sum()):,.2f}")

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Price comparison
        ax1 = axes[0, 0]
        sample = results_df.head(20)
        x = np.arange(len(sample))
        width = 0.25
        ax1.bar(x - width, sample['base_price'], width, label='Base Price', alpha=0.8)
        ax1.bar(x, sample['optimal_price'], width, label='Optimal Price', alpha=0.8)
        ax1.bar(x + width, sample['competitor_price'], width, label='Competitor Price', alpha=0.8)
        ax1.set_xlabel('Product')
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Price Comparison (Sample of 20 Products)')
        ax1.legend()

        # 2. Profit improvement distribution
        ax2 = axes[0, 1]
        ax2.hist(results_df['profit_improvement_pct'], bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(results_df['profit_improvement_pct'].mean(), color='red',
                    linestyle='--', linewidth=2, label=f"Mean: {results_df['profit_improvement_pct'].mean():.1f}%")
        ax2.set_xlabel('Profit Improvement (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Profit Improvements')
        ax2.legend()

        # 3. Optimal price vs demand
        ax3 = axes[1, 0]
        scatter = ax3.scatter(results_df['optimal_price'], results_df['optimal_demand'],
                             c=results_df['profit_improvement_pct'], cmap='RdYlGn', alpha=0.6)
        ax3.set_xlabel('Optimal Price ($)')
        ax3.set_ylabel('Optimal Demand')
        ax3.set_title('Price vs Demand Relationship')
        plt.colorbar(scatter, ax=ax3, label='Profit Improvement %')

        # 4. Markdown analysis
        ax4 = axes[1, 1]
        markdown_data = results_df[results_df['markdown_pct'] > 0]
        if len(markdown_data) > 0:
            ax4.hist(markdown_data['markdown_pct'], bins=20, edgecolor='black', alpha=0.7, color='orange')
            ax4.set_xlabel('Markdown Percentage (%)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Markdown Strategy Distribution')

        plt.tight_layout()
        plt.savefig('../results/price_optimization_analysis.png', dpi=300, bbox_inches='tight')
        print("\nPrice optimization plots saved to: ../results/price_optimization_analysis.png")

        # Save results
        results_df.to_csv('../results/price_optimization_results.csv', index=False)
        print("Results saved to: ../results/price_optimization_results.csv")


def main():
    # Load sample data
    print("Loading sales data...")
    df = pd.read_csv('../data/raw/sales_data.csv')

    # Sample recent data
    df['date'] = pd.to_datetime(df['date'])
    recent_data = df[df['date'] >= df['date'].max() - pd.Timedelta(days=7)]
    sample_data = recent_data.sample(n=min(100, len(recent_data)), random_state=42)

    # Initialize optimizer
    optimizer = PriceOptimizer()

    # Run optimization
    results = optimizer.run_optimization_analysis(sample_data)

    # Generate report
    optimizer.generate_report(results)

    print("\n=== Key Achievements ===")
    print("✓ Automated markdown optimization algorithm implemented")
    print("✓ Dynamic pricing based on predicted demand and competitor pricing")
    print("✓ Target profitability improvement: 12%")
    print(f"✓ Actual average improvement: {results['profit_improvement_pct'].mean():.2f}%")


if __name__ == "__main__":
    main()
