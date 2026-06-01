"""
Market Basket Analysis Training Module

This module provides functions for cleaning transaction data and training
market basket analysis models using efficient_apriori library.

PRODUCTION USAGE:
- Functions are imported and used by the Flask web application (mbapp.py)
- Data comes from user uploads through the web interface
- No file paths needed - data is passed as pandas DataFrames

TESTING USAGE:
- Run this file directly: python train_dataset_with_polar.py
- Update the data_path in the __main__ section to point to your test CSV

Required CSV columns: InvoiceNumber, ProductID, ProductName
"""

# We using polars for efficient data processing, pandas for compatibility
import polars as pl
import pandas as pd

# Basket Libraries
from efficient_apriori import apriori

def clean_data(data):
    """
    Clean the input data for market basket analysis using Polars for memory efficiency.

    Args:
        data: pandas DataFrame or str (CSV file path)

    Returns:
        pandas DataFrame with cleaned data
    """
    # Handle both file path and DataFrame
    if isinstance(data, str):
        pl_data = pl.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        pl_data = pl.from_pandas(data)
    else:
        raise ValueError("Input to clean_data must be a file path or a pandas DataFrame.")

    # Keep only required columns
    required_cols = ["InvoiceNumber", "ProductID", "ProductName"]
    missing_cols = [col for col in required_cols if col not in pl_data.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return None
    pl_data = pl_data.select(required_cols)

    # Strip whitespace and cast to string for all required columns in one call
    pl_data = pl_data.with_columns([
        pl.col(col).cast(pl.Utf8).str.strip_chars().alias(col) for col in required_cols
    ])

    # Drop rows with nulls in required columns
    pl_data = pl_data.drop_nulls(required_cols)

    # Remove duplicates by grouping
    before_nunique = pl_data["ProductID"].n_unique()
    print(f'Number of unique products before grouping: {before_nunique}')
    pl_data = pl_data.unique(subset=required_cols)
    after_nunique = pl_data["ProductID"].n_unique()
    print(f'Number of unique products after grouping: {after_nunique}')

    # Final info
    print("Final cleaned data info:")
    print(pl_data.describe())

    # Convert back to pandas for compatibility
    cleaned_data = pl_data.to_pandas()
    return cleaned_data


def model_training(df, product_ids=None, memory_optimized=False):
    """
    Train a market basket analysis model using efficient_apriori.
    Optimized for best accuracy with aggressive parameters.

    Args:
        df: pandas DataFrame containing the transaction data
        product_ids: Optional list of product IDs to filter by (not used in this implementation)
        memory_optimized: Kept for compatibility but ignored (always uses best accuracy parameters)

    Returns:
        pandas DataFrame containing association rules
    """
    # Ignore memory_optimized parameter - always use best accuracy
    print(f"Starting model training optimized for best accuracy")

    try:
        import gc
        # Convert pandas DataFrame to polars for efficient processing
        df_pl = pl.from_pandas(df)

        # Only keep necessary columns for transactions
        df_pl = df_pl.select(["InvoiceNumber", "ProductID"])

        # Convert ProductID to categorical for memory efficiency
        df_pl = df_pl.with_columns([
            pl.col("ProductID").cast(pl.Categorical)
        ])

        # Group transactions and create transaction lists
        print("Creating transaction format...")
        df_grouped = (
            df_pl.sort(["InvoiceNumber", "ProductID"])
            .unique(subset=["InvoiceNumber", "ProductID"], maintain_order=True)
            .group_by("InvoiceNumber").agg([pl.col("ProductID")])
        )

        # Use Polars' to_list for efficient extraction
        transactions = df_grouped["ProductID"].to_list()
        print(f"Number of transactions: {len(transactions)}")

        # Free memory from large intermediates
        del df_pl, df_grouped
        gc.collect()

        # Calculate dynamic parameters based on dataset size
        N = len(transactions)

        # Always use optimized parameters for best accuracy (lower thresholds to capture more patterns)
        if N > 5000:
            min_support = 0.00025  # Even lower for maximum pattern discovery
            min_confidence = 0.01  # Very low confidence to capture weak but valid associations
        elif N > 1000:
            min_support = 0.0005
            min_confidence = 0.01
        else:
            min_support = 0.002
            min_confidence = 0.05

        print(f"Using min_support: {min_support}, min_confidence: {min_confidence}")

        # Generate frequent itemsets and rules
        print("Generating frequent itemsets and association rules...")
        itemsets, rules = apriori(transactions,
                                min_support=min_support,
                                min_confidence=min_confidence)

        print(f"Number of frequent itemsets: {len(itemsets)}")
        print(f"Number of rules generated: {len(rules)}")

        # Filter rules to ensure both antecedents and consequents exist
        filtered_rules = [rule for rule in rules if len(rule.lhs) >= 1 and len(rule.rhs) >= 1]
        print(f"Number of filtered rules: {len(filtered_rules)}")

        # Sort the filtered rules by confidence (descending)
        sorted_rules = sorted(filtered_rules, key=lambda rule: rule.confidence, reverse=True)

        # Use all rules for best accuracy (no limiting)

        # Create a list of dictionaries for the DataFrame
        rules_data = [
            {
                "antecedents": list(rule.lhs),  # Convert lhs to a list
                "consequents": list(rule.rhs),  # Convert rhs to a list
                "lift": rule.lift,
                "confidence": rule.confidence,
                "conviction": rule.conviction,
                "support": rule.support,
            }
            for rule in sorted_rules
        ]

        # Create a pandas DataFrame (not polars) for compatibility with the rest of the system
        rules_df = pd.DataFrame(rules_data)

        print(f"Final number of rules in DataFrame: {len(rules_df)}")
        if len(rules_df) > 0:
            print("Sample rules:")
            print(rules_df.head())

        return rules_df

    except Exception as e:
        print(f"Error in model training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    import pandas as pd
    import os

    print("=" * 60)
    print("Market Basket Analysis - Training Module")
    print("=" * 60)
    print()
    print("This module provides functions for cleaning data and training")
    print("market basket analysis models using efficient_apriori.")
    print()
    print("USAGE:")
    print("- In production: Data is provided by user uploads through the web interface")
    print("- For testing: Update the data_path below to point to your CSV file")
    print()
    print("Required CSV columns: InvoiceNumber, ProductID, ProductName")
    print("=" * 60)

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # For testing purposes only - update this path to your actual CSV file
    # In production, data comes from user uploads via the web interface
    data_path = 'data/cleaned_monthlySale.csv'  # Update this path for testing

    if os.path.exists(data_path):
        try:
            print(f"Loading test data from: {data_path}")
            data = pd.read_csv(data_path)
            print(f"Loaded data with {len(data)} rows")

            # Clean the data
            print("\nCleaning data...")
            cleaned_data = clean_data(data_path)

            if cleaned_data is not None:
                # Save cleaned data
                cleaned_data.to_csv('data/cleaned_monthlySale.csv', index=False)
                print("Saved cleaned data to data/cleaned_monthlySale.csv")

                # Train model
                print("\nTraining model...")
                association_rules = model_training(cleaned_data)

                # Save model
                import pickle
                with open('data/test_trained_model.pkl', 'wb') as file:
                    pickle.dump(association_rules, file)
                print("Saved trained model to data/test_trained_model.pkl")
                print(f"Generated {len(association_rules)} association rules")
            else:
                print("Data cleaning failed")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Test data file not found: {data_path}")
        print("\nTo test this module:")
        print("1. Place your CSV file in the data/ directory")
        print("2. Update the data_path variable above")
        print("3. Run this script again")
        print("\nNote: In production, data is uploaded through the web interface")




















