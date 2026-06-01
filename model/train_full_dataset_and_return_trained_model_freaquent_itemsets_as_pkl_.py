def clean_data(data):
    if 'InvoiceNumber' in data.columns:
        data['InvoiceNumber'] = data['InvoiceNumber']#if column names have simple adjestments
        data['InvoiceNumber'] = data['InvoiceNumber'].astype(str).str.strip()
    else:
        print("Error: 'InvoiceNumber' column not found in the data.")
        return None
    if 'ProductID' in data.columns:
        data['ProductID'] = data['ProductID'].astype(str).str.strip()
    else:
        print("Error: 'ProductID' column not found in the data.")
        return None
    if 'ProductName' in data.columns:
        data['ProductName'] = data['ProductName'].astype(str).str.strip()
    else:
        print("Error: 'ProductName' column not found in the data.")
        return None
    data.info()

    if 'InvoiceNumber' in data.columns and 'ProductID' in data.columns and 'ProductName' in data.columns:
        data = data.dropna(subset=['InvoiceNumber', 'ProductID','ProductName'])
        data.info()
    # Group by the appropriate columns
    print(f'number of products{data["ProductID"].nunique()}')
    cleaned_data= data.groupby(['InvoiceNumber', 'ProductID','ProductName']).sum().reset_index()
    cleaned_data=cleaned_data[['InvoiceNumber', 'ProductID','ProductName']]
    print(f'number of products{cleaned_data["ProductID"].nunique()}')
    cleaned_data['InvoiceNumber'] = cleaned_data['InvoiceNumber'].astype(str).str.strip()
    cleaned_data.info()
    return cleaned_data


# cleaned_data.to_csv('data/cleaned_monthlySale.csv')

def calculate_min_support(N):
    if N > 5000:
        return 0.0025  # lower than before
    elif 2000 < N <= 5000:
        return 0.003
    elif 1000 < N <= 2000:
        return 0.01
    else:
        return 0.02


def calculate_min_threshold(N):
    if N > 1000:
        return 0.18  # lower to get more rules
    elif 500 < N <= 1000:
        return 0.2
    else:
        return 0.25



def model_training(flat_data, product_ids=None, memory_optimized=False):
    """
    Train a market basket analysis model on the provided data.

    Args:
        flat_data: DataFrame containing the transaction data
        product_ids: Optional list of product IDs to filter by
        memory_optimized: If True, use more conservative parameters to reduce memory usage

    Returns:
        DataFrame containing association rules
    """
    print(f"Starting model training with memory_optimized={memory_optimized}")

    try:
        # For very large datasets, sample the data to reduce memory usage
        if memory_optimized and len(flat_data) > 100000:
            sample_size = min(100000, int(len(flat_data) * 0.5))  # 50% or max 100k rows
            print(f"Memory optimization: Sampling {sample_size} rows from {len(flat_data)} total rows")
            flat_data = flat_data.sample(n=sample_size, random_state=42)

        # Pivot the data to create a basket format
        print("Creating basket format...")
        basket = (
            flat_data
            .groupby(['InvoiceNumber', 'ProductID'])['ProductID']
            .count().unstack()
            .reset_index().fillna(0)
            .set_index("InvoiceNumber")
        )

        print(f"Basket shape: {basket.shape}")

        # Function to hot encode the values
        def encode_values(x):
            return 1 if x > 0 else 0

        # Hot encode the basket
        print("Hot encoding basket...")
        if memory_optimized:
            # Process in chunks to reduce memory usage
            chunk_size = 1000
            chunks = []
            for i in range(0, len(basket), chunk_size):
                end = min(i + chunk_size, len(basket))
                chunk = basket.iloc[i:end].applymap(encode_values)
                chunks.append(chunk)
            basket_encoded = pd.concat(chunks)
            del chunks  # Free memory
        else:
            basket_encoded = basket.applymap(encode_values)

        # Free memory
        del basket

        # Filter transactions containing at least 2 items
        print("Filtering transactions...")
        basket_filtered = basket_encoded[(basket_encoded > 0).sum(axis=1) >= 2]

        # Free memory
        del basket_encoded

        # Calculate the number of transactions after filtering
        N = len(basket_filtered)
        print(f"Number of transactions after filtering: {N}")

        # Dynamically set min_support and min_threshold
        min_support = calculate_min_support(N)
        min_threshold = calculate_min_threshold(N)

        # For memory optimization, use more conservative parameters
        if memory_optimized:
            min_support = max(min_support, 0.01)  # Minimum 1% support
            min_threshold = max(min_threshold, 0.3)  # Minimum 30% confidence

        print(f"Using min_support: {min_support}, min_threshold: {min_threshold}")

        # Generate frequent itemsets
        print("Generating frequent itemsets...")
        frequent_itemsets = apriori(
            basket_filtered, min_support=min_support, use_colnames=True,
            low_memory=memory_optimized  # Use low memory mode if optimizing
        )

        # Free memory
        del basket_filtered

        print(f"Number of frequent itemsets: {len(frequent_itemsets)}")
        if len(frequent_itemsets) > 0:
            print(frequent_itemsets.head())

        # Generate association rules
        print("Generating association rules...")
        assoc_rules = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=min_threshold, num_itemsets=1
        ).sort_values(['confidence', 'lift'], ascending=[False, False]).reset_index(drop=True)

        # Free memory
        del frequent_itemsets

        # Limit the number of rules for memory optimization
        if memory_optimized and len(assoc_rules) > 10000:
            print(f"Memory optimization: Limiting rules from {len(assoc_rules)} to 10000")
            assoc_rules = assoc_rules.head(10000)

        print(f"Number of rules generated: {len(assoc_rules)}")
        if len(assoc_rules) > 0:
            print(assoc_rules.head())

        return assoc_rules

    except MemoryError:
        print("Memory error occurred during model training")
        raise MemoryError("Insufficient memory to complete model training. Try using a smaller dataset or more memory.")
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise



# from mlxtend.frequent_patterns import apriori, association_rules
# # flat_data_extracted = clean_data(data)
# association_rules = model_training(cleaned_data)



# # save trained model as  pkl

# import pickle
# with open('data/trained_model.pkl', 'wb') as file:
#     pickle.dump(association_rules, file)


from mlxtend.frequent_patterns import apriori, association_rules

# Add this section to load data, clean it, and save it
if __name__ == "__main__":
    import pandas as pd
    import os

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Load your data - update the path as needed
    data_path = 'path/to/your/monthlySale.csv'  # Update this path
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded data with {len(data)} rows")

        # Clean the data
        cleaned_data = clean_data(data)

        if cleaned_data is not None:
            # Save cleaned data
            cleaned_data.to_csv('data/cleaned_monthlySale.csv', index=False)
            print("Saved cleaned data to data/cleaned_monthlySale.csv")

            # Train model
            association_rules = model_training(cleaned_data)

            # Save model
            import pickle
            with open('data/trained_model.pkl', 'wb') as file:
                pickle.dump(association_rules, file)
            print("Saved trained model to data/trained_model.pkl")
        else:
            print("Data cleaning failed")
    except FileNotFoundError:
        print(f"Error: Could not find file {data_path}")

