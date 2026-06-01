import pandas as pd
import pickle
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add error handling for loading rules
try:
    rules = pickle.load(open('/root/Desktop/D-ML-projects/Market_Basket_Analysis/marketbasketbackendapp/data/trained_model.pkl', 'rb'))
    if not isinstance(rules, pd.DataFrame):
        rules = pd.DataFrame(rules, columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
except FileNotFoundError:
    logging.error("Trained model file not found.")
    rules = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
except Exception as e:
    logging.error(f"Error loading trained model: {e}")
    rules = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])

filename = 'trained_model.pkl'

# Save rules with error handling
try:
    pickle.dump(rules, open(filename, 'wb'))
    logging.info(f"Trained model saved to {filename}")
except Exception as e:
    logging.error(f"Error saving trained model: {e}")

rules
def recommend_items(items, rules, top_n=200):
    # Convert input items to a set
    item_set = set(items)

    # Filter rules for exact matches
    exact_match_rules = rules[rules['antecedents'].apply(lambda x: set(x) == item_set)]
    print(f"Number of exact match rules: {len(exact_match_rules)}")

    # Filter rules for single-item antecedents that overlap with input items
    single_item_rules = rules[rules['antecedents'].apply(lambda x: any(item in x for item in items))]
    print(f"Number of single item rules: {len(single_item_rules)}")


    # Combine exact and single-item rules
    combined_rules = pd.concat([exact_match_rules, single_item_rules]).drop_duplicates()
    print(f"Number of combined rules: {len(combined_rules)}")

    # Filter for single-consequent rules
    single_consequent_rules = combined_rules[combined_rules['consequents'].apply(lambda x: len(x) == 1)]
    print(f"Number of single-consequent rules: {len(single_consequent_rules)}")

    if not single_consequent_rules.empty:
        # Sort by confidence and lift
        sorted_rules = single_consequent_rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])
        recommendations = sorted_rules.head(top_n)
        return recommendations
    else:
        print("No single-consequent rules found, considering multi-item consequents.")

        # Fall back to multi-consequent rules
        multi_consequent_rules = combined_rules[combined_rules['consequents'].apply(lambda x: len(x) > 1)]
        print(f"Number of multi-consequent rules: {len(multi_consequent_rules)}")

        if multi_consequent_rules.empty:
            print("No recommendations found.")
            return pd.DataFrame()

        # Sort and return top multi-consequent recommendations
        sorted_multi_rules = multi_consequent_rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])
        recommendations = sorted_multi_rules.head(top_n)
        return recommendations
def output(data, recommendations, item_to_check):

    if 'ProductID' not in data.columns:
        print("Error: 'ProductID' column missing in data.")
        return [], []

    # Check if ProductName column exists; fallback to ProductID if missing
    if 'ProductName' in data.columns:
        data_unique = data[['ProductID', 'ProductName']].drop_duplicates()
        # Ensure both keys and values are strings to avoid type issues
        product_id_to_name = {str(pid): str(pname) for pid, pname in
                             zip(data_unique['ProductID'], data_unique['ProductName'])}
        print(f"Created product_id_to_name dictionary with {len(product_id_to_name)} entries")
        # Print a few examples for debugging
        items = list(product_id_to_name.items())[:5]
        for pid, pname in items:
            print(f"Example mapping: '{pid}' -> '{pname}'")
    else:
        print("Warning: 'ProductName' column not found in data. Using ProductID instead.")
        data_unique = data[['ProductID']].drop_duplicates()
        product_id_to_name = {str(pid): str(pid) for pid in data_unique['ProductID']}

    # Display the recommendations
    print(f"Top unique items to recommend for '{item_to_check}':")
    if recommendations.empty:
        print("No recommendations found.")
        return [], []

    print(recommendations[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    # Gather product names for the given item_to_check
    given_item_names = []
    for item in item_to_check:
        item_str = str(item)
        given_item_name = product_id_to_name.get(item_str, "Product Name Not Found")

        # Try alternative formats if not found
        if given_item_name == "Product Name Not Found":
            # Try without leading zeros
            alt_id = item_str.lstrip('0')
            if alt_id in product_id_to_name:
                given_item_name = product_id_to_name[alt_id]
            # Try as integer if possible
            elif item_str.isdigit() and str(int(item_str)) in product_id_to_name:
                given_item_name = product_id_to_name[str(int(item_str))]

        print(f"For given Item '{item}': {given_item_name}")
        given_item_names.append(given_item_name)

    # Gather recommended product names
    product_ids = set()
    for consequents in recommendations['consequents']:
        if isinstance(consequents, (frozenset, set, list)):
            # Convert each item to string and add to set
            for item in consequents:
                product_ids.add(str(item))
        else:
            product_ids.add(str(consequents))  # Ensure single items are strings

    print(f"Found {len(product_ids)} unique product IDs in recommendations")

    # Map product_id to confidence value from recommendations
    # For each product_id, find the max confidence among rules recommending it
    output = []
    for product_id in product_ids:
        # Get product name, with fallback if not found
        product_name = product_id_to_name.get(product_id, "Product Name Not Found")
        if product_name == "Product Name Not Found":
            alt_id = product_id.lstrip('0')
            if alt_id in product_id_to_name:
                product_name = product_id_to_name[alt_id]
            elif product_id.isdigit() and str(int(product_id)) in product_id_to_name:
                product_name = product_id_to_name[str(int(product_id))]

        # Find all rules in recommendations where this product_id is in the consequents
        confidence_values = []
        for idx, row in recommendations.iterrows():
            consequents = row['consequents']
            if isinstance(consequents, (frozenset, set, list)):
                if product_id in map(str, consequents):
                    confidence_values.append(row['confidence'])
            else:
                if str(consequents) == product_id:
                    confidence_values.append(row['confidence'])
        # Use the max confidence value for this product_id
        confidence = max(confidence_values) if confidence_values else 0.0
        confidence_percent = round(confidence * 100, 2)
        output.append({
            'product_id': product_id,
            'product_name': product_name,
            'probability_percent': confidence_percent
        })

    return given_item_names, output

def Model_run(data, item_to_check, rules_path=None):
    # Use the provided rules_path if given, otherwise use the global rules
    if rules_path and os.path.exists(rules_path):
        try:
            assoc_rules = pickle.load(open(rules_path, 'rb'))
            if not isinstance(assoc_rules, pd.DataFrame):
                assoc_rules = pd.DataFrame(assoc_rules, columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
        except Exception as e:
            print(f"Error loading rules from {rules_path}: {e}")
            assoc_rules = rules
    else:
        assoc_rules = rules

    # Generate recommendations
    recommendations = recommend_items(item_to_check, assoc_rules)
    if recommendations is None or recommendations.empty:
        print("No recommendations found.")
        return
    else:
        print('F1 SUCCESS')

    # Generate the final output
    final_output = output(data, recommendations,item_to_check)  # Ensure that item_to_check is passed if needed
    if final_output is None or len(final_output) == 0:
        print("No final output generated.")
        return
    else:
        print(final_output)
        print('F2 SUCCESS')
        return final_output