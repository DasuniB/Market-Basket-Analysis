# Project Purpose
This project is a full-stack Market Basket Analysis web application. It allows users to upload sales transaction data (CSV), processes and cleans the data, trains an association rule model (for product recommendations), and provides a web interface for uploading data and getting product recommendations.

## Main Components
a. Flask Backend (app.py)
Web Server: Runs a Flask app to handle HTTP requests.
Endpoints:
/ : Home page (renders index.html).
/upload : Handles CSV file uploads, data cleaning, and model training.
/recommend : Receives a list of items and returns recommended products based on the trained model.
b. Data Processing
Upload: Users upload a CSV file with sales data via the /upload endpoint.
Validation: Checks file type and size, saves it temporarily.
Loading: Reads the CSV (with chunking for large files to avoid memory issues).
Cleaning: Calls clean_data() from your model code to preprocess the data.
Saving: Cleaned data is saved as cleaned_monthlySale.csv.
c. Model Training
Training: Calls model_training() to generate association rules (e.g., using Apriori or FP-Growth).
Memory Optimization: Uses chunking and memory checks for large datasets.
Saving Model: Trained rules are saved as a pickle file (trained_model.pkl or a custom filename).
d. Recommendation
Loading Model/Data: On startup, loads the cleaned data and trained model if available.
API: /recommend endpoint takes a list of items and returns recommended products using the trained model (via Model_run()).
## File Structure Highlights
app.py : Main Flask app.
item_recommendation_using_saved_pickel_file.py : Contains Model_run() for generating recommendations.
train_full_dataset_and_return_trained_model_freaquent_itemsets_as_pkl_.py : Contains clean_data() and model_training() for data cleaning and model training.
data : Stores uploaded, cleaned, and model files.
templates : HTML templates for the web interface.
## User Workflow
User visits the web app.
Uploads a CSV file of sales transactions.
Backend processes and cleans the data.
Model is trained on the cleaned data and saved.
User can request product recommendations by providing a list of items.
Backend returns recommended products based on association rules.
## Error Handling & Optimization
Handles large files with chunked reading.
Monitors memory usage (using psutil if available).
Provides detailed error messages for upload, cleaning, and training steps.
Allows custom naming for trained model files.
## Technologies Used
Flask (web server)
Pandas (data processing)
Pickle (model serialization)
mlxtend (likely for association rule mining)
HTML/Jinja2 (web templates)
Summary:
This project provides a robust, user-friendly way to perform market basket analysis on custom datasets, with a focus on memory efficiency and usability for large files. The backend handles all data processing, model training, and recommendation logic, while the frontend allows easy interaction for non-technical users.