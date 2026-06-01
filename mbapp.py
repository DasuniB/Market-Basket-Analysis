print("Loading app.py...")
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import os
import pickle
import sys
from werkzeug.utils import secure_filename
# Import the Model_run function from the app package
from app.item_recommendation_using_saved_pickel_file import Model_run
# Add the model directory to the path so we can import the training module
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
sys.path.append(model_dir)
# from model.train_full_dataset_and_return_trained_model_freaquent_itemsets_as_pkl_ import clean_data, model_training
from model.train_dataset_with_polar import clean_data, model_training


app = Flask(__name__)
app.secret_key = 'market_basket_analysis_secret_key'

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Configure timeout settings
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes
app.config['REQUEST_TIMEOUT'] = 1800  # 30 minutes for request timeout

# Create data directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Configure longer timeouts for processing large datasets
PROCESSING_TIMEOUT = 1800  # 30 minutes in seconds

# Load cleaned data once at startup
DATA_PATH = os.path.join("data", "cleaned_monthlySale.csv")
RULES_PATH = os.path.join("data", "trained_model.pkl")

try:
    data = pd.read_csv(DATA_PATH)
    print(f"Loaded data with {len(data)} rows")
except FileNotFoundError:
    data = None
    print("Could not load dataset.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_products():
    if data is None:
        return jsonify({"error": "Cleaned data not loaded."}), 500

    try:
        # Get items from request
        request_data = request.get_json()
        item_to_check = request_data.get('item_to_check', [])

        # Get recommendations
        recommendations = Model_run(data, item_to_check)

        if recommendations:
            # Unpack the tuple returned by Model_run
            given_items, recommended_items = recommendations

            # Return the recommended items directly
            return jsonify({"recommended_items": recommended_items})
        else:
            return jsonify({"recommended_items": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global data  # Declare global variable at the beginning of the function

    if request.method == 'POST':
        print("Received POST request to /upload")
        print(f"Request form data: {request.form}")
        print(f"Request files: {request.files}")

        # Check if the post request has the file part
        if 'file' not in request.files:
            print("Error: No file part in the request")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        print(f"File received: {file.filename}, {file.content_type}, {file.content_length} bytes")

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            print("Error: Empty filename")
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_' + filename)
            print(f"Saving file to temporary path: {temp_path}")

            try:
                file.save(temp_path)
                print(f"File saved successfully to {temp_path}")

                # Load the uploaded CSV file
                try:
                    data = pd.read_csv(temp_path)
                    print(f"Successfully loaded data with {len(data)} rows")
                except Exception as e:
                    print(f"Error loading CSV: {str(e)}")
                    return jsonify({"error": f"Error loading CSV file: {str(e)}"}), 400

                # Clean the data
                cleaned_data = clean_data(data)

                if cleaned_data is not None:
                    # Save cleaned data
                    cleaned_data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_monthlySale.csv')
                    cleaned_data.to_csv(cleaned_data_path, index=False)
                    print(f"Saved cleaned data to {cleaned_data_path}")

                    # Train model for best accuracy
                    try:
                        print("Starting model training...")

                        # Always use the best accuracy parameters (no memory optimization)
                        association_rules = model_training(cleaned_data, memory_optimized=False)

                        print(f"Model training completed successfully. Generated {len(association_rules)} rules.")
                    except Exception as e:
                        print(f"Error during model training: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                        return jsonify({"error": f"Error during model training: {str(e)}"}), 500

                    # Get custom model filename if provided, otherwise use default
                    model_filename = request.form.get('model_filename', '').strip()
                    if not model_filename:
                        model_filename = 'trained_model.pkl'
                    elif not model_filename.endswith('.pkl'):
                        model_filename += '.pkl'

                    # Save model with the specified filename
                    model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
                    with open(model_path, 'wb') as file:
                        pickle.dump(association_rules, file)
                    print(f"Saved trained model to {model_path}")

                    # If using a custom filename, also save a copy as the default filename for the recommendation endpoint
                    if model_filename != 'trained_model.pkl':
                        default_model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'trained_model.pkl')
                        with open(default_model_path, 'wb') as file:
                            pickle.dump(association_rules, file)
                        print(f"Also saved a copy to {default_model_path} for the recommendation endpoint")

                    # Remove temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    # Reload the data for the recommendation endpoint
                    data = cleaned_data

                    return jsonify({
                        "success": True,
                        "message": "File uploaded and model trained successfully",
                        "rows_processed": len(cleaned_data),
                        "rules_generated": len(association_rules),
                        "model_filename": model_filename
                    })
                else:
                    return jsonify({"error": "Data cleaning failed"}), 400
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                print(f"Error during processing: {str(e)}")
                print(f"Traceback: {error_traceback}")
                return jsonify({"error": str(e), "traceback": error_traceback}), 500
        else:
            print(f"Error: File type not allowed. Filename: {file.filename}")
            return jsonify({"error": "File type not allowed. Only CSV files are accepted."}), 400

    # If GET request, return the upload form
    return render_template('upload.html')

if __name__ == '__main__':
    print("Starting Flask application...")
    # Configure Flask with extended timeout settings
    app.run(
        host='0.0.0.0',
        debug=True,
        port=2000,
        threaded=True  # Enable threading for better handling of concurrent requests
    )



