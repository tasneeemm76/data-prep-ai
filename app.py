from flask import Flask, jsonify, request, render_template, send_from_directory
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import logging

app = Flask(__name__)

# Set folder paths
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed_data'

# Ensure folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@app.route('/')
def home():
    return render_template('index.html', log_details=None, download_link=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        return render_template('index.html', log_details="Error: No file provided.", download_link=None), 400
    
    # Check if the file is a CSV
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext != '.csv':
        return render_template('index.html', log_details="Error: Only CSV files are supported.", download_link=None), 400
    
    # Save the file to the uploads folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Read the file into a pandas DataFrame
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return render_template('index.html', log_details=f"Error: Failed to read file: {str(e)}", download_link=None), 400
    
    log_details = "Processing log:\n"
    
    # Check columns and data types
    log_details += f"Columns: {df.columns.tolist()}\n"
    log_details += f"Data types: {df.dtypes.tolist()}\n"

    # Handle Missing Values
    missing_values_columns = df.columns[df.isnull().any()].tolist()
    if missing_values_columns:
        log_details += f"Missing values found in columns: {missing_values_columns}\n"
        for col in missing_values_columns:
            df[col].fillna(df[col].mode()[0] if df[col].dtype == 'object' else df[col].mean(), inplace=True)
        log_details += "Missing values filled.\n"
    else:
        log_details += "No missing values found.\n"
    
    # Remove Duplicates
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - df.shape[0]
    log_details += f"Removed {removed_duplicates} duplicate rows.\n"
    
    # Handle Inconsistent Data Types (example: categorical columns encoded as integers)
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dtype != 'object':
            log_details += f"Column {col} has inconsistent data type, encoding it.\n"
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
    
    
    # Save the processed DataFrame in the processed_data folder
    processed_filename = f"processed_{file.filename}"
    processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    df.to_csv(processed_file_path, index=False)
    
    # Provide download link and logs
    download_link = f"/download/{processed_filename}"
    return render_template('index.html', log_details=log_details, download_link=download_link)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

@app.route('/more_processing')
def more_processing():
    return render_template('x.html')

@app.route('/upload_x', methods=['POST'])
def upload_x():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        # Read the uploaded CSV file
        df = pd.read_csv(file)
        log = []
        
        # Data Type Conversion (Detect datetime, categorical features)
        for column in df.columns:
            if df[column].dtype == 'object':
                try:
                    df[column] = pd.to_datetime(df[column])
                    log.append(f"Converted '{column}' to datetime.")
                except:
                    # If not datetime, try encoding categorical columns
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column].astype(str))
                    log.append(f"Encoded '{column}' as categorical.")

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')  # Mean imputation for numerical features
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        log.append("Imputed missing values with mean for numerical columns.")

        # Feature Scaling (Standardization/Normalization)
        scaler = StandardScaler()
        df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
        log.append("Standardized numerical features.")

        # Handling Outliers (Remove outliers based on IQR for each numerical feature)
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]
        log.append("Removed outliers based on IQR for numerical features.")

        # Feature Selection (Using Random Forest to select features)
        X = df.drop(columns='target', errors='ignore')  # Assume 'target' is the column to predict
        y = df['target'] if 'target' in df.columns else None
        
        if y is not None:
            model = RandomForestClassifier()
            model.fit(X, y)
            selector = SelectFromModel(model, threshold="mean")
            X_selected = selector.transform(X)
            df = pd.DataFrame(X_selected, columns=[X.columns[i] for i in range(len(X.columns)) if selector.get_support()[i]])
            log.append("Selected features based on Random Forest importance.")

        # Handle Class Imbalance (Resampling minority class)
        if y is not None:
            df_minority = df[df['target'] == 1]
            df_majority = df[df['target'] == 0]
            df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=123)
            df_balanced = pd.concat([df_majority, df_minority_upsampled])
            df = df_balanced
            log.append("Handled class imbalance by oversampling the minority class.")

        # Splitting the data (80% train, 20% test)
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target'), df['target'], test_size=0.2, random_state=42)
            log.append("Split data into 80% training and 20% testing.")

        # Save the processed data
        processed_file_path = os.path.join('processed_data', file.filename)
        df.to_csv(processed_file_path, index=False)
        
        log_message = "\n".join(log)
        return jsonify({
            "log": log_message,
            "download_link": f"/download/{file.filename}"
        })


if __name__ == '__main__':
    if not os.path.exists('processed_data'):
        os.mkdir('processed_data')  # Create folder to save processed data if it doesn't exist
    app.run(debug=True)


