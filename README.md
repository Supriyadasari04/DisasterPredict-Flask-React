# Disaster Predict (Flask-React)

DisasterPredict is a full-stack application for predicting disaster types based on user-provided inputs. The backend is built using Flask and utilizes a pre-trained machine learning model (stored as Joblib files) to make predictions. The frontend is built with React, providing an interactive user interface to submit disaster-related parameters and view prediction results.

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
  - [Backend (Flask)](#backend-flask)
  - [Frontend (React)](#frontend-react)
- [Usage](#usage)
  - [Running the Flask Backend](#running-the-flask-backend)
  - [Running the React Frontend](#running-the-react-frontend)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project is divided into two main components:

- **Flask Backend:**  
  - Loads a pre-trained machine learning model along with scaler and other preprocessing objects.
  - Provides an API endpoint (`/predict`) which accepts disaster-related input parameters, processes the data, and returns a prediction.
  - Maps the numeric output from the model to a human-readable disaster name.

- **React Frontend:**  
  - Presents a user-friendly form to input parameters such as Year, Magnitude Scale, Disaster Magnitude Value, Longitude, Latitude, and Country.
  - Submits the data to the Flask backend and displays the prediction result on the UI.

## Folder Structure

The project is organized as follows:

```
DisasterPredict-Flask-React/
├── backend/
│   ├── app.py                   # Flask application entry point
│   ├── requirements.txt         # Python dependencies for Flask
│   ├── ml/                      # Folder containing model files (Joblib)
│   │   ├── random_forest_model.joblib
│   │   └── scaler.joblib
│   └── venv/                    # Python virtual environment (usually not committed)
└── frontend/
    ├── node_modules/            # Installed Node.js dependencies
    ├── public/                  # Public assets for the React app
    ├── src/
    │   ├── App.js               # Main React component file
    │   ├── App.css              # Styling for the React app
    │   ├── index.js             # Entry point for the React app
    │   └── components/
    │         └── DisasterPrediction.jsx  # React component for disaster prediction
    ├── package.json             # Node.js dependencies and scripts
    └── package-lock.json        # Lock file for Node.js dependencies
```

## Installation

### Backend (Flask)

1. **Navigate to the backend folder:**

   ```bash
   cd DisasterPredict-Flask-React/backend
   ```

2. **Create and activate a Python virtual environment:**

   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the required Python packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Place your model files (.joblib) in the `ml` folder.**  
   Ensure that `random_forest_model.joblib` and `scaler.joblib` are available in the `backend/ml` directory.

### Frontend (React)

1. **Navigate to the frontend folder:**

   ```bash
   cd DisasterPredict-Flask-React/frontend
   ```

2. **Install the Node.js dependencies:**

   ```bash
   npm install
   ```

   *Note: Ensure you have [Node.js](https://nodejs.org/) installed on your machine.*

## Usage

### Running the Flask Backend

1. **Activate your virtual environment in the `backend` folder (if not already activated).**
2. **Run the Flask application:**

   ```bash
   python app.py
   ```

   The Flask backend will start on [http://localhost:5000](http://localhost:5000).  
   The API endpoint `/predict` is used for making predictions.

### Running the React Frontend

1. **Open a new terminal and navigate to the `frontend` folder:**

   ```bash
   cd DisasterPredict-Flask-React/frontend
   ```

2. **Start the React development server:**

   ```bash
   npm start
   ```

   The React app will open in your default browser at [http://localhost:3000](http://localhost:3000).

### Using the Application

- **Input Data:**  
  Use the form in the React application to enter details such as:
  - **Year:** e.g., 2020
  - **Magnitude Scale:** Choose from options like "0: Km²", "1: Richter Scale", etc.
  - **Disaster Magnitude Value:** e.g., 6.5
  - **Longitude & Latitude:** Location coordinates
  - **Country:** Select from the provided list

- **Prediction:**  
  Click on **Predict Disaster**. This sends a POST request to the Flask backend (`/predict`) with your input data. The backend processes the data, makes a prediction, and maps the numeric label to a disaster name.

- **Result:**  
  The prediction result is returned as JSON and displayed on the React frontend.

## API Endpoints

### `POST /predict`

**Description:**  
Accepts disaster-related input data in JSON format, processes it through the pre-trained machine learning model, and returns the predicted disaster type.

**Expected JSON Payload:**

```json
{
  "year": 2020,
  "mag_scale_index": 1,
  "dis_mag_value": 6.5,
  "country_code_index": 0,
  "longitude": 85.324,
  "latitude": 27.7172
}
```

**Response Example:**

```json
{
  "predicted_label": 2,
  "predicted_disaster_name": "Earthquake"
}
```

## Contributing

Contributions are welcome! Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your fork and submit a pull request.

## License

This README file provides an in-depth explanation of the project’s structure, installation steps, usage instructions, and contribution guidelines. Customize it further if needed to better suit your project details.