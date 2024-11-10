# Human Skin Burn Classification

This repository contains a Flask application for classifying human skin burns into different severity levels using a trained machine learning model. The project includes a frontend with file upload functionality and displays the predicted burn severity.

## Folder Structure

- **data/**: Contains the dataset used for training the model.
- **static/uploads/**: Stores uploaded images for classification.
- **templates/**: Contains HTML templates for the frontend.
- **app.py**: The main Flask application file to run the server.
- **train.ipynb**: Jupyter Notebook for training the burn classification model.
- **requirements.txt**: Lists the dependencies required for the project.

## Setup and Usage

### 1. Install Dependencies

To install the required packages, run:

```bash
pip install -r requirements.txt
```

### 2. Create a Model Folder

Create a `model/` directory in the root folder to store the trained model:

```bash
mkdir model
```

### 3. Train the Model

Run `train.ipynb` to train the model. Once training is complete, save the model file in the `model/` directory. This file will be used for burn severity predictions in the Flask app.

### 4. Run the Application

Start the Flask application with:

```bash
python app.py
```

The app will be accessible at `http://localhost:5000`, where you can upload images and receive burn severity predictions.


