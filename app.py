from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your trained ResNet50 model
model_path = os.path.join('model', 'skin_burn_classification_model.keras')
model = load_model(model_path)

# Ensure the uploads directory exists
uploads_dir = os.path.join('static', 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

# Define a function to preprocess images
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale as done during training
    return img_array

# Define route for homepage
@app.route('/')
def home():
    return render_template('index.html', uploaded_image=None)

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("No file part in the request.")
        return render_template('index.html', uploaded_image=None)

    file = request.files['image']
    if file.filename == '':
        print("No selected file.")
        return render_template('index.html', uploaded_image=None)

    img_path = os.path.join(uploads_dir, file.filename)
    file.save(img_path)
    print(f"File saved at: {img_path}")

    # Preprocess the image and make prediction
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    result = np.argmax(prediction)
    severity = ['1st Degree', '2nd Degree', '3rd Degree', 'No Burn']

    # Pass the image path and prediction result to the result template
    return render_template('result.html', prediction=severity[result], uploaded_image=url_for('static', filename=f'uploads/{file.filename}'))

if __name__ == '__main__':
    app.run(debug=True)
