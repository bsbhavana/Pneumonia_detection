from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
model = load_model('model_vgg16.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.  # Rescale to [0, 1]
    return x

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'})
    
    file = request.files['file']
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)
    
    processed_image = preprocess_image(img_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    
    if predicted_class == 0:
        result = 'Normal'
    else:
        result = 'Pneumonia'
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)

