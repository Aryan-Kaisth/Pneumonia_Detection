import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = r"/home/aryan/AI/Project C/Uploaded Images"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("/home/aryan/AI/Project C/notebooks/model.keras")  # Replace with your model
img_size = (256, 256)
class_names = ["Normal", "Pneumonia"]

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(img_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            processed_img = preprocess(save_path)
            prediction = model.predict(processed_img)[0][0]
            label_index = int(prediction > 0.5)
            result = class_names[label_index]
            confidence = round(float(prediction) * 100 if label_index else (1 - prediction) * 100, 2)

    return render_template("index.html", result=result, confidence=confidence, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
