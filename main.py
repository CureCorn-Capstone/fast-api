from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
from fastapi.responses import HTMLResponse

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("model.h5")

CLASS_NAMES = ["Leaf Spot", 'Leaf Blight', 'Common Rust', 'Sehat']
MESSAGE_NAMES = ["Bercak", 'Hawar', 'Karat', 'Sehat']

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <html>
    <head>
        <title>CureCorn App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #fff;
            margin: 0;
            padding: 0;
        }

        .container {
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
        }

        h1 {
            text-align: center;
            color: #333;
            font-size: 32px;
            margin-bottom: 20px;
        }

        .nav {
            margin-bottom: 40px;
            text-align: center;
        }

        .nav a {
            margin-right: 20px;
            text-decoration: none;
            color: #333;
            font-size: 20px;
        }

        .form-container {
            text-align: center;
        }

        .form-container input[type="file"] {
            margin-bottom: 20px;
            display: block;
            width: 100%;
            padding: 15px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 18px;
        }

        .form-container button {
            padding: 15px 30px;
            background-color: #A0D8B3;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 18px;
            transition: background-color 0.3s ease;
        }

        .form-container button:hover {
            background-color: #47A992;
        }
    </style>
    </head>
    <body>
        <div class="container">
            <h1>Welcome to CureCorn Web Server App!</h1>
            <div class="nav">
                <a href="/docs">API Documentation</a>
            </div>
            <div class="form-container">
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept=".jpg,.jpeg,.png">
                    <br>
                    <button type="submit">Upload and Predict</button>
                </form>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img = cv2.resize(image,(150,150))
    img_batch = np.expand_dims(img, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class_index = np.argmax(predictions[0])
    predicted_class = MESSAGE_NAMES[predicted_class_index]
    predicted_message = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(predictions[0]))

    if predicted_class_index == 0:
        condition = "Sehat"
        message = "Daun jagung anda tidak terdeteksi penyakit"
    else:
        condition = "Daun jagung anda terdeteksi " + predicted_class.lower()
        message = predicted_message

    response = {
        'Condition': condition,
        'Message': message,
        'Confidence': confidence
    }

    return response

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)