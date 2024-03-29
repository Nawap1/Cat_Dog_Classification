from fastapi import FastAPI
from fastapi import UploadFile,File
from io import BytesIO
import uvicorn
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from fastapi.responses import JSONResponse
classifier = load_model('resources/dogcat_model_bak.h5')
app = FastAPI()

@app.post("/predict")
async def predict_image(file:UploadFile = File(...)):
    try:
        uploaded_image = BytesIO(file.file.read())
        print("Image read successfully")
        # Rest of your code...
    except Exception as e:
        print("Error:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=400)

    img1 = image.load_img(uploaded_image, target_size=(64, 64))
    img = image.img_to_array(img1)
    img = img / 255

    # Make predictions
    img = np.expand_dims(img, axis=0)
    prediction = classifier.predict(img, batch_size=None, steps=1)  # gives all class prob.
    if (prediction[:, :] > 0.5):
        return {'prediction':'Dog'}
    else:
        return {'prediction':'Cat'}

