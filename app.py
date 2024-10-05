import os
import sys
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from keras.models import load_model

# Fuerzo codificación utf-8 en el entorno Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, 'strict')

# Cargo el modelo previamente entrenado
model_path = './model_weights.h5'
model = load_model(model_path)

# Creo una instancia de la aplicación FastAPI
app = FastAPI()

# Configuro CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permito todas las conexiones;
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Función para predecir la imagen
def predict_image(img):
    # Redimensiono la imagen
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalizo la imagen

    # Verifico y añado el canal de color (si es necesario)
    if img_array.ndim == 2:  # Imagen en escala de grises
        img_array = np.stack((img_array,) * 3, axis=-1)  # Convierto a 3 canales

    img_array = np.expand_dims(img_array, axis=0)  # Añado dimensión para batch size
    prediction = model.predict(img_array)
    return prediction[0][0]  # Devuelvo el valor de predicción

# Endpoint para la predicción
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(BytesIO(await file.read()))  # Abro la imagen desde el archivo subido
        prediction = predict_image(img)
        result = "Pneumonia" if prediction > 0.5 else "Normal"  # 0.5 es el umbral de decisión

        # Calculo el porcentaje de confianza
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

        return {
            "prediction": result,
            # "probability": float(prediction),
            "confidence": f"{confidence:.2f}%"  # Devuelvo el porcentaje con 2 decimales
        }

    except Exception as e:
        return {"error": str(e)}

# Página HTML simple para cargar imágenes
@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <html>
        <head>
            <title>Predicción de Neumonía</title>
        </head>
        <body>
            <h1>Sube una imagen para predecir neumonía</h1>
            <form action="/predict/" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*">
                <input type="submit" value="Predecir">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=content)
