# Proyecto 7 Ciencia de datos e IA UDD - Franco Odone (API)

## Detección de Neumonía con FastAPI

### Descripción del Proyecto

La aplicación permite cargar una radiografía de tórax a través de un formulario HTML simple. Una vez cargada la imagen, el modelo predice si el paciente tiene neumonía o si está sano. El resultado de la predicción se acompaña de un porcentaje de confianza, basado en el valor de salida del modelo.

El modelo está entrenado para recibir imágenes de rayos X, que se redimensionan y normalizan antes de hacer la predicción.

### Requisitos del Sistema

- Python 3.7+
- FastAPI
- Keras con TensorFlow backend
- PIL (Pillow)
- NumPy
- Uvicorn (para ejecutar la aplicación)