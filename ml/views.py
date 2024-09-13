from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import numpy as np
from PIL import Image
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import tensorflow as tf
from django.conf import settings

# Load the model once when the server starts
model_path = os.path.join(settings.BASE_DIR, 'model/Seq_model_1000.h5')
print(f"Loading model from: {model_path}")  # Print model path for verification
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")


@csrf_exempt
def index(request):
    predicted_class_name = None
    if request.method == 'POST':
        print("Received POST request.")

        if 'image' in request.FILES:
            print("Image file found in request.")
            image_file = request.FILES['image']

            try:
                img = Image.open(image_file)
                print(f"Image opened successfully. Format: {img.format}, Size: {img.size}, Mode: {img.mode}")

                img = img.resize((224, 224))
                print("Image resized to 224x224.")

                img_array = np.array(img)
                print(f"Image converted to numpy array. Shape: {img_array.shape}")

                if img_array.shape[-1] != 3:
                    print("Image does not have 3 channels. Converting to RGB.")
                    img_array = np.stack([img_array] * 3, axis=-1)

                img_array = np.expand_dims(img_array, axis=0)
                print(f"Image array shape after expanding dimensions: {img_array.shape}")

                try:
                    prediction = model.predict(img_array)
                    print(f"Prediction array: {prediction}")

                    # Assuming binary classification with output in [0, 1]
                    predicted_prob = prediction[0][0]
                    print(f"Predicted probability: {predicted_prob}")

                    # Threshold to decide class
                    threshold = 0.5
                    if predicted_prob > threshold:
                        predicted_class_name = "fake"
                    else:
                        predicted_class_name = "real"

                    print(f"Predicted class name: {predicted_class_name}")

                except Exception as e:
                    print(f"Error during model prediction: {e}")
                    return render(request, 'ml/index.html', {'error': 'Prediction failed. Please try again.'})

            except Exception as e:
                print(f"Error opening or processing image: {e}")
                return render(request, 'ml/index.html', {'error': 'Image processing failed. Please try again.'})

        else:
            print("No image file provided in request.")
            return render(request, 'ml/index.html', {'error': 'No image file provided.'})

    print("GET request or other method received.")
    return render(request, 'ml/index.html', {'predicted_class_name': predicted_class_name})
