import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = r"C:\Users\user\OneDrive\Desktop\cv-flipped-project\chest_xray_model.h5"
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)
print("Model loaded successfully!")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    class_index = np.argmax(preds)

    classes = ["NORMAL", "PNEUMONIA"]
    print(f"Prediction: {classes[class_index]}")

# UPDATE THIS PATH ↓↓↓
test_image_path = r"C:\Users\user\OneDrive\Desktop\cv-flipped-project\dataset\chest_xray\test\NORMAL\IM-0001-0001.jpeg"

predict_image(test_image_path)
