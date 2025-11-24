import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ---------------------------
# PATH SETUP
# ---------------------------
DATASET_PATH = r"C:\Users\user\OneDrive\Desktop\cv-flipped-project\dataset\chest_xray"

train_dir = os.path.join(DATASET_PATH, "train")
test_dir = os.path.join(DATASET_PATH, "test")
val_dir = os.path.join(DATASET_PATH, "val")

print("Train classes:", os.listdir(train_dir))
print("Test classes:", os.listdir(test_dir))
print("Val classes:", os.listdir(val_dir))

# ---------------------------
# IMAGE GENERATORS
# ---------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ---------------------------
# SIMPLE CNN MODEL
# ---------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # NORMAL vs PNEUMONIA
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())
print("Model compiled successfully!")

# ---------------------------
# TRAIN FOR 1 EPOCH (TEST RUN)
# ---------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=1
)

print("Training completed!")


model.save("chest_xray_model.h5")
print("Model saved successfully!")
