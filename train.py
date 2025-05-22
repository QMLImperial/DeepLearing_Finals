import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
DATA_DIR = "data/train"
IMG_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
MODEL_SAVE_PATH = "models/ship_classifier.h5"

# Label Mapping
label_map = {
    0: 'Cargo',
    1: 'Military',
    2: 'Carrier',
    3: 'Cruise',
    4: 'Tankers'
}

target_class_names = list(label_map.values())

# Load and filter DataFrame
df = pd.read_csv(CSV_PATH)
df = df[df['category'].isin(label_map.keys())]
df['category'] = df['category'].map(label_map)

# Image Preprocessing
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMG_DIR,
    x_col="image",
    y_col="category",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
    classes=target_class_names
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMG_DIR,
    x_col="image",
    y_col="category",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True,
    seed=42,
    classes=target_class_names
)

# Build Model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train Model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

# Save Model
os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")
