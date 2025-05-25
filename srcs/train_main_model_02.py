import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


IMG_DIR     = 'images'
CSV_PATH = 'data/HAM10000_metadata.csv'
IMG_SIZE    = 128        
BATCH_SIZE  = 32
NUM_CLASSES = 7
EPOCHS      = 30       


df = pd.read_csv(CSV_PATH)
df = df[df['dx'].isin(df['dx'].value_counts().index[:NUM_CLASSES])]
df['filename'] = df['image_id'] + '.jpg'
le = LabelEncoder()
df['label'] = le.fit_transform(df['dx'])


train_df = df.sample(frac=0.8, random_state=42)
val_df   = df.drop(train_df.index)
cw = class_weight.compute_class_weight('balanced',
                                       classes=np.unique(train_df['label']),
                                       y=train_df['label'])
class_weights = dict(enumerate(cw))


train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1./255)

train_flow = train_gen.flow_from_dataframe(
    train_df, directory=IMG_DIR,
    x_col='filename', y_col='dx',
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical'
)
val_flow = val_gen.flow_from_dataframe(
    val_df, directory=IMG_DIR,
    x_col='filename', y_col='dx',
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)


base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE,IMG_SIZE,3),
    include_top=False, weights='imagenet'
)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(base.input, outputs)


for layer in base.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)


history1 = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[es, rlp],
    verbose=1
)


for layer in base.layers[-20:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history2 = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS//2,
    class_weight=class_weights,
    callbacks=[es, rlp],
    verbose=1
)


loss, acc = model.evaluate(val_flow, verbose=0)
print(f"\nFinal Accuracy: {acc:.2f}")

y_pred = np.argmax(model.predict(val_flow), axis=1)
y_true = val_flow.classes
print("\nClassification Report:")
print(classification_report(
    y_true, y_pred, target_names=le.classes_
))


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history1.history['accuracy'] + history2.history['accuracy'], label='Train Acc')
plt.plot(history1.history['val_accuracy'] + history2.history['val_accuracy'], label='Val Acc')
plt.legend(), plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history1.history['loss'] + history2.history['loss'], label='Train Loss')
plt.plot(history1.history['val_loss'] + history2.history['val_loss'], label='Val Loss')
plt.legend(), plt.title('Loss')
plt.tight_layout(), plt.show()


model.save("models/model_02.h5")
print("model_02.h5")
