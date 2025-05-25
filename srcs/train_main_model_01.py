import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.utils import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


IMG_DIR = 'images'
CSV_PATH = 'data/HAM10000_metadata.csv'
IMG_SIZE = 64  
NUM_CLASSES = 7


df = pd.read_csv(CSV_PATH)
df = df[df['dx'].isin(df['dx'].value_counts().index[:7])]  


le = LabelEncoder()
df['label'] = le.fit_transform(df['dx'])


images, labels = [], []
for _, row in df.iterrows():
    img_path = os.path.join(IMG_DIR, row['image_id'] + '.jpg')
    try:
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(row['label'])
    except:
        continue

images = np.array(images)
labels = np.array(labels)


labels_cat = to_categorical(labels, NUM_CLASSES)


X_train, X_test, y_train, y_test = train_test_split(images, labels_cat, test_size=0.2, random_state=42)


y_train_int = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train_int),
                                                  y=y_train_int)
class_weights = dict(enumerate(class_weights))


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


history = model.fit(X_train, y_train,epochs=40,validation_data=(X_test, y_test), batch_size=32,class_weight=class_weights,callbacks=[early_stop])


loss, acc = model.evaluate(X_test, y_test)
print(f"\n Accuracy: {acc:.2f}")


y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()


model.save("models/model_01.h5")
print("\n Model saved as 'model_01.h5'")