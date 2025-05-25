import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array


IMG_SIZE   = 128  
MODEL_PATH = 'models/model_02.h5'
CLASS_NAMES = [
    'Actinic keratoses and intraepithelial carcinoma',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic nevi',
    'Vascular lesions'
]


model = load_model(MODEL_PATH)


def predict_image(file_path):
    img = load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], preds[idx]


root = tk.Tk()
root.title(" Skin Disease Classifier Model 02 ")
root.geometry("800x800")
root.resizable(False, False)

def open_image():
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not path:
        return

    
    img = Image.open(path)
    img.thumbnail((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

  
    label, conf = predict_image(path)
    result_label.config(
        text=f"Result: {label}\nConfidence: {conf*100:.1f}%"
    )


btn = tk.Button(root, text="📂 Select Image", command=open_image)
btn.pack(pady=10)


img_label = Label(root)
img_label.pack(pady=10)


result_label = Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
