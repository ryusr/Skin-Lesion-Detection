import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
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

def predict_image(path):
    img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], preds[idx]

def select_folder():
    folder = filedialog.askdirectory()
    if not folder:
        return
    jpgs = glob.glob(os.path.join(folder, '*.jpg'))
    if not jpgs:
        messagebox.showwarning("No Images", "No .jpg files found")
        return
    result_text.delete('1.0', tk.END)
    for fp in jpgs:
        img = Image.open(fp)
        img.thumbnail((150, 150))
        photo = ImageTk.PhotoImage(img)
        img_label.configure(image=photo)
        img_label.image = photo
        lbl, conf = predict_image(fp)
        result_text.insert(tk.END, f"{os.path.basename(fp)} -> {lbl} ({conf*100:.1f}%)\n")
        root.update()

root = tk.Tk()
root.title(" Folder Skin Disease Tester ")
root.geometry("1200x600")

img_label = tk.Label(root)
img_label.pack(pady=5)

btn = tk.Button(root, text="Select Folder", command=select_folder)
btn.pack(pady=5)

result_text = tk.Text(root)
result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

root.mainloop()
