import gradio as gr
from fastai.vision.all import *
import skimage

import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner('export.pkl')
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred, idx, probs = learn.predict(img)
    return{labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Breed Classifier"
description = "A animal/pet Breed Classifier"
examples = ['cat.jpg.webp']
interpretation = 'default'
enable_queue = True

gr.Interface(fn = predict, inputs = gr.Image(), outputs = gr.Label(num_top_classes = 3), title = title, description = description , examples = examples).launch(share=True)
