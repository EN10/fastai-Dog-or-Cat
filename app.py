from fastai.vision.all import *
import gradio as gr

l2 = load_learner('model.lrn')
categories = ('Dog', 'Cat')

def classify_image(img):
    pred,idx,probs = l2.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(128, 128))
label = gr.outputs.Label()
examples = ['dog.jpg', 'cat.jpg', 'dunno.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch()

