import torch
from transformers import AutoModel
from processing_minicpm import MiniCPMProcessor
import gradio as gr
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained("openbmb/MiniCPM-V-2", trust_remote_code=True).to(device)
processor = MiniCPMProcessor.from_pretrained("openbmb/MiniCPM-V-2", trust_remote_code=True)

def predict(image: Image.Image, prompt: str):
    if not prompt:
        return "Por favor ingrese una prompt."

    # build the lists the model.generate API expects
    data_list = [prompt]
    # each entry in img_list must be an iterable of images
    img_list = [[image]] if image is not None else None

    with torch.no_grad():
        outputs = model.generate(
            data_list=data_list,
            img_list=img_list,
            tokenizer=processor.tokenizer,
            max_new_tokens=100
        )

    # Debug: print what outputs looks like
    print(f"Type of outputs: {type(outputs)}")
    print(f"Length of outputs: {len(outputs) if hasattr(outputs, '__len__') else 'No length'}")
    print(f"Type of outputs[0]: {type(outputs[0])}")
    print(f"First few elements of outputs[0]: {outputs[0][:10] if hasattr(outputs[0], '__getitem__') else outputs[0]}")

    # Check if outputs[0] is already a string
    if isinstance(outputs[0], str):
        return outputs[0]
    
    # decode the first generated sequence
    return processor.decode(outputs[0], skip_special_tokens=True)

gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Prompt")],
    outputs="text",
    title="Demo del modelo multimodal MiniCPM-V-2"
).launch()
