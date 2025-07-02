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

    try:
        with torch.no_grad():
            outputs = model.generate(
                data_list=data_list,
                img_list=img_list,
                tokenizer=processor.tokenizer,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )

        # Debug: print what outputs looks like
        print(f"Type of outputs: {type(outputs)}")
        print(f"Length of outputs: {len(outputs) if hasattr(outputs, '__len__') else 'No length'}")
        if len(outputs) > 0:
            print(f"Type of outputs[0]: {type(outputs[0])}")
            print(f"Content of outputs[0]: '{outputs[0]}'")
            print(f"Length of outputs[0]: {len(outputs[0]) if hasattr(outputs[0], '__len__') else 'No length'}")

        # The model returns strings directly, no need to decode
        if isinstance(outputs, list) and len(outputs) > 0:
            result = outputs[0].strip()
            if result:
                return result
            else:
                return "El modelo no generó una respuesta. Intente con una imagen y prompt diferentes."
        else:
            return "Error: No se pudo generar una respuesta."
            
    except Exception as e:
        print(f"Error during generation: {e}")
        return f"Error al generar respuesta: {str(e)}"

gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Prompt")],
    outputs="text",
    title="Demo del modelo multimodal MiniCPM-V-2"
).launch()
