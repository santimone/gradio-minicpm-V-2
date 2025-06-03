from typing import Optional, Union, List
from transformers.processing_utils import ProcessorMixin
from transformers import AutoTokenizer, CLIPImageProcessor


class MiniCPMProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"
    image_processor_class = "CLIPImageProcessor"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        images,
        text: Union[str, List[str]] = None,
        return_tensors=None,
        **kwargs
    ):
        inputs = self.image_processor(images, return_tensors=return_tensors)
        if text is not None:
            text_inputs = self.tokenizer(
                text, return_tensors=return_tensors, padding=True, truncation=True
            )
            inputs.update(text_inputs)
        return inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16", **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(image_processor=image_processor, tokenizer=tokenizer)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
