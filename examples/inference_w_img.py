import sys
sys.path.append('../')

from scripts.inference import inference_once

model_path = "mtgv/MobileVLM_V2-7B"
image_file = "../assets/samples/test11.png"
# prompt_str = "Who is the author of this book?\nAnswer the question using a single word or phrase."
prompt_str = 'Please point out to me which objects are in this picture. Give me their specific names.'
# prompt_str = 'Do not put the apple next to the banana.'
# (or) What is the title of this book?
# (or) Is this book related to Education & Teaching?

args = type('Args', (), {
    "model_path": model_path,
    "image_file": image_file,
    "prompt": prompt_str,
    "conv_mode": "v1",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "load_8bit": False,
    "load_4bit": False,
})()

inference_once(args)