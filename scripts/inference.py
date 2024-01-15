import sys
import torch
import argparse
from PIL import Image
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


def inference_once(args):

    """
    disable_torch_init(): 修改PyTorch中某些层的默认初始化行为，以加快模型创建的速度.
    将Linear层的reset_parameters方法替换为一个什么也不做的lambda函数。通常，reset_parameters方法用于初始化层的权重和偏置
    修改了PyTorch的LayerNorm层，将其reset_parameters方法替换为一个空的lambda函数。LayerNorm层通常用于归一化神经网络中的激活值
    """
    disable_torch_init()
    model_name = args.model_path.split('/')[-1]
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path)

    images = [Image.open(args.image_file).convert("RGB")] #PIL Image 318*500
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16) #返回处理完毕的（resize crop等）图片tensor
    # conv是conversation
    conv = conv_templates[args.conv_mode].copy() #Conversation(system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", roles=('USER', 'ASSISTANT'), messages=[], offset=0, sep_style=<SeparatorStyle.TWO: 2>, sep=' ', sep2='</s>', version='v1', skip_next=False)
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + args.prompt) #加入到conv类的message属性中： ['USER', '<image>\nWho is the author of this book?\nAnswer the question using a single word or phrase.']
    conv.append_message(conv.roles[1], None)  #conv.roles[0]： user   conv.roles[0]：assistance即为模型的输入， inference为none
    prompt = conv.get_prompt()  #合并起来 A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image> Who is the author of this book?Answer the question using a single word or phrase. ASSISTANT:
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2 #'</s>' 什么时候结束输入
    # Input
    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    # Inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    # Result-Decode
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    print(f"🚀 {model_name}: {outputs.strip()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="mtgv/MobileVLM-1.7B")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    inference_once(args)
