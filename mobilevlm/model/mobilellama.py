from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from mobilevlm.model.mobilevlm import MobileVLMMetaModel, MobileVLMMetaForCausalLM


class MobileVLMConfig(LlamaConfig):
    model_type = "mobilevlm"


class MobileLlamaModel(MobileVLMMetaModel, LlamaModel):
    config_class = MobileVLMConfig

    def __init__(self, config: LlamaConfig):
        super(MobileLlamaModel, self).__init__(config)


class MobileLlamaForCausalLM(LlamaForCausalLM, MobileVLMMetaForCausalLM):
    config_class = MobileVLMConfig #cls.config_class.from_pretrained中被调用

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MobileLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()  # Initialize weights and apply final processing

    def get_model(self):
        return self.model

    '''
    outputs = self(
    **model_inputs,
    return_dict=True,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
)

    调用 __call__ 方法：
    self(...) 触发了模型的 __call__ 方法。

    执行预处理和钩子：
        在 __call__ 方法中，PyTorch 会执行一些预处理，可能包括运行注册的前向钩子。

    调用 forward 方法：
        然后，__call__ 方法会调用 forward 方法，并将所有传入的参数传递给它。

    执行后处理和钩子：
        在 forward 方法返回后，__call__ 方法可能会执行一些后处理，包括运行注册的后向钩子。

    返回结果：
        最后，__call__ 方法返回 forward 方法的输出。
    
    '''
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        '''
        use_cache = true
        output_attentions = false
        output_hidden_states = false
        return_dict = true
        attention_mask = tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')
        past_key_values = 
        image = (1,3,336,336)
        其余none
        '''
        #id就是id，但是attention mask是加了图片token的，id里面图片token只有一个，就算200
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0] #（1，205，2048）
        logits = self.lm_head(hidden_states) #（1，205，32000）相当于分类，输出205个分词，每205行中的一行都是在对32000个词做分类任务

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous() #contiguous() 方法用于确保张量在内存中是连续存储的，这通常在执行切片操作后需要做的。切片操作可能会导致张量在内存中变得不连续，而某些 PyTorch 操作要求张量在内存中是连续的。.contiguous() 确保张量满足这一要求
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # loss：
    #     loss 是模型在给定输入和目标时计算出的损失值。在训练过程中，这个损失值用于指导模型参数的优化。
    #
    # logits：
    #     logits 是模型输出的原始得分，通常是最后一个线性层的输出。在处理语言模型时，logits 通常被解释为每个词汇对应的预测得分，可以通过 softmax 函数转换为概率分布。
    #
    # past_key_values：
    #     past_key_values 包含了模型中注意力层的历史信息。在自回归模型中，这些历史信息（或称为缓存）被用于加速连续的令牌生成，因为模型不需要重新计算先前令牌的隐藏状态和注意力得分。
    #
    # hidden_states：
    #     hidden_states 是模型每一层的输出表示，可以用于深入分析模型的工作机制，例如在特征提取或模型解释性分析中。
    #
    # attentions：
    #     attentions 是模型的注意力矩阵，提供了模型每一层中每个头的注意力得分。这对于理解模型如何关注不同部分的输入特别有用。

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:] #保留最后一列，仍然是二维数组

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("mobilevlm", MobileVLMConfig) #有点像软链接，MobileVLMConfig 类注册到 AutoConfig 的注册表中，并与键 "mobilevlm" 相关联
AutoModelForCausalLM.register(MobileVLMConfig, MobileLlamaForCausalLM) #当用户使用 MobileVLMConfig 配置类实例化模型时，AutoModelForCausalLM 知道它需要创建和返回一个 MobileLlamaForCausalLM 模型实例
