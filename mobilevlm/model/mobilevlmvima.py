from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from mobilevlm.model.mobilevlm import MobileVLMMetaModel, MobileVLMMetaForCausalLM
from .policy.policy import PolicyModel


class MobileVLMConfig(LlamaConfig):
    model_type = "mobilevlm"


class MobileLlamaModel(MobileVLMMetaModel, LlamaModel):
    config_class = MobileVLMConfig

    def __init__(self, config: LlamaConfig):
        super(MobileLlamaModel, self).__init__(config)


class MobileLlamaForCausalLM(LlamaForCausalLM, MobileVLMMetaForCausalLM):
    config_class = MobileVLMConfig  # cls.config_class.from_pretrained中被调用

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MobileLlamaModel(config)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.policy = PolicyModel(config.hidden_size)
        self.post_init()  # Initialize weights and apply final processing

    def get_model(self):
        return self.model

    def get_discretize(self, action):
        return self.policy.discretize_action(action)

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
    ) -> Dict[str, Union[torch.Tensor, Any]]:

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
        labels_placeholder = None
        input_ids, attention_mask, past_key_values, inputs_embeds, labels_None = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels_placeholder, images)

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

        hidden_states = outputs[0]  # （1，205，2048）

        action = self.get_discretize(labels)
        # cache target action
        tar_action = {k: v.clone() for k, v in action.items()}

        predicted_action_tokens = hidden_states[:,-1,:].unsqueeze(1)  # (1,1,2048) For 3B, the hidden_states is 2560
        dist_dict = self.policy.forward_action_decoder(predicted_action_tokens)   #到这里是一个类
        actions = {k: v.mode() for k, v in dist_dict.items()} # change to {'pose0_position': tensor([[[16, 35]]]), 'pose0_rotation': tensor([[[25, 25, 25, 49]]]), 'pose1_position': tensor([[[12, 86]]]), 'pose1_rotation': tensor([[[25, 25, 49, 19]]])}
        action_tokens = self.policy.forward_action_token(actions)  # (1, B, E) (1,1,768)
        action_tokens = action_tokens.squeeze(0)  # (B, E) (1,768)
        actions = self.policy._de_discretize_actions(actions)  # output two groups of action position ratio and rotation
        loss = self.compute_cross_entropy_loss(actions, tar_action)

        return {'loss': loss, 'actions':actions, 'action_tokens':action_tokens}

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

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

    def compute_cross_entropy_loss(self, pred_actions, tar_action):
        import torch.nn.functional as F

        total_loss = 0
        for key in pred_actions.keys():
            pred = pred_actions[key]
            true = tar_action[key]
            total_loss += F.cross_entropy(pred, true)
        return total_loss


AutoConfig.register("mobilevlm", MobileVLMConfig)  # 有点像软链接，MobileVLMConfig 类注册到 AutoConfig 的注册表中，并与键 "mobilevlm" 相关联
AutoModelForCausalLM.register(MobileVLMConfig,
                              MobileLlamaForCausalLM)  # 当用户使用 MobileVLMConfig 配置类实例化模型时，AutoModelForCausalLM 知道它需要创建和返回一个 MobileLlamaForCausalLM 模型实例
