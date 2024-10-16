#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Yanwei Li
# ------------------------------------------------------------------------

from typing import List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging
from transformers.generation.utils import GenerateOutput
from dataclasses import dataclass

from mgm.model.mgm_arch import MGMMetaModel, MGMMetaForCausalLM
from torch.nn import CrossEntropyLoss
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import ModelOutput

logger = logging.get_logger(__name__)

class MoDMGMConfig(LlamaConfig):
    model_type = "mod_mgm"
    def __init__(self,
                 mod_enable=True,
                 mod_mode='sparse',
                 mod_layers_idx=None,
                 capacity_factor=0.5,
                 router_aux_loss_coef=0.01,
                 **kwargs):
        self.mod = dict(
                mod_enable=mod_enable,
                mod_mode=mod_mode,
                mod_layers_idx=mod_layers_idx,
                capacity_factor=capacity_factor,
                router_aux_loss_coef=router_aux_loss_coef,
            )

        super(MoDMGMConfig, self).__init__(**kwargs)

class MoDMGMLlamaModel(MGMMetaModel, LlamaModel):
    config_class = MoDMGMConfig
    
    def __init__(self, config: LlamaConfig):
        super(MoDMGMLlamaModel, self).__init__(config)
@dataclass
class MoDBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mod_loss_list: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class MoDCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mod_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    mod_loss_list: Optional[Tuple[torch.FloatTensor]] = None
    
def LlamaDecoderLayer(self):
    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    return forward
def MoDLlamaDecoderLayer_forward(self):
    def forward(
        # self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = True,
        use_cache: Optional[bool] = False,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            capacity_factor(float): topk tokens will pass into this layer's transformer block
        """
        # print(labels[0,:])
        batch_size,seq_len,dim = hidden_states.shape
        ######### the next line should be modified once changed framework #########
        image_token_number = 2880 # depends on how you process image tokens
        img_and_instruction_token_len = image_token_number# naive solution,35 is the len of instruction tokens, image_token_number is img tokens(resolution)
        question_token_len = seq_len - img_and_instruction_token_len
        router_logits = self.router(self.input_layernorm(hidden_states))
        route_probabilities = F.softmax(router_logits, dim=-1)[:, :, 1]  # align with MOE
        ###################### mask for generation token #########################
        if question_token_len <= 0:
            ############# only text stage #############
            top_k = int(math.ceil(seq_len * 0.5))
            mod_token_prob_mask = torch.ones(batch_size, seq_len, dtype=hidden_states.dtype, device=hidden_states.device)
            mod_token_prob_mask[labels != -100] = 0
        else:
            mask_1 = torch.zeros(batch_size,img_and_instruction_token_len,dtype=hidden_states.dtype,device=hidden_states.device)
            mask_2 = torch.ones(batch_size,question_token_len,dtype=hidden_states.dtype,device=hidden_states.device)
            capacity_factor = 1 - ((img_and_instruction_token_len * 0.5 ) / seq_len)
            top_k = int(math.ceil(seq_len * capacity_factor))
            mod_token_prob_mask = torch.cat([mask_1, mask_2], dim=1)
        token_weights, token_index = torch.topk(route_probabilities + mod_token_prob_mask, top_k, dim=-1)
        selected_tokens, index = torch.sort(token_index, dim=1) # both are [bs, 223(topk of tokens)]
        r_weights = torch.gather(route_probabilities, dim=1, index=selected_tokens)
        indices_expanded = selected_tokens.unsqueeze(-1).expand(-1, -1, dim)
        selected_hidden_states = torch.gather(input=hidden_states, dim=1, index=indices_expanded)
        new_attention_mask = torch.gather(attention_mask, 1, selected_tokens)
        new_position_ids = torch.arange(0,top_k).unsqueeze(0).to(selected_hidden_states.device)
        residual = selected_hidden_states
        selected_hidden_states = self.input_layernorm(selected_hidden_states)
        selected_hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=selected_hidden_states,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        selected_hidden_states = residual + selected_hidden_states
        residual = selected_hidden_states
        selected_hidden_states = self.post_attention_layernorm(selected_hidden_states)
        selected_hidden_states = self.mlp(selected_hidden_states) * r_weights.unsqueeze(-1) # now router's weights is in gradient flow
        selected_hidden_states = (residual + selected_hidden_states)
        hidden_states = hidden_states.scatter(dim=1, index=indices_expanded, src=selected_hidden_states) # if hidden_states now is [bs,topk,dim]
        outputs = (hidden_states,)
        # outputs = (selected_hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        router_targets = torch.zeros_like(route_probabilities)
        for i in range(batch_size):
            router_targets[i, token_index[i]] = 1
        aux_loss = F.cross_entropy(router_logits.view(-1, 2), router_targets.view(-1).long())
        outputs += (aux_loss,)
        return outputs
    return forward
def MoDLlamaDecoderLayer_forward_inference(self):
    if not hasattr(self, 'total_tokens'):
        self.total_tokens = 0
    if not hasattr(self, 'routed_tokens'):
        self.routed_tokens = 0
    def forward(
        # self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = True,
        use_cache: Optional[bool] = False,
        capacity_factor :Optional[float] = 1.0,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            capacity_factor(float): topk tokens will pass into this layer's transformer block
        """

        batch_size,seq_len,dim = hidden_states.shape
        past_key_value_length = past_key_value.get_seq_length() if past_key_value is not None else 0
        router_logits = self.router(self.input_layernorm(hidden_states))
        route_probabilities = F.softmax(router_logits, dim=-1)[:, :, 1] # [bs,seq_len]
        self.total_tokens += seq_len
        # if seq_len <= 1 and route_probabilities[0,0] < 0.5:
        #     # # Update routed tokens count
        #     self.routed_tokens += seq_len
        #     outputs = (hidden_states,)
        #     if output_attentions:
        #         outputs += (None,)
        #     if use_cache:
        #         outputs += (past_key_value,)
        #     return outputs
        # else:
        if seq_len > 1:
            high_prob_mask = route_probabilities > 0.005
            batch_indices, token_indices = high_prob_mask.nonzero(as_tuple=True)
            selected_tokens = token_indices.view(1,batch_indices.size(0))
            selected_hidden_states = hidden_states[:, token_indices, :]
            tokens_chosen = selected_hidden_states.shape[1]  # Sum of chosen tokens per batch
            self.routed_tokens += (seq_len - int(tokens_chosen)) # Update routed tokens count for prefilling stage
            new_attention_mask = torch.zeros(batch_size, 1, int(tokens_chosen), int(tokens_chosen)).to(selected_hidden_states.device)
            upper_tri_indices = torch.triu_indices(row=int(tokens_chosen), col=int(tokens_chosen), offset=1)
            r_weights = torch.gather(route_probabilities, dim=1, index=selected_tokens)
            new_attention_mask[:, :, upper_tri_indices[0], upper_tri_indices[1]] = -65504. # torch.Size([1, 1, 14, 14])
            if past_key_value is not None:
                new_position_ids = torch.arange(
                    0, tokens_chosen,
                    dtype=torch.long, device=hidden_states.device
                ).unsqueeze(0)
            else:
                new_position_ids = torch.arange(0,int(tokens_chosen)).unsqueeze(0).to(selected_hidden_states.device)
        else:
            r_weights = route_probabilities
            kv_seq_len = past_key_value_length+seq_len
            new_attention_mask = torch.zeros(batch_size, 1, seq_len, kv_seq_len).to(hidden_states.device)
            if past_key_value is not None:
                new_position_ids = torch.arange(
                    past_key_value_length-1, past_key_value_length,
                    dtype=torch.long, device=hidden_states.device
                ).unsqueeze(0)
            else:
                new_position_ids = position_ids
            selected_hidden_states = hidden_states
        residual = selected_hidden_states
        selected_hidden_states = self.input_layernorm(selected_hidden_states)
        selected_hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=selected_hidden_states,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        selected_hidden_states = residual + selected_hidden_states
        # Fully Connected
        residual = selected_hidden_states
        selected_hidden_states = self.post_attention_layernorm(selected_hidden_states)
        selected_hidden_states = self.mlp(selected_hidden_states) * r_weights.unsqueeze(-1)
        selected_hidden_states = (residual + selected_hidden_states)
        outputs = (selected_hidden_states,)
        if seq_len <= 1:
            outputs = (selected_hidden_states,)
        else:
            indices_expanded = selected_tokens.unsqueeze(-1).expand(-1, -1, dim)
            hidden_states = hidden_states.scatter(dim=1, index=indices_expanded, src=selected_hidden_states) # if hidden_states now is [bs,topk,dim]
            outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
    return forward
def MoDLlamaModel_forward(self):
    def forward(
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_mod_loss: Optional[bool] = True,
        capacity_factor : Optional[float] = 0.5,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoDBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_mod_loss = [] if output_mod_loss else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    labels,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if output_mod_loss:
                if self.training:
                    if len(layer_outputs) >= 2:
                        all_mod_loss.extend([layer_outputs[-1]])
                    else:
                        all_mod_loss.append(torch.tensor(0, device=hidden_states.device, dtype=hidden_states.dtype))
                else:
                    pass
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return MoDBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            mod_loss_list=all_mod_loss,
        )
    return forward
class MoDMGMLlamaForCausalLM(LlamaForCausalLM, MGMMetaForCausalLM):
    config_class = MoDMGMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MoDMGMLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoDCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_aux
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )

        hidden_states = outputs[0]
        ######### calculate the percentage of routed tokens used for speed_evaluation ############
        mod_layers_idx = self.config.mod['mod_layers_idx']
        for idx in mod_layers_idx:
            self.total_tokens += self.model.layers[idx].total_tokens
            self.routed_tokens += self.model.layers[idx].routed_tokens

        # Calculate and print the overall percentage
        if self.total_tokens > 0:
            percentage = (self.routed_tokens / self.total_tokens) * 100
            print(f"Overall percentage of routed tokens: {percentage:.2f}%")
        
        
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        mod_loss, mod_losses = None, []
        if len(outputs[-1]) > 0:
            mod_loss_list = outputs[-1]
            for mod_loss in mod_loss_list:
                if mod_loss is not None:
                    mod_losses.append(mod_loss)
            mod_loss = sum(mod_losses) * self.router_aux_loss_coef
            if labels is not None:
                loss = loss + mod_loss
        if not return_dict:
            output = (logits,) + outputs[1:]
            output = (mod_loss,) + output if mod_loss is not None else output
            return (loss,) + output if loss is not None else output

        return MoDCausalLMOutputWithPast(
            loss=loss,
            mod_loss = mod_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mod_loss_list=outputs.mod_loss_list
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                images_aux
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        images_aux = kwargs.pop("images_aux", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if images_aux is not None:
            _inputs['images_aux'] = images_aux
        return _inputs
    def initialize_mod_modules(self, model_args):
        self.config.mod['mod_mode'] = model_args.mod_mode
        self.config.mod['mod_layers_idx'] = model_args.mod_layers_idx
        self.config.mod['capacity_factor'] = model_args.capacity_factor # topk = Int(squence_len * capacity_factor)
        self.config.mod['router_aux_loss_coef'] = self.router_aux_loss_coef = model_args.router_aux_loss_coef
        num_layers = self.config.num_hidden_layers
        mod_layers_idx = model_args.mod_layers_idx
        if model_args.mod_layers_idx is not None:
            model_args.mod_mode = 'custom'
            assert len(model_args.mod_layers_idx) <= num_layers
            assert max(model_args.mod_layers_idx) < num_layers
            assert min(model_args.mod_layers_idx) >= 0
        else:
            if model_args.mod_mode == "first_half":
                mod_layers_idx = list(range(0, num_layers // 2))
            elif model_args.mod_mode == "second_half":
                mod_layers_idx = list(range(num_layers // 2, num_layers-1))
            elif model_args.mod_mode == "sparse":
                mod_layers_idx = list(range(num_layers))[::2]
            elif model_args.mod_mode == "dense":
                mod_layers_idx = list(range(num_layers))
            elif model_args.mod_mode == "first_last_dense":
                mod_layers_idx = list(range(1, num_layers - 1))
            elif model_args.mod_mode == "last_two_thirds":
                mod_layers_idx = list(range(int(num_layers // 3)+2, num_layers-1))
            elif model_args.mod_mode == "first_five_dense":
                mod_layers_idx = list(range(5, num_layers - 1))
            elif model_args.mod_mode == "arank_mod":
                # Select 1st, 2nd, 3rd, and last layer
                mod_layers_idx = list(range(num_layers))
                # Remove the first three layers and the last one
                mod_layers_idx = [i for i in mod_layers_idx if i not in {0, 1, 2, num_layers-1}]
            else:
                raise NotImplementedError(
                    f'Only support ["first_half", "second_half", "sparse", "dense", "first_last_dense","last_two_thirds","first_five_dense","arank_mod"], but found {model_args.mod_mode}')
        self.config.mod['mod_layers_idx'] = mod_layers_idx
        print(mod_layers_idx)
        router = nn.Linear(self.config.hidden_size, 2, bias=False)
        torch.nn.init.xavier_uniform_(router.weight) # initialize the router weights
        for i in range(num_layers):
            self.model.layers[i].forward = LlamaDecoderLayer(self.model.layers[i]) 
        for idx in mod_layers_idx:
            self.model.layers[idx].router = router
            self.model.layers[idx].forward = MoDLlamaDecoderLayer_forward(self.model.layers[idx])
        self.model.forward = MoDLlamaModel_forward(self.model)
        self.model._init_weights(self.model) # register the router weights  
class EvalMoDMGMLlamaForCausalLM(MoDMGMLlamaForCausalLM):
    config_class = MoDMGMConfig

    def __init__(self, config):
        super(EvalMoDMGMLlamaForCausalLM, self).__init__(config)
        self.config.mod['mod_mode'] = True
        self.config.mod['capacity_factor'] = 0.5 # topk = Int(squence_len * capacity_factor)
        mod_layers_idx = self.config.mod['mod_layers_idx']
        router = nn.Linear(self.config.hidden_size, 2, bias=False)
        # Initialize counters for the entire model
        self.total_tokens = 0
        self.routed_tokens = 0
        for idx in mod_layers_idx:
            self.model.layers[idx].router = router
            self.model.layers[idx].forward = MoDLlamaDecoderLayer_forward_inference(self.model.layers[idx])
            # Initialize counters for each layer
            self.model.layers[idx].total_tokens = 0
            self.model.layers[idx].routed_tokens = 0
        self.model.forward = MoDLlamaModel_forward(self.model)

AutoConfig.register("mod_mgm", MoDMGMConfig)
AutoModelForCausalLM.register(MoDMGMConfig, MoDMGMLlamaForCausalLM)

AutoModelForCausalLM.register(MoDMGMConfig, EvalMoDMGMLlamaForCausalLM)