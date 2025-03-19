import re
from collections import namedtuple
from typing import List
import lark
import logging
from transformers import CLIPTokenizer,T5EncoderModel,T5Tokenizer
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline
from diffusers import StableDiffusion3Pipeline
import math
from diffusers import FluxPipeline
from typing import Tuple
import gc
import typing

logger = logging.getLogger(__name__)

def get_prompts_tokens_with_weights(
    clip_tokenizer: CLIPTokenizer
    , prompt: str = None
):
    """
    Get prompt token ids and weights, this function works for both prompt and negative prompt
    
    Args:
        pipe (CLIPTokenizer)
            A CLIPTokenizer
        prompt (str)
            A prompt string with weights
            
    Returns:
        text_tokens (list)
            A list contains token ids
        text_weight (list) 
            A list contains the correspodent weight of token ids
    
    Example:
        import torch
        from diffusers_plus.tools.sd_embeddings import get_prompts_tokens_with_weights
        from transformers import CLIPTokenizer

        clip_tokenizer = CLIPTokenizer.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , subfolder = "tokenizer"
            , dtype = torch.float16
        )

        token_id_list, token_weight_list = get_prompts_tokens_with_weights(
            clip_tokenizer = clip_tokenizer
            ,prompt = "a (red:1.5) cat"*70
        )
    """
    if (prompt is None) or (len(prompt)<1):
        prompt = "empty"
    
    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens,text_weights = [],[]
    for word, weight in texts_and_weights:
        # tokenize and discard the starting and the ending token
        token = clip_tokenizer(
            word
            , truncation = False        # so that tokenize whatever length prompt
        ).input_ids[1:-1]
        # the returned token is a 1d list: [320, 1125, 539, 320]
        
        # merge the new tokens to the all tokens holder: text_tokens
        text_tokens = [*text_tokens,*token]
        
        # each token chunk will come with one weight, like ['red cat', 2.0]
        # need to expand weight for each token.
        chunk_weights = [weight] * len(token) 
        
        # append the weight back to the weight holder: text_weights
        text_weights = [*text_weights, *chunk_weights]
    return text_tokens,text_weights

def get_prompts_tokens_with_weights_t5(
    t5_tokenizer: T5Tokenizer
    , prompt: str
):
    """
    Get prompt token ids and weights, this function works for both prompt and negative prompt
    """
    if (prompt is None) or (len(prompt)<1):
        prompt = "empty"
    
    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens,text_weights = [],[]
    for word, weight in texts_and_weights:
        # tokenize and discard the starting and the ending token
        token = t5_tokenizer(
            word
            , truncation            = False        # so that tokenize whatever length prompt
            , add_special_tokens    = True
        ).input_ids
        # the returned token is a 1d list: [320, 1125, 539, 320]
        
        # merge the new tokens to the all tokens holder: text_tokens
        text_tokens = [*text_tokens,*token]
        
        # each token chunk will come with one weight, like ['red cat', 2.0]
        # need to expand weight for each token.
        chunk_weights = [weight] * len(token) 
        
        # append the weight back to the weight holder: text_weights
        text_weights = [*text_weights, *chunk_weights]
    return text_tokens,text_weights

def group_tokens_and_weights(
    token_ids: list
    , weights: list
    , pad_last_block = False
):
    """
    Produce tokens and weights in groups and pad the missing tokens
    
    Args:
        token_ids (list)
            The token ids from tokenizer
        weights (list)
            The weights list from function get_prompts_tokens_with_weights
        pad_last_block (bool)
            Control if fill the last token list to 75 tokens with eos
    Returns:
        new_token_ids (2d list)
        new_weights (2d list)
    
    Example:
        from diffusers_plus.tools.sd_embeddings import group_tokens_and_weights
        token_groups,weight_groups = group_tokens_and_weights(
            token_ids = token_id_list
            , weights = token_weight_list
        )
    """
    bos,eos = 49406,49407
    
    # this will be a 2d list 
    new_token_ids = []
    new_weights   = []  
    while len(token_ids) >= 75:
        # get the first 75 tokens
        head_75_tokens = [token_ids.pop(0) for _ in range(75)]
        head_75_weights = [weights.pop(0) for _ in range(75)]
        
        # extract token ids and weights
        temp_77_token_ids = [bos] + head_75_tokens + [eos]
        temp_77_weights   = [1.0] + head_75_weights + [1.0]
        
        # add 77 token and weights chunk to the holder list
        new_token_ids.append(temp_77_token_ids)
        new_weights.append(temp_77_weights)
    
    # padding the left
    if len(token_ids) > 0:
        padding_len         = 75 - len(token_ids) if pad_last_block else 0
        
        temp_77_token_ids   = [bos] + token_ids + [eos] * padding_len + [eos]
        new_token_ids.append(temp_77_token_ids)
        
        temp_77_weights     = [1.0] + weights   + [1.0] * padding_len + [1.0]
        new_weights.append(temp_77_weights)
        
    return new_token_ids, new_weights

def get_weighted_text_embeddings_sd15(
    pipe: StableDiffusionPipeline
    , prompt : str      = ""
    , neg_prompt: str   = ""
    , pad_last_block    = False
    , clip_skip:int     = 0
):
    """
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion v1.5
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    
    Example:
        from diffusers import StableDiffusionPipeline
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , torch_dtype = torch.float16
            , safety_checker = None
        ).to("cuda:0")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_v15(
            pipe = text2img_pipe
            , prompt = "a (white) cat" 
            , neg_prompt = "blur"
        )
        image = text2img_pipe(
            prompt_embeds = prompt_embeds
            , negative_prompt_embeds = neg_prompt_embeds
            , generator = torch.Generator(text2img_pipe.device).manual_seed(2)
        ).images[0]
    """
    original_clip_layers = pipe.text_encoder.text_model.encoder.layers
    if clip_skip > 0:
        pipe.text_encoder.text_model.encoder.layers = original_clip_layers[:-clip_skip]
    
    eos = pipe.tokenizer.eos_token_id 
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )
    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )
    
    # padding the shorter one
    prompt_token_len        = len(prompt_tokens)
    neg_prompt_token_len    = len(neg_prompt_tokens)
    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens   = (
            neg_prompt_tokens  + 
            [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights  = (
            neg_prompt_weights + 
            [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens       = (
            prompt_tokens  
            + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        prompt_weights      = (
            prompt_weights 
            + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups ,prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
        
    # get prompt embeddings one by one is not working
    # we must embed prompt group by group
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            ,dtype = torch.long, device = pipe.device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )
        
        token_embedding = pipe.text_encoder(token_tensor)[0].squeeze(0) 
        for j in range(len(weight_tensor)):
            token_embedding[j] = token_embedding[j] * weight_tensor[j]
        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding)
        
        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , dtype = torch.long, device = pipe.device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )
        neg_token_embedding = pipe.text_encoder(neg_token_tensor)[0].squeeze(0) 
        for z in range(len(neg_weight_tensor)):
            neg_token_embedding[z] = (
                neg_token_embedding[z] * neg_weight_tensor[z]
            )
        neg_token_embedding = neg_token_embedding.unsqueeze(0)
        neg_embeds.append(neg_token_embedding)
    
    prompt_embeds       = torch.cat(embeds, dim = 1)
    neg_prompt_embeds   = torch.cat(neg_embeds, dim = 1)
    
    # recover clip layers
    if clip_skip > 0:
        pipe.text_encoder.text_model.encoder.layers = original_clip_layers
    
    return prompt_embeds, neg_prompt_embeds


def get_weighted_text_embeddings_sdxl(
    pipe: "StableDiffusionXLPipeline",
    prompt: str = "",
    neg_prompt: str = "",
    pad_last_block: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成 SDXL 的加权文本嵌入，确保正向和负向提示的形状一致。
    
    Args:
        pipe: StableDiffusionXLPipeline 对象
        prompt: 正向提示词 (str)
        neg_prompt: 负向提示词 (str)
        pad_last_block: 是否填充最后一个 token 块 (bool)
    
    Returns:
        Tuple[prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds]
    """
    device = pipe._execution_device
    max_length = pipe.tokenizer.model_max_length  # 通常为 77
    dtype = pipe.text_encoder_2.dtype if pipe.text_encoder_2 else pipe.unet.dtype

    with torch.no_grad():
        # 获取正向提示的分词和权重
        tokens, weights = get_prompts_tokens_with_weights(pipe.tokenizer, prompt)
        tokens_2, weights_2 = get_prompts_tokens_with_weights(pipe.tokenizer_2, prompt)
        token_groups, weight_groups = group_tokens_and_weights(tokens, weights, pad_last_block)
        token_groups_2, weight_groups_2 = group_tokens_and_weights(tokens_2, weights_2, pad_last_block)
        num_groups_prompt = len(token_groups)

        # 获取负向提示的分词和权重
        if not neg_prompt:
            num_groups_neg = 1  # 空负提示时，默认 1 块
        else:
            neg_tokens, neg_weights = get_prompts_tokens_with_weights(pipe.tokenizer, neg_prompt)
            neg_tokens_2, neg_weights_2 = get_prompts_tokens_with_weights(pipe.tokenizer_2, neg_prompt)
            neg_token_groups, neg_weight_groups = group_tokens_and_weights(neg_tokens, neg_weights, pad_last_block)
            neg_token_groups_2, neg_weight_groups_2 = group_tokens_and_weights(neg_tokens_2, neg_weights_2, pad_last_block)
            num_groups_neg = len(neg_token_groups)

        # 对齐分块数量，取最大值
        num_groups = max(num_groups_prompt, num_groups_neg)
        embed_dim_1 = pipe.text_encoder.config.hidden_size
        embed_dim_2 = pipe.text_encoder_2.config.hidden_size

        # 处理正向提示
        if num_groups_prompt == 1 and all(w == 1.0 for w in weight_groups[0]):  # 无权重单块
            input_ids = pipe.tokenizer(prompt, padding="max_length", max_length=max_length, 
                                      truncation=True, return_tensors="pt").input_ids.to(device)
            out_1 = pipe.text_encoder(input_ids, output_hidden_states=True)
            emb_1 = out_1.hidden_states[-2]

            input_ids_2 = pipe.tokenizer_2(prompt, padding="max_length", max_length=max_length, 
                                         truncation=True, return_tensors="pt").input_ids.to(device)
            out_2 = pipe.text_encoder_2(input_ids_2, output_hidden_states=True)
            emb_2 = out_2.hidden_states[-2]
            pooled_prompt_embeds = out_2[0]

            prompt_embeds = torch.cat([emb_1, emb_2], dim=-1)
            if num_groups > 1:  # 如果负提示分块更多，扩展正提示
                prompt_embeds = torch.cat([prompt_embeds, torch.zeros((1, (num_groups - 1) * max_length, embed_dim_1 + embed_dim_2), 
                                                                      dtype=dtype, device=device)], dim=1)
        else:  # 带权重或多块
            prompt_embeds = torch.zeros((1, num_groups * max_length, embed_dim_1 + embed_dim_2), 
                                       dtype=dtype, device=device)
            for j in range(num_groups_prompt):
                token_tensor = torch.tensor([token_groups[j]], dtype=torch.long, device=device)
                out_1 = pipe.text_encoder(token_tensor, output_hidden_states=True)
                emb_1 = out_1.hidden_states[-2].squeeze(0)

                token_tensor_2 = torch.tensor([token_groups_2[j]], dtype=torch.long, device=device)
                out_2 = pipe.text_encoder_2(token_tensor_2, output_hidden_states=True)
                emb_2 = out_2.hidden_states[-2].squeeze(0)
                if j == num_groups_prompt - 1:
                    pooled_prompt_embeds = out_2[0]

                emb = torch.cat([emb_1, emb_2], dim=-1)
                w = torch.tensor(weight_groups[j], dtype=dtype, device=device)
                mask = w != 1.0
                if mask.any():
                    emb[mask] = emb[mask] * w[mask].unsqueeze(-1)
                prompt_embeds[:, j * max_length:(j + 1) * max_length] = emb.unsqueeze(0)
            if num_groups > num_groups_prompt:  # 填充到与负提示一致
                prompt_embeds[:, num_groups_prompt * max_length:] = 0

        # 处理负向提示
        if not neg_prompt:  # 空负提示
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        else:
            if num_groups_neg == 1 and all(w == 1.0 for w in neg_weight_groups[0]):  # 无权重单块
                input_ids = pipe.tokenizer(neg_prompt, padding="max_length", max_length=max_length, 
                                          truncation=True, return_tensors="pt").input_ids.to(device)
                out_1 = pipe.text_encoder(input_ids, output_hidden_states=True)
                emb_1 = out_1.hidden_states[-2]

                input_ids_2 = pipe.tokenizer_2(neg_prompt, padding="max_length", max_length=max_length, 
                                             truncation=True, return_tensors="pt").input_ids.to(device)
                out_2 = pipe.text_encoder_2(input_ids_2, output_hidden_states=True)
                emb_2 = out_2.hidden_states[-2]
                negative_pooled_prompt_embeds = out_2[0]

                negative_prompt_embeds = torch.cat([emb_1, emb_2], dim=-1)
                if num_groups > 1:  # 如果正提示分块更多，扩展负提示
                    negative_prompt_embeds = torch.cat([negative_prompt_embeds, 
                                                        torch.zeros((1, (num_groups - 1) * max_length, embed_dim_1 + embed_dim_2), 
                                                                    dtype=dtype, device=device)], dim=1)
            else:  # 带权重或多块
                negative_prompt_embeds = torch.zeros((1, num_groups * max_length, embed_dim_1 + embed_dim_2), 
                                                    dtype=dtype, device=device)
                for j in range(num_groups_neg):
                    token_tensor = torch.tensor([neg_token_groups[j]], dtype=torch.long, device=device)
                    out_1 = pipe.text_encoder(token_tensor, output_hidden_states=True)
                    emb_1 = out_1.hidden_states[-2].squeeze(0)

                    token_tensor_2 = torch.tensor([neg_token_groups_2[j]], dtype=torch.long, device=device)
                    out_2 = pipe.text_encoder_2(token_tensor_2, output_hidden_states=True)
                    emb_2 = out_2.hidden_states[-2].squeeze(0)
                    if j == num_groups_neg - 1:
                        negative_pooled_prompt_embeds = out_2[0]

                    emb = torch.cat([emb_1, emb_2], dim=-1)
                    w = torch.tensor(neg_weight_groups[j], dtype=dtype, device=device)
                    mask = w != 1.0
                    if mask.any():
                        emb[mask] = emb[mask] * w[mask].unsqueeze(-1)
                    negative_prompt_embeds[:, j * max_length:(j + 1) * max_length] = emb.unsqueeze(0)
                if num_groups > num_groups_neg:  # 填充到与正提示一致
                    negative_prompt_embeds[:, num_groups_neg * max_length:] = 0

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def get_weighted_text_embeddings_sdxl_refiner(
    pipe: StableDiffusionXLPipeline
    , prompt : str      = ""
    , neg_prompt: str   = ""
):
    """
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion XL
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    
    Example:
        from diffusers import StableDiffusionPipeline
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , torch_dtype = torch.float16
            , safety_checker = None
        ).to("cuda:0")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_v15(
            pipe = text2img_pipe
            , prompt = "a (white) cat" 
            , neg_prompt = "blur"
        )
        image = text2img_pipe(
            prompt_embeds = prompt_embeds
            , negative_prompt_embeds = neg_prompt_embeds
            , generator = torch.Generator(text2img_pipe.device).manual_seed(2)
        ).images[0]
    """
    import math
    eos = 49407 #pipe.tokenizer.eos_token_id 
    
    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt
    )

    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, neg_prompt
    )
    
    # padding the shorter one for token set 2
    prompt_token_len_2        = len(prompt_tokens_2)
    neg_prompt_token_len_2    = len(neg_prompt_tokens_2)
    
    if prompt_token_len_2 > neg_prompt_token_len_2:
        # padding the neg_prompt with eos token
        neg_prompt_tokens_2   = (
            neg_prompt_tokens_2  + 
            [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        neg_prompt_weights_2  = (
            neg_prompt_weights_2 + 
            [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    else:
        # padding the prompt
        prompt_tokens_2       = (
            prompt_tokens_2  
            + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        prompt_weights_2      = (
            prompt_weights_2 
            + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy()
        , prompt_weights_2.copy()
    )
    
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy()
        , neg_prompt_weights_2.copy()
    )
        
    # get prompt embeddings one by one is not working. 
    for i in range(len(prompt_token_groups_2)):
        # get positive prompt embeddings with weights        
        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]]
            ,dtype = torch.long, device = pipe.device
        )
        
        weight_tensor_2 = torch.tensor(
            prompt_weight_groups_2[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2.to(pipe.device)
            , output_hidden_states = True
        )
        prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
        pooled_prompt_embeds = prompt_embeds_2[0]

        prompt_embeds_list = [prompt_embeds_2_hidden_states]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0)
        
        for j in range(len(weight_tensor_2)):
            if weight_tensor_2[j] != 1.0:
                ow = weight_tensor_2[j] - 1
                
                # optional process
                # To map number of (0,1) to (-1,1)
                tanh_weight = (math.exp(ow)/(math.exp(ow) + 1) - 0.5) * 2
                weight = 1 + tanh_weight
                
                # add weight method 1:
                # token_embedding[j] = token_embedding[j] * weight
                # token_embedding[j] = (
                #     token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight
                # )
                
                # add weight method 2:
                token_embedding[j] = (
                    token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight_tensor_2[j]
                )

        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding)
        
        # get negative prompt embeddings with weights
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_token_groups_2[i]]
            , dtype = torch.long, device = pipe.device
        )
        neg_weight_tensor_2 = torch.tensor(
            neg_prompt_weight_groups_2[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )
        
        # use second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(
            neg_token_tensor_2.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2]
        negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]

        neg_prompt_embeds_list = [neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0)
        
        for z in range(len(neg_weight_tensor_2)):
            if neg_weight_tensor_2[z] != 1.0:
                
                ow = neg_weight_tensor_2[z] - 1
                #neg_weight = 1 + (math.exp(ow)/(math.exp(ow) + 1) - 0.5) * 2
                
                # add weight method 1:
                # neg_token_embedding[z] = neg_token_embedding[z] * neg_weight
                # neg_token_embedding[z] = (
                #     neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight
                # )
                
                # add weight method 2:
                neg_token_embedding[z] = (
                    neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight_tensor_2[z]
                )
                
        neg_token_embedding = neg_token_embedding.unsqueeze(0)
        neg_embeds.append(neg_token_embedding)
    
    prompt_embeds           = torch.cat(embeds, dim = 1)
    negative_prompt_embeds  = torch.cat(neg_embeds, dim = 1)
    
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def get_weighted_text_embeddings_sdxl_2p(
    pipe: StableDiffusionXLPipeline
    , prompt : str      = ""
    , prompt_2 : str    = None
    , neg_prompt: str   = ""
    , neg_prompt_2: str = None
):
    """
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion XL, support two prompt sets.
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    
    Example:
        from diffusers import StableDiffusionPipeline
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , torch_dtype = torch.float16
            , safety_checker = None
        ).to("cuda:0")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_v15(
            pipe = text2img_pipe
            , prompt = "a (white) cat" 
            , neg_prompt = "blur"
        )
        image = text2img_pipe(
            prompt_embeds = prompt_embeds
            , negative_prompt_embeds = neg_prompt_embeds
            , generator = torch.Generator(text2img_pipe.device).manual_seed(2)
        ).images[0]
    """
    prompt_2        = prompt_2 or prompt
    neg_prompt_2    = neg_prompt_2 or neg_prompt
    
    import math
    eos = pipe.tokenizer.eos_token_id 
    
    # tokenizer 1
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )

    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )
    
    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt_2
    )

    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, neg_prompt_2
    )
    
    # padding the shorter one
    prompt_token_len        = len(prompt_tokens)
    neg_prompt_token_len    = len(neg_prompt_tokens)
    
    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens   = (
            neg_prompt_tokens  + 
            [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights  = (
            neg_prompt_weights + 
            [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens       = (
            prompt_tokens  
            + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        prompt_weights      = (
            prompt_weights 
            + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    
    # padding the shorter one for token set 2
    prompt_token_len_2        = len(prompt_tokens_2)
    neg_prompt_token_len_2    = len(neg_prompt_tokens_2)
    
    if prompt_token_len_2 > neg_prompt_token_len_2:
        # padding the neg_prompt with eos token
        neg_prompt_tokens_2   = (
            neg_prompt_tokens_2  + 
            [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        neg_prompt_weights_2  = (
            neg_prompt_weights_2 + 
            [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    else:
        # padding the prompt
        prompt_tokens_2       = (
            prompt_tokens_2  
            + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        prompt_weights_2      = (
            prompt_weights_2
            + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    
    # now, need to ensure prompt and prompt_2 has the same lemgth
    prompt_token_len        = len(prompt_tokens)
    prompt_token_len_2      = len(prompt_tokens_2)
    if prompt_token_len > prompt_token_len_2:
        prompt_tokens_2     = prompt_tokens_2   + [eos] * abs(prompt_token_len - prompt_token_len_2)
        prompt_weights_2    = prompt_weights_2  + [1.0] * abs(prompt_token_len - prompt_token_len_2)
    else:
        prompt_tokens       = prompt_tokens     + [eos] * abs(prompt_token_len - prompt_token_len_2)
        prompt_weights      = prompt_weights    + [1.0] * abs(prompt_token_len - prompt_token_len_2)
    
    # now, need to ensure neg_prompt and net_prompt_2 has the same lemgth
    neg_prompt_token_len        = len(neg_prompt_tokens)
    neg_prompt_token_len_2      = len(neg_prompt_tokens_2)
    if neg_prompt_token_len > neg_prompt_token_len_2:
        neg_prompt_tokens_2     = neg_prompt_tokens_2   + [eos] * abs(neg_prompt_token_len - neg_prompt_token_len_2)
        neg_prompt_weights_2    = neg_prompt_weights_2  + [1.0] * abs(neg_prompt_token_len - neg_prompt_token_len_2)
    else:
        neg_prompt_tokens       = neg_prompt_tokens     + [eos] * abs(neg_prompt_token_len - neg_prompt_token_len_2)
        neg_prompt_weights      = neg_prompt_weights    + [1.0] * abs(neg_prompt_token_len - neg_prompt_token_len_2)
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
    )
    
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
    )
    
    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy()
        , prompt_weights_2.copy()
    )
    
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy()
        , neg_prompt_weights_2.copy()
    )
        
    # get prompt embeddings one by one is not working. 
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            ,dtype = torch.long, device = pipe.device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , device    = pipe.device
        )
        
        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]]
            , device = pipe.device
        )
        
        weight_tensor_2 = torch.tensor(
            prompt_weight_groups_2[i]
            , device    = pipe.device
        )
        
        # use first text encoder
        prompt_embeds_1 = pipe.text_encoder(
            token_tensor.to(pipe.device)
            , output_hidden_states = True
        )
        prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2.to(pipe.device)
            , output_hidden_states = True
        )
        prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
        pooled_prompt_embeds = prompt_embeds_2[0]
        
        prompt_embeds_1_hidden_states = prompt_embeds_1_hidden_states.squeeze(0)
        prompt_embeds_2_hidden_states = prompt_embeds_2_hidden_states.squeeze(0)
        
        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                prompt_embeds_1_hidden_states[j] = (
                    prompt_embeds_1_hidden_states[-1] + (prompt_embeds_1_hidden_states[j] - prompt_embeds_1_hidden_states[-1]) * weight_tensor[j]
                )
            
            if weight_tensor_2[j] != 1.0:
                prompt_embeds_2_hidden_states[j] = (
                    prompt_embeds_2_hidden_states[-1] + (prompt_embeds_2_hidden_states[j] - prompt_embeds_2_hidden_states[-1]) * weight_tensor_2[j]
                )
                              
        prompt_embeds_1_hidden_states = prompt_embeds_1_hidden_states.unsqueeze(0)
        prompt_embeds_2_hidden_states = prompt_embeds_2_hidden_states.unsqueeze(0)
        
        prompt_embeds_list = [prompt_embeds_1_hidden_states, prompt_embeds_2_hidden_states]
        token_embedding = torch.cat(prompt_embeds_list, dim=-1)
        
        embeds.append(token_embedding)
        
        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , device = pipe.device
        )
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_token_groups_2[i]]
            , device = pipe.device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , device    = pipe.device
        )
        neg_weight_tensor_2 = torch.tensor(
            neg_prompt_weight_groups_2[i]
            , device    = pipe.device
        )
        
        # use first text encoder
        neg_prompt_embeds_1 = pipe.text_encoder(
            neg_token_tensor.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[-2]

        # use second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(
            neg_token_tensor_2.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2]
        negative_pooled_prompt_embeds = neg_prompt_embeds_2[0]
        
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1_hidden_states.squeeze(0)
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2_hidden_states.squeeze(0)
        
        for z in range(len(neg_weight_tensor)):
            if neg_weight_tensor[z] != 1.0:
                neg_prompt_embeds_1_hidden_states[z] = (
                    neg_prompt_embeds_1_hidden_states[-1] + (neg_prompt_embeds_1_hidden_states[z] - neg_prompt_embeds_1_hidden_states[-1]) * neg_weight_tensor[z]
                )
            
            if neg_weight_tensor_2[z] != 1.0:
                neg_prompt_embeds_2_hidden_states[z] = (
                    neg_prompt_embeds_2_hidden_states[-1] + (neg_prompt_embeds_2_hidden_states[z] - neg_prompt_embeds_2_hidden_states[-1]) * neg_weight_tensor_2[z]
                )
        
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1_hidden_states.unsqueeze(0)
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2_hidden_states.unsqueeze(0)

        neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states, neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.cat(neg_prompt_embeds_list, dim=-1)
        
        neg_embeds.append(neg_token_embedding)
    
    prompt_embeds           = torch.cat(embeds, dim = 1)
    negative_prompt_embeds  = torch.cat(neg_embeds, dim = 1)
    
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def get_weighted_text_embeddings_s_cascade(
        pipe: typing.Union[StableCascadePriorPipeline, StableCascadeDecoderPipeline]
        , prompt: str = ""
        , neg_prompt: str = ""
        , pad_last_block: bool = True
):
    """
     This function can process long prompt with weights, no length limitation
     for Stable Cascade

     Args:
         pipe (typing.Union[StableCascadePriorPipeline, StableCascadeDecoderPipeline])
         prompt (str)
         neg_prompt (str)
     Returns:
         prompt_embeds (torch.Tensor)
         neg_prompt_embeds (torch.Tensor)
         pooled_prompt_embeds (torch.Tensor)
         negative_pooled_prompt_embeds (torch.Tensor)
     """
    import math
    eos = pipe.tokenizer.eos_token_id

    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )

    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )

    # padding the shorter one
    prompt_token_len = len(prompt_tokens)
    neg_prompt_token_len = len(neg_prompt_tokens)

    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens = (
                neg_prompt_tokens +
                [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights = (
                neg_prompt_weights +
                [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens = (
                prompt_tokens
                + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        prompt_weights = (
                prompt_weights
                + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )

    embeds = []
    neg_embeds = []

    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block=pad_last_block
    )

    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
        , pad_last_block=pad_last_block
    )

    # get prompt embeddings one by one is not working.
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            , dtype=torch.long, device=pipe.device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , dtype=torch.float16
            , device=pipe.device
        )

        prompt_embeds_1 = pipe.text_encoder(
            token_tensor.to(pipe.device)
            , output_hidden_states=True
        )
        prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-1].cpu()

        pooled_prompt_embeds = prompt_embeds_1.text_embeds.unsqueeze(1)

        prompt_embeds_list = [prompt_embeds_1_hidden_states]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0).to(pipe.device)

        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                # ow = weight_tensor[j] - 1

                # optional process
                # To map number of (0,1) to (-1,1)
                # tanh_weight = (math.exp(ow)/(math.exp(ow) + 1) - 0.5) * 2
                # weight = 1 + tanh_weight

                # add weight method 1:
                # token_embedding[j] = token_embedding[j] * weight
                # token_embedding[j] = (
                #     token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight
                # )

                # add weight method 2:
                # token_embedding[j] = (
                #     token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight_tensor[j]
                # )

                # add weight method 3:
                token_embedding[j] = token_embedding[j] * weight_tensor[j]

        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding.cpu())

        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , dtype=torch.long, device=pipe.device
        )

        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , dtype=torch.float16
            , device=pipe.device
        )

        neg_prompt_embeds_1 = pipe.text_encoder(
            neg_token_tensor.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[-1].cpu()
        negative_pooled_prompt_embeds = neg_prompt_embeds_1.text_embeds.unsqueeze(1)

        neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0).to(pipe.device)

        for z in range(len(neg_weight_tensor)):
            if neg_weight_tensor[z] != 1.0:
                # ow = neg_weight_tensor[z] - 1
                # neg_weight = 1 + (math.exp(ow)/(math.exp(ow) + 1) - 0.5) * 2

                # add weight method 1:
                # neg_token_embedding[z] = neg_token_embedding[z] * neg_weight
                # neg_token_embedding[z] = (
                #     neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight
                # )

                # add weight method 2:
                # neg_token_embedding[z] = (
                #     neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight_tensor[z]
                # )

                # add weight method 3:
                neg_token_embedding[z] = neg_token_embedding[z] * neg_weight_tensor[z]

        neg_token_embedding = neg_token_embedding.unsqueeze(0)
        neg_embeds.append(neg_token_embedding.cpu())

    prompt_embeds = torch.cat(embeds, dim=1).to(pipe.device)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1).to(pipe.device)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def get_weighted_text_embeddings_sd3(
    pipe: StableDiffusion3Pipeline
    , prompt : str      = ""
    , neg_prompt: str   = ""
    , pad_last_block    = True
    , use_t5_encoder    = True
):
    """
    This function can process long prompt with weights, no length limitation 
    for Stable Diffusion 3
    
    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        neg_prompt (str)
    Returns:
        sd3_prompt_embeds (torch.Tensor)
        sd3_neg_prompt_embeds (torch.Tensor)
        pooled_prompt_embeds (torch.Tensor)
        negative_pooled_prompt_embeds (torch.Tensor)
    """
    import math
    eos = pipe.tokenizer.eos_token_id 
    
    # tokenizer 1
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )

    neg_prompt_tokens, neg_prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, neg_prompt
    )
    
    # tokenizer 2
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, prompt
    )

    neg_prompt_tokens_2, neg_prompt_weights_2 = get_prompts_tokens_with_weights(
        pipe.tokenizer_2, neg_prompt
    )
    
    # tokenizer 3
    prompt_tokens_3, prompt_weights_3 = get_prompts_tokens_with_weights_t5(
        pipe.tokenizer_3, prompt
    )

    neg_prompt_tokens_3, neg_prompt_weights_3 = get_prompts_tokens_with_weights_t5(
        pipe.tokenizer_3, neg_prompt
    )
    
    # padding the shorter one
    prompt_token_len        = len(prompt_tokens)
    neg_prompt_token_len    = len(neg_prompt_tokens)
    
    if prompt_token_len > neg_prompt_token_len:
        # padding the neg_prompt with eos token
        neg_prompt_tokens   = (
            neg_prompt_tokens  + 
            [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        neg_prompt_weights  = (
            neg_prompt_weights + 
            [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    else:
        # padding the prompt
        prompt_tokens       = (
            prompt_tokens  
            + [eos] * abs(prompt_token_len - neg_prompt_token_len)
        )
        prompt_weights      = (
            prompt_weights 
            + [1.0] * abs(prompt_token_len - neg_prompt_token_len)
        )
    
    # padding the shorter one for token set 2
    prompt_token_len_2        = len(prompt_tokens_2)
    neg_prompt_token_len_2    = len(neg_prompt_tokens_2)
    
    if prompt_token_len_2 > neg_prompt_token_len_2:
        # padding the neg_prompt with eos token
        neg_prompt_tokens_2   = (
            neg_prompt_tokens_2  + 
            [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        neg_prompt_weights_2  = (
            neg_prompt_weights_2 + 
            [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    else:
        # padding the prompt
        prompt_tokens_2       = (
            prompt_tokens_2  
            + [eos] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
        prompt_weights_2      = (
            prompt_weights_2 
            + [1.0] * abs(prompt_token_len_2 - neg_prompt_token_len_2)
        )
    
    embeds = []
    neg_embeds = []
    
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups, neg_prompt_weight_groups = group_tokens_and_weights(
        neg_prompt_tokens.copy()
        , neg_prompt_weights.copy()
        , pad_last_block = pad_last_block
    )
    
    prompt_token_groups_2, prompt_weight_groups_2 = group_tokens_and_weights(
        prompt_tokens_2.copy()
        , prompt_weights_2.copy()
        , pad_last_block = pad_last_block
    )
    
    neg_prompt_token_groups_2, neg_prompt_weight_groups_2 = group_tokens_and_weights(
        neg_prompt_tokens_2.copy()
        , neg_prompt_weights_2.copy()
        , pad_last_block = pad_last_block
    )
        
    # get prompt embeddings one by one is not working. 
    for i in range(len(prompt_token_groups)):
        # get positive prompt embeddings with weights
        token_tensor = torch.tensor(
            [prompt_token_groups[i]]
            ,dtype = torch.long, device = pipe.device
        )
        weight_tensor = torch.tensor(
            prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )
        
        token_tensor_2 = torch.tensor(
            [prompt_token_groups_2[i]]
            ,dtype = torch.long, device = pipe.device
        )
        
        # use first text encoder
        prompt_embeds_1 = pipe.text_encoder(
            token_tensor.to(pipe.device)
            , output_hidden_states = True
        )
        prompt_embeds_1_hidden_states = prompt_embeds_1.hidden_states[-2]
        pooled_prompt_embeds_1 = prompt_embeds_1[0]

        # use second text encoder
        prompt_embeds_2 = pipe.text_encoder_2(
            token_tensor_2.to(pipe.device)
            , output_hidden_states = True
        )
        prompt_embeds_2_hidden_states = prompt_embeds_2.hidden_states[-2]
        pooled_prompt_embeds_2 = prompt_embeds_2[0]

        prompt_embeds_list = [prompt_embeds_1_hidden_states, prompt_embeds_2_hidden_states]
        token_embedding = torch.concat(prompt_embeds_list, dim=-1).squeeze(0).to(pipe.device)
        
        for j in range(len(weight_tensor)):
            if weight_tensor[j] != 1.0:
                #ow = weight_tensor[j] - 1
                
                # optional process
                # To map number of (0,1) to (-1,1)
                # tanh_weight = (math.exp(ow)/(math.exp(ow) + 1) - 0.5) * 2
                # weight = 1 + tanh_weight
                
                # add weight method 1:
                # token_embedding[j] = token_embedding[j] * weight
                # token_embedding[j] = (
                #     token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight
                # )
                
                # add weight method 2:
                # token_embedding[j] = (
                #     token_embedding[-1] + (token_embedding[j] - token_embedding[-1]) * weight_tensor[j]
                # )
                
                # add weight method 3:
                token_embedding[j] = token_embedding[j] * weight_tensor[j]

        token_embedding = token_embedding.unsqueeze(0)
        embeds.append(token_embedding)
        
        # get negative prompt embeddings with weights
        neg_token_tensor = torch.tensor(
            [neg_prompt_token_groups[i]]
            , dtype = torch.long, device = pipe.device
        )
        neg_token_tensor_2 = torch.tensor(
            [neg_prompt_token_groups_2[i]]
            , dtype = torch.long, device = pipe.device
        )
        neg_weight_tensor = torch.tensor(
            neg_prompt_weight_groups[i]
            , dtype     = torch.float16
            , device    = pipe.device
        )
        
        # use first text encoder
        neg_prompt_embeds_1 = pipe.text_encoder(
            neg_token_tensor.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_1_hidden_states = neg_prompt_embeds_1.hidden_states[-2]
        negative_pooled_prompt_embeds_1 = neg_prompt_embeds_1[0]

        # use second text encoder
        neg_prompt_embeds_2 = pipe.text_encoder_2(
            neg_token_tensor_2.to(pipe.device)
            , output_hidden_states=True
        )
        neg_prompt_embeds_2_hidden_states = neg_prompt_embeds_2.hidden_states[-2]
        negative_pooled_prompt_embeds_2 = neg_prompt_embeds_2[0]

        neg_prompt_embeds_list = [neg_prompt_embeds_1_hidden_states, neg_prompt_embeds_2_hidden_states]
        neg_token_embedding = torch.concat(neg_prompt_embeds_list, dim=-1).squeeze(0).to(pipe.device)
        
        for z in range(len(neg_weight_tensor)):
            if neg_weight_tensor[z] != 1.0:
                
                # ow = neg_weight_tensor[z] - 1
                # neg_weight = 1 + (math.exp(ow)/(math.exp(ow) + 1) - 0.5) * 2
                
                # add weight method 1:
                # neg_token_embedding[z] = neg_token_embedding[z] * neg_weight
                # neg_token_embedding[z] = (
                #     neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight
                # )
                
                # add weight method 2:
                # neg_token_embedding[z] = (
                #     neg_token_embedding[-1] + (neg_token_embedding[z] - neg_token_embedding[-1]) * neg_weight_tensor[z]
                # )
                
                # add weight method 3:
                neg_token_embedding[z] = neg_token_embedding[z] * neg_weight_tensor[z]
                
        neg_token_embedding = neg_token_embedding.unsqueeze(0)
        neg_embeds.append(neg_token_embedding)
    
    prompt_embeds           = torch.cat(embeds, dim = 1)
    negative_prompt_embeds  = torch.cat(neg_embeds, dim = 1)
    
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)
    negative_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds_1, negative_pooled_prompt_embeds_2], dim=-1)
    
    if use_t5_encoder and pipe.text_encoder_3:        
        # ----------------- generate positive t5 embeddings --------------------
        prompt_tokens_3 = torch.tensor([prompt_tokens_3],dtype=torch.long)
        
        t5_prompt_embeds    = pipe.text_encoder_3(prompt_tokens_3.to(pipe.device))[0].squeeze(0)
        t5_prompt_embeds    = t5_prompt_embeds.to(device=pipe.device)
        
        # add weight to t5 prompt
        for z in range(len(prompt_weights_3)):
            if prompt_weights_3[z] != 1.0:
                t5_prompt_embeds[z] = t5_prompt_embeds[z] * prompt_weights_3[z]
        t5_prompt_embeds = t5_prompt_embeds.unsqueeze(0)
    else:
        t5_prompt_embeds    = torch.zeros(1, 4096, dtype = prompt_embeds.dtype).unsqueeze(0)
        t5_prompt_embeds    = t5_prompt_embeds.to(device=pipe.device)
        
    # merge with the clip embedding 1 and clip embedding 2
    clip_prompt_embeds = torch.nn.functional.pad(
        prompt_embeds, (0, t5_prompt_embeds.shape[-1] - prompt_embeds.shape[-1])
    )
    sd3_prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
    
    if use_t5_encoder and pipe.text_encoder_3:  
        # ---------------------- get neg t5 embeddings -------------------------
        neg_prompt_tokens_3 = torch.tensor([neg_prompt_tokens_3],dtype=torch.long)
        
        t5_neg_prompt_embeds    = pipe.text_encoder_3(neg_prompt_tokens_3.to(pipe.device))[0].squeeze(0)
        t5_neg_prompt_embeds    = t5_neg_prompt_embeds.to(device=pipe.device)
        
        # add weight to neg t5 embeddings
        for z in range(len(neg_prompt_weights_3)):
            if neg_prompt_weights_3[z] != 1.0:
                t5_neg_prompt_embeds[z] = t5_neg_prompt_embeds[z] * neg_prompt_weights_3[z]
        t5_neg_prompt_embeds = t5_neg_prompt_embeds.unsqueeze(0)
    else: 
        t5_neg_prompt_embeds    = torch.zeros(1, 4096, dtype = prompt_embeds.dtype).unsqueeze(0)
        t5_neg_prompt_embeds    = t5_prompt_embeds.to(device=pipe.device)

    clip_neg_prompt_embeds = torch.nn.functional.pad(
        negative_prompt_embeds, (0, t5_neg_prompt_embeds.shape[-1] - negative_prompt_embeds.shape[-1])
    )
    sd3_neg_prompt_embeds = torch.cat([clip_neg_prompt_embeds, t5_neg_prompt_embeds], dim=-2)
    
    # padding 
    import torch.nn.functional as F
    size_diff = sd3_neg_prompt_embeds.size(1) - sd3_prompt_embeds.size(1)
    # Calculate padding. Format for pad is (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    # Since we are padding along the second dimension (axis=1), we need (0, 0, padding_top, padding_bottom, 0, 0)
    # Here padding_top will be 0 and padding_bottom will be size_diff

    # Check if padding is needed
    if size_diff > 0:
        padding = (0, 0, 0, abs(size_diff), 0, 0)
        sd3_prompt_embeds = F.pad(sd3_prompt_embeds, padding)
    elif size_diff < 0:
        padding = (0, 0, 0, abs(size_diff), 0, 0)
        sd3_neg_prompt_embeds = F.pad(sd3_neg_prompt_embeds, padding)
    
    return sd3_prompt_embeds, sd3_neg_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def get_weighted_text_embeddings_flux1(
    pipe: FluxPipeline
    , prompt: str       = ""
    , prompt2: str      = None
    , device            = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function can process long prompt with weights for flux1 model
    
    Args:
        pipe (['FluxPipeline']): The FluxPipeline
        prompt (['string']): the 1st prompt
        prompt2 (['string']): the 2nd prompt
        device (['string']): target device
    Returns:
        A tuple include two embedding tensors
    """
    prompt2 = prompt if prompt2 is None else prompt2
    
    # so that user can assign custom cuda
    if device is None or device == 'cpu':
        target_device = 'cuda:0'
    else:
        target_device = device
        
    # prepare text_encoder and text_encoder_2
    if not pipe.device.type.startswith('cuda'):
        pipe.text_encoder.to(target_device)
        pipe.text_encoder_2.to(target_device)
    
    # tokenizer 1 - openai/clip-vit-large-patch14
    prompt_tokens, prompt_weights = get_prompts_tokens_with_weights(
        pipe.tokenizer, prompt
    )
    
    # tokenizer 2 - google/t5-v1_1-xxl
    prompt_tokens_2, prompt_weights_2 = get_prompts_tokens_with_weights_t5(
        pipe.tokenizer_2, prompt2
    )
    
    prompt_token_groups, prompt_weight_groups = group_tokens_and_weights(
        prompt_tokens.copy()
        , prompt_weights.copy()
        , pad_last_block = True
    )
        
    # # get positive prompt embeddings, flux1 use only text_encoder 1 pooled embeddings
    # token_tensor = torch.tensor(
    #     [prompt_token_groups[0]]
    #     , dtype = torch.long, device = device
    # )
    # # use first text encoder
    # prompt_embeds_1 = pipe.text_encoder(
    #     token_tensor.to(device)
    #     , output_hidden_states  = False
    # )
    # pooled_prompt_embeds_1  = prompt_embeds_1.pooler_output
    # prompt_embeds           = pooled_prompt_embeds_1.to(dtype = pipe.text_encoder.dtype, device = device)
    
    # use avg pooling embeddings
    pool_embeds_list = []
    for token_group in prompt_token_groups:
        token_tensor = torch.tensor(
            [token_group]
            , dtype = torch.long
            , device = target_device
        )
        with torch.no_grad():
            prompt_embeds_1 = pipe.text_encoder(
                token_tensor.to(target_device)
                , output_hidden_states  = False
            )
        pooled_prompt_embeds = prompt_embeds_1.pooler_output.squeeze(0)
        pool_embeds_list.append(pooled_prompt_embeds)
        
    prompt_embeds = torch.stack(pool_embeds_list,dim=0)
    
    # get the avg pool
    prompt_embeds = prompt_embeds.mean(dim=0, keepdim=True)
    # prompt_embeds = prompt_embeds.unsqueeze(0)
    prompt_embeds = prompt_embeds.to(dtype = pipe.text_encoder.dtype, device = target_device)
            
    # generate positive t5 embeddings 
    prompt_tokens_2 = torch.tensor([prompt_tokens_2],dtype=torch.long)
    
    with torch.no_grad():
        t5_prompt_embeds    = pipe.text_encoder_2(prompt_tokens_2.to(target_device))[0].squeeze(0)
    t5_prompt_embeds    = t5_prompt_embeds.to(device = target_device)
    
    # add weight to t5 prompt
    for z in range(len(prompt_weights_2)):
        if prompt_weights_2[z] != 1.0:
            t5_prompt_embeds[z] = t5_prompt_embeds[z] * prompt_weights_2[z]
    t5_prompt_embeds = t5_prompt_embeds.unsqueeze(0)
    t5_prompt_embeds = t5_prompt_embeds.to(dtype = pipe.text_encoder_2.dtype, device = target_device)
    
    # release text encoder from vram
    if pipe.device.type.startswith('cpu'):
        pipe.text_encoder.to('cpu')
        pipe.text_encoder_2.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
    
    return t5_prompt_embeds,prompt_embeds

schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER "]"
alternate: "[" prompt ("|" prompt)+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")

def get_learned_conditioning_prompt_schedules(prompts, steps):
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    >>> g("[a|(b:1.1)]")
    [[1, 'a'], [2, '(b:1.1)'], [3, 'a'], [4, '(b:1.1)'], [5, 'a'], [6, '(b:1.1)'], [7, 'a'], [8, '(b:1.1)'], [9, 'a'], [10, '(b:1.1)']]
    """

    def collect_steps(steps, tree):
        l = [steps]
        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                tree.children[-1] = float(tree.children[-1])
                if tree.children[-1] < 1:
                    tree.children[-1] *= steps
                tree.children[-1] = min(steps, int(tree.children[-1]))
                l.append(tree.children[-1])
            def alternate(self, tree):
                l.extend(range(1, steps+1))
        CollectSteps().visit(tree)
        return sorted(set(l))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, _, when = args
                yield before or () if step <= when else after
            def alternate(self, args):
                yield next(args[(step - 1)%len(args)])
            def start(self, args):
                def flatten(x):
                    if type(x) == str:
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)
                return ''.join(flatten(args))
            def plain(self, args):
                yield args[0].value
            def __default__(self, data, children, meta):
                for child in children:
                    yield child
        return AtStep().transform(tree)

    def get_schedule(prompt):
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError as e:
            if 0:
                import traceback
                traceback.print_exc()
            return [[steps, prompt]]
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]

def parse_scheduled_prompts(text,steps=30):
    '''
    This function will handle scheduled and alternative prompt
    '''
    text = text.strip()
    parse_result = None
    try:
        parse_result = get_learned_conditioning_prompt_schedules([text],steps=steps)[0]
        logger.info(
            f"parse_result from get_learned_conditioning_prompt_schedules function:\n {str(parse_result)}"
        )
    except Exception as e:
        logger.error(f"Parse scheduled prompt error:\n {e}")

    if len(parse_result) == 1:
        # no scheduling
        return parse_result
    
    prompts_list = []
    
    for i in range(steps):
        current_prompt_step, current_prompt_content = parse_result[0][0],parse_result[0][1]
        step = i + 1
        if step < current_prompt_step:
            prompts_list.append(current_prompt_content)
            continue
        
        if step == current_prompt_step:
            prompts_list.append(current_prompt_content)
            parse_result.pop(0)
        
    return prompts_list




ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])


def get_learned_conditioning(model, prompts, steps):
    """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),
    and the sampling step at which this condition is to be replaced by the next one.

    Input:
    (model, ['a red crown', 'a [blue:green:5] jeweled crown'], 20)

    Output:
    [
        [
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0523,  ..., -0.4901, -0.3066,  0.0674], ..., [ 0.3317, -0.5102, -0.4066,  ...,  0.4119, -0.7647, -1.0160]], device='cuda:0'))
        ],
        [
            ScheduledPromptConditioning(end_at_step=5, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.0192,  0.3867, -0.4644,  ...,  0.1135, -0.3696, -0.4625]], device='cuda:0')),
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.7352, -0.4356, -0.7888,  ...,  0.6994, -0.4312, -1.2593]], device='cuda:0'))
        ]
    ]
    """
    res = []

    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps)
    cache = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):

        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        texts = [x[1] for x in prompt_schedule]
        conds = model.get_learned_conditioning(texts)

        cond_schedule = []
        for i, (end_at_step, text) in enumerate(prompt_schedule):
            cond_schedule.append(ScheduledPromptConditioning(end_at_step, conds[i]))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)

    return res


re_AND = re.compile(r"\bAND\b")
re_weight = re.compile(r"^(.*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")

def get_multicond_prompt_list(prompts):
    res_indexes = []

    prompt_flat_list = []
    prompt_indexes = {}

    for prompt in prompts:
        subprompts = re_AND.split(prompt)

        indexes = []
        for subprompt in subprompts:
            match = re_weight.search(subprompt)

            text, weight = match.groups() if match is not None else (subprompt, 1.0)

            weight = float(weight) if weight is not None else 1.0

            index = prompt_indexes.get(text, None)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index

            indexes.append((index, weight))

        res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes


class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0):
        self.schedules: List[ScheduledPromptConditioning] = schedules
        self.weight: float = weight


class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape  # the shape field is needed to send this object to DDIM/PLMS
        self.batch: List[List[ComposableScheduledPromptConditioning]] = batch

def get_multicond_learned_conditioning(model, prompts, steps) -> MulticondLearnedConditioning:
    """same as get_learned_conditioning, but returns a list of ScheduledPromptConditioning along with the weight objects for each prompt.
    For each prompt, the list is obtained by splitting the prompt using the AND separator.

    https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
    """

    res_indexes, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)

    learned_conditioning = get_learned_conditioning(model, prompt_flat_list, steps)

    res = []
    for indexes in res_indexes:
        res.append([ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])

    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res)


def reconstruct_cond_batch(c: List[List[ScheduledPromptConditioning]], current_step):
    param = c[0][0].cond
    res = torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype)
    for i, cond_schedule in enumerate(c):
        target_index = 0
        for current, (end_at, cond) in enumerate(cond_schedule):
            if current_step <= end_at:
                target_index = current
                break
        res[i] = cond_schedule[target_index].cond

    return res


def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step):
    param = c.batch[0][0].schedules[0].cond

    tensors = []
    conds_list = []

    for batch_no, composable_prompts in enumerate(c.batch):
        conds_for_batch = []

        for cond_index, composable_prompt in enumerate(composable_prompts):
            target_index = 0
            for current, (end_at, cond) in enumerate(composable_prompt.schedules):
                if current_step <= end_at:
                    target_index = current
                    break

            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(composable_prompt.schedules[target_index].cond)

        conds_list.append(conds_for_batch)

    # if prompts have wildly different lengths above the limit we'll get tensors fo different shapes
    # and won't be able to torch.stack them. So this fixes that.
    token_count = max([x.shape[0] for x in tensors])
    for i in range(len(tensors)):
        if tensors[i].shape[0] != token_count:
            last_vector = tensors[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - tensors[i].shape[0], 1])
            tensors[i] = torch.vstack([tensors[i], last_vector_repeated])

    return conds_list, torch.stack(tensors).to(device=param.device, dtype=param.dtype)


re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res