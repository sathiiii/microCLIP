import json
from tqdm import tqdm
import hashlib
import os
import urllib
import warnings
from pathlib import Path
import copy
from typing import Any, Union, List
from tqdm import tqdm
from sklearn.manifold import TSNE
import numpy as np
from torchvision import transforms as T
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
import torch.nn.functional as F
from clip.model import build_model
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .coder import *
from typing import Type, Any, Dict, Optional, List, Tuple
from torch import Tensor
from functools import reduce
from operator import mul
import math

from src.open_clip.tokenizer import tokenize as metaclip_tokenize
from src.open_clip.factory import create_model_and_transforms
# from transformers import AutoModel, AutoProcessor


NoneType = Type[None]


_tokenizer = _Tokenizer()
_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}

CUSTOM_TEMPLATES = {
    "pets": "a photo of a {}, a type of pet.",
    "flowers": "a photo of a {}, a type of flower.",
    "fgvc": "a photo of a {}, a type of aircraft.",
    "dtd": "a photo of a {}, a type of a texture.",
    "eurosat": "a photo of a centered satellite photo of {}.",
    "cars": "a photo of a {}.",
    "food101": "a photo of a {}, a type of food.",
    "sun397": "a photo of a {}.",
    "cifar10": "a photo of a {}.",
    "cifar100": "a photo of a {}.",
    "caltech101": "a photo of a {}.",
    "ucf101": "a photo of a {}.",
    "imagenet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "gtsrb": "a photo of a {} traffic sign.",
}

DATASET_MAP = {
    'cars': 'stanford_cars',
    'fgvc': 'Aircraft',
    'dtd': 'DTD',
    'flowers': 'oxford_flowers',
    'pets': 'oxford-pet',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'cub': 'CUB200'
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")
    return download_target

def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).
    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")
    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model
    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]
    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)
    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)
    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()
        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)
            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)
        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()
    return model


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

class Linear(nn.Module):
    def __init__(self, in_dim: int, identity_init: bool = True, bias=False) -> NoneType:
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim, bias=bias)
        if identity_init:
            nn.init.zeros_(self.linear.weight)
            self.linear.weight.data += torch.eye(in_dim)
        else:
            nn.init.normal_(self.linear.weight, std=in_dim**-0.5)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class AdapterMLP(nn.Module):
    '''
    MLP Network for low-shot adaptation (trained on top of frozen features)
    '''
    def __init__(self, input_size):
        super(AdapterMLP, self).__init__()

        self.mlp = nn.Sequential(
            Linear(input_size),
            nn.ReLU(inplace=True),
            Linear(input_size)
        )
    def forward(self, x):
        out = self.mlp(x)
        return out
    
# class CustomSigLIP(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.model = AutoModel.from_pretrained(args.train_config['pretrained'])
#         self.model.float()
#         self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
#         self.tokenizer = lambda texts: self.processor.text_processor(
#             text=texts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True
#         )
#         with torch.no_grad():
#             dummy_image = torch.randn(args.batch_size, 3, 224, 224)  # Example image tensor
#             dummy_image = (dummy_image - dummy_image.min()) / (dummy_image.max() - dummy_image.min())  # Normalize to [0, 1]
#             self.embed_dim = self(dummy_image).shape[-1]  # Get the embedding dimension from the last dimension of the output

#     def forward(self, images, return_cls_token=False, return_patch_tokens=False):
#         device = next(self.parameters()).device
#         image_tensor = self.processor.image_processor.preprocess(
#             images=images,
#             return_tensors="pt",
#             do_rescale=False
#         ).pixel_values.to(device)

#         vision_model = self.model.vision_model
#         encoder_layers = vision_model.encoder.layers
#         num_layers = len(encoder_layers)

#         # === Step 1: Embedding ===
#         hidden_states = vision_model.embeddings(image_tensor, interpolate_pos_encoding=True)

#         if not return_patch_tokens:
#             # Forward pass without patch token extraction
#             for layer_module in encoder_layers:
#                 hidden_states = layer_module(hidden_states)[0]
#             last_hidden_state = vision_model.post_layernorm(hidden_states)
#             cls_token = last_hidden_state[:, 0, :]  # CLS token (first token)
            
#             # Normalize CLS token and apply projection if defined
#             cls_token = self.ln_post(cls_token)
#             if self.proj is not None:
#                 cls_proj = cls_token @ self.proj
#             cls_proj = F.normalize(cls_proj, dim=-1)

#             returns = (cls_proj,)  # Return cls projection

#             # Optionally return cls_token
#             if return_cls_token:
#                 returns += (cls_token,)

#             return returns[0] if len(returns) == 1 else returns

#         else:
#             # With patch token extraction
#             patch_tokens = None
#             for i, layer_module in enumerate(encoder_layers):
#                 if i == num_layers - 1:
#                     break
#                 hidden_states = layer_module(hidden_states)[0]
#                 if i == num_layers - 2:
#                     patch_tokens = hidden_states[:, 1:, :].clone()  # Extract patch tokens

#             hidden_states = encoder_layers[-1](hidden_states)[0]
#             last_hidden_state = vision_model.post_layernorm(hidden_states)
#             cls_token = last_hidden_state[:, 0, :]  # CLS token

#             # Normalize CLS token and apply projection if defined
#             cls_token = self.ln_post(cls_token)
#             if self.proj is not None:
#                 cls_proj = cls_token @ self.proj
#             cls_proj = F.normalize(cls_proj, dim=-1)

#             returns = (cls_proj,)  # Return cls projection

#             # Optionally return patch tokens
#             if return_patch_tokens:
#                 patch_tokens = patch_tokens.permute(1, 0, 2)  # Adjust shape if needed
#                 patch_tokens = self.ln_post(patch_tokens)  # Normalize patch tokens
#                 patch_tokens = patch_tokens[:, 1:]  # Ignore first token (CLS token)
#                 patch_tokens = patch_tokens.reshape(patch_tokens.size(0), -1, patch_tokens.size(-1)).contiguous()
#                 returns += (patch_tokens,)

#             # Optionally return cls_token
#             if return_cls_token:
#                 returns += (cls_token,)

#             return returns[0] if len(returns) == 1 else returns

#     def encode_text(self, texts):
#         text_outputs = self.model.text_model(**texts)
#         text_embeds = self.model.text_projection(text_outputs.last_hidden_state[:, 0, :])
#         return text_embeds

class CLIPClassifier(nn.Module):
    def __init__(self, args):        
        super().__init__()
        
        self.dataset_name = args.dataset
        self.args = args
        classname_path = os.path.join("./configs", args.classname)
        with open(classname_path, 'r') as classname_file:
            classnames = json.load(classname_file)
        self.classnames = classnames[self.dataset_name]
        
        # self.clip_model = load(args.clip_model, jit=False)
        # self.clip_model.float()
        self.tokenize = tokenize
        if args.train_config['source_model'] == 'CLIP':
            self.clip_model = load(args.clip_model, jit=False)
            self.clip_model.float()
            self.tokenize = tokenize
        elif args.train_config['source_model'] == 'MetaCLIP':
            self.clip_model = create_model_and_transforms(args.train_config['vision_backbone'], pretrained=args.train_config['pretrained'])[0] # for 2.5B use 'metaclip_fullcc' in OpenCLIP or 'metaclip_2_5b'
            self.clip_model.float()
            self.tokenize = metaclip_tokenize
        # elif args.train_config['source_model'] == 'SigLIP':
        #     self.clip_model = CustomSigLIP(args)
        #     self.clip_model.float()
        #     self.tokenize = self.clip_model.tokenizer
        
        self.clip_model = self.clip_model.to(args.device)
        self.embed_dim = self.clip_model.embed_dim
        print(f"Embedding dimension: {self.embed_dim}")
        ## ------------------------------------------------------------------
        PATH_TO_PROMPTS = f"{args.text_descriptions_path}/{self.dataset_name}.json"
        with open(PATH_TO_PROMPTS) as f:
            gpt3_prompts = json.load(f)
        templates_path = os.path.join("./configs", args.template)
        with open(templates_path, 'r') as templates_file:
            self.templates = json.load(templates_file)
        self.templates = self.templates[args.dataset]
        
        print(f'Number of classes: {len(self.classnames)}')
        self.load_zs_classname_embeddings(args)
        self.init_classifier_weights_gpt3(args, gpt3_prompts)
        
        if args.train_config['use_handcrafted']:
            print("Using handcrafted textual prototypes as classifier.")
            weights = self.classname_embeddings
        else:
            print("Using CuPL textual prototypes as classifier.")
            weights = self.zs_weights
        self.classifier = nn.Parameter(copy.deepcopy(weights).to(args.device))
        if args.use_unr_token:
            self.unr_token = nn.Parameter(torch.zeros(1, 768).to(args.device)) # for the unrelated token
        
        self.query_proj = Linear(768).to(args.device) # for query projection in the attention pooling
        self.key_proj = Linear(768).to(args.device) # for key projection in the attention pooling
        self.value_proj = Linear(768).to(args.device) # for value projection in the attention pooling
        
        if not args.vis:
            if args.train_config['source_model'] == 'SigLIP':
                del self.clip_model.model.text_model
                del self.clip_model.model.text_projection
                del self.clip_model.model.logit_scale
            else:
                del self.clip_model.transformer, self.clip_model.token_embedding
                del self.clip_model.positional_embedding, self.clip_model.ln_final
                del self.clip_model.text_projection, self.clip_model.logit_scale
            
        self.zs_encoder = copy.deepcopy(self.clip_model.visual)
        self.device = args.device
        
        self.layer_activations = []

    def load_zs_classname_embeddings(self, args):
        classname_embeddings = []
        with torch.no_grad():
            print(f"Using {['single', 'ensembled'][args.train_config['prompt_type'] == 'multi']} handcrafted prompts for zero-shot class embeddings.")
            for classname in tqdm(self.classnames, desc="Loading zero-shot class embeddings"):
                if args.train_config['prompt_type'] == 'multi':
                    # Use the mean of the handcrafted ensemble of prompts
                    prompt = [template.format(classname.replace("_", " ")) for template in self.templates]
                elif args.train_config['prompt_type'] == 'single':
                    # Use a single handcrafted prompt
                    prompt = CUSTOM_TEMPLATES[args.dataset].format(classname.replace("_", " "))
                tokenized_prompt = self.tokenize(prompt).to(args.device)
                class_embedding = self.clip_model.encode_text(tokenized_prompt)
                if args.train_config['prompt_type'] == 'multi':
                    class_embedding /= class_embedding.norm()
                    class_embedding = class_embedding.mean(dim=0)
                else:
                    class_embedding = class_embedding.squeeze(0)
                class_embedding /= class_embedding.norm()
                classname_embeddings.append(class_embedding)
        classname_embeddings = torch.stack(classname_embeddings, dim=0).to(args.device)
        classname_embeddings /= classname_embeddings.norm(dim=-1, keepdim=True)
        self.classname_embeddings = classname_embeddings

    def init_classifier_weights_gpt3(self, args, gpt3_prompts):
        all_desc_embeddings = []
        all_text_labels = []
        all_texts = [[] for _ in range(len(self.classnames))]

        with torch.no_grad():
            zs_weights = []
            for class_idx, classname in tqdm(enumerate(self.classnames), desc="Loading GPT3 prompts", total=len(self.classnames)):
                ## ---------------------Get the class label for each desc----------------------------------
                texts = gpt3_prompts[classname]
                all_texts[class_idx].extend(texts)
                all_text_labels.extend([class_idx] * len(texts))
                tokenized_texts  = self.tokenize(texts).to(args.device) #tokenize
                class_embeddings = self.clip_model.encode_text(tokenized_texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                all_desc_embeddings.append(class_embeddings)
                if not hasattr(self.clip_model, "classifier"):
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zs_weights.append(class_embedding)
                
        self.zs_weights = torch.stack(zs_weights, dim=0).to(args.device)
        self.all_texts = all_texts

        text_scale = 1
        max_len = max([len(emb) for emb in all_desc_embeddings])
        all_desc_embeddings = [F.pad(emb, (0, 0, 0, max_len - len(emb))) for emb in all_desc_embeddings]
        all_desc_embeddings = [F.normalize(emb, dim=-1) for emb in all_desc_embeddings]
        self.all_desc_embeddings = torch.stack(all_desc_embeddings) # (num_classes, num_descriptions, embed_dim)
        all_desc_weights = [text_scale * cls_desc_emb @ cls_weight.unsqueeze(1).to(args.device) \
            for cls_desc_emb, cls_weight in zip(all_desc_embeddings, self.classname_embeddings)]
        self.all_desc_weights = torch.stack(all_desc_weights).squeeze(-1) # (num_classes, num_descriptions)
        self.text_labels = torch.tensor(all_text_labels).to(args.device)

    def get_layer_activation_hook(self, weights):

        def inner_hook(module, input, output):
            inp = input[1].transpose(1, 0)  # all inputs to attn are same, does not matter. Can also do input[0] or input[2] 
            output_qkv = torch.matmul(inp, weights.transpose(0, 1))  # (B, 197, dim * 3)
            B, T, _ = output_qkv.shape
            num_heads = self.clip_model.visual.transformer.resblocks[-1].attn.num_heads
            output_qkv = output_qkv.reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
            k_feats = output_qkv[1].transpose(1, 2).reshape(B, T, -1)
            self.layer_activations.append(k_feats)
        
        return inner_hook
    
    def clear_layer_activations(self):
        self.layer_activations = []

    def extract_last_layer_key_feats(self, input_imgs):
        input_imgs = input_imgs.to(self.device)
        
        handlers = [self.clip_model.visual.transformer.resblocks[-1].attn.register_forward_hook(self.get_layer_activation_hook(self.clip_model.visual.transformer.resblocks[-1].attn.in_proj_weight))]
        
        img_feat, cls_token = self(input_imgs, return_cls_token=True) 
            
        feats = self.layer_activations[0]
            
        for h in handlers:
            h.remove()
            
        return img_feat, cls_token, feats.float()

    def forward(self, images, **kwargs):
        image_features = self.clip_model.visual(images, **kwargs)
        return image_features

    def get_classifier(self):
        return F.normalize(self.classifier, dim=-1)
    
    def get_descriptions(self):
        return self.all_desc_embeddings, self.all_desc_weights
    
    def get_fixed_classifier(self):
        all_desc_embeds, all_desc_weights = self.get_descriptions()
        all_desc_embeds = all_desc_embeds * all_desc_weights.unsqueeze(-1)
        sum_desc_weights = all_desc_weights.sum(dim=-1, keepdim=True)
        all_desc_embeds = all_desc_embeds.sum(dim=1) / sum_desc_weights
        zs_weights = F.normalize(all_desc_embeds, dim=-1)
        return zs_weights
    
    def encode_text(self, texts):
        return self.clip_model.encode_text(texts)