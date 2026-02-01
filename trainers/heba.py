import os
import os.path as osp
import sys
from collections import OrderedDict
import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from .gsl import gradient_scale_layer
from .gpt3_prompts import load_CuPL_templates

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(state_dict or model.state_dict())
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, adapater_parser=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        if adapater_parser == None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, adapater_parser])
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = -log_preds.sum(dim=-1)
        loss = loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction='none')
        return loss.mean()

class RobustAdapterBlock(nn.Module):
    def __init__(self, input_dim, reduction=4, is_visual=False):
        super().__init__()
        self.is_visual = is_visual
        
        hidden_dim = input_dim // reduction
        
        if self.is_visual:
            # 2D Conv for images (Spatial awareness)
            self.down = nn.Conv2d(input_dim, hidden_dim, 1, bias=False)
            self.up = nn.Conv2d(hidden_dim, input_dim, 1, bias=False)
            # Spatial Context (Depthwise Conv)
            self.mid = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        else:
            # Linear for text (Semantic awareness)
            self.down = nn.Linear(input_dim, hidden_dim, bias=False)
            self.up = nn.Linear(hidden_dim, input_dim, bias=False)
            self.mid = nn.Identity()

        self.act = nn.GELU()
        self.norm = nn.LayerNorm(input_dim)
        
        # --- INITIALIZATION: Kaiming Normal (No Zero-Init) ---
        nn.init.kaiming_normal_(self.up.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        residual = x
        
        if self.is_visual:
            # Reshape for Conv2D: [N, B, D] -> [B, D, H, W]
            n_token, b_size, d_model = x.shape
            p_size = int(math.sqrt(n_token))
            x_img = x.permute(1, 2, 0).reshape(b_size, d_model, p_size, p_size)
            
            y = self.down(x_img)
            y = self.mid(y)
            y = self.act(y)
            y = self.up(y)
            
            # Back to [N, B, D]
            y = y.reshape(b_size, d_model, -1).permute(2, 0, 1)
        else:
            # Text: Just Linear
            y = self.down(x)
            y = self.mid(y)
            y = self.act(y)
            y = self.up(y)
            
        return self.norm(residual + y)

class AdapterLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.clip_dtype = clip_model.dtype

        self.text_adapter_parser = lambda x : self.return_text_adapter(x)
        self.text_adapted_func = lambda x, y, z: self.return_text_adapted_x(x, y, z)
        
        self.visual_adapter_parser = lambda x : self.return_visual_adapter(x)
        self.visual_adapted_func = lambda x, y, z: self.return_visual_adapted_x(x, y, z)

        # Build Adapters
        self.text_adapter = self._build_adapter(
            clip_model.ln_final.weight.shape[0], # 512
            len(clip_model.transformer.resblocks), 
            cfg.TRAINER.HeBA.ADAPTER_START,
            cfg.TRAINER.HeBA.ADAPTER_END,
            is_visual=False,
            channel_dim=clip_model.ln_final.weight.shape[0] * 4 # 2048
        )
        
        self.visual_adapter = self._build_adapter(
            clip_model.visual.ln_post.weight.shape[0], # 768
            len(clip_model.visual.transformer.resblocks), 
            cfg.TRAINER.HeBA.ADAPTER_START,
            cfg.TRAINER.HeBA.ADAPTER_END,
            is_visual=True,
            channel_dim=clip_model.visual.ln_post.weight.shape[0] * 4 # 3072
        )

        self._build_text_prompts(cfg, classnames)
        
        self.scale_base = float(cfg.TRAINER.HeBA.ADAPTER_SCALE_BASE)
        self.scale_novel = float(cfg.TRAINER.HeBA.ADAPTER_SCALE_NOVEL)
        
        # Flag to manually toggle during inference if needed
        self.use_novel_scale = False 

        self.adapter_scale_factor = float(cfg.TRAINER.HeBA.ADAPTER_SCALE_FACTOR)
        self.slow_fast_ratio = cfg.TRAINER.HeBA.SLOW_FAST_RATIO

    def _build_text_prompts(self, cfg, classnames):
        text_ctx_init = cfg.TRAINER.HeBA.TEXT_CTX_INIT
        
        # Standard CuPL Prompt Loading
        prompt_ctxs = load_CuPL_templates(cfg.DATASET.NAME)
        prompt_ctxs = {k.lower().replace("_", " "): v for k, v in prompt_ctxs.items()}
            
        classnames = [name.replace("_", " ") for name in classnames]
        tk_prompts = []
        for cname in classnames:
            key = cname.lower().replace("_", " ")
            suffix = prompt_ctxs.get(key, [])
            if not suffix: suffix = [f"a photo of a {cname}."]
            clean_prompts = []
            for p in suffix:
                 if "{}" in p: clean_prompts.append(p.format(cname))
                 else: clean_prompts.append(p)
            prompts = [text_ctx_init + " " + cname + ", " + ctx for ctx in clean_prompts]
            prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts])
            tk_prompts.append(prompts)
        max_len = max([p.shape[0] for p in tk_prompts])
        padded_prompts = []
        for p in tk_prompts:
            if p.shape[0] < max_len:
                diff = max_len - p.shape[0]
                padding = p[0].unsqueeze(0).repeat(diff, 1)
                padded_prompts.append(torch.cat([p, padding]))
            else:
                padded_prompts.append(p)
        self.register_buffer("tk_prompts", torch.stack(padded_prompts, dim=0))

    def _build_adapter(self, d_model, n_layers, l_start, l_end, is_visual=False, channel_dim=None):
        adapter = [None] * (n_layers + 1)
        for i in range(l_start, l_end+1):
            adapter[i] = nn.Sequential(OrderedDict([
                ("att_conv", RobustAdapterBlock(d_model, is_visual=is_visual)),
                ("mlp_conv", RobustAdapterBlock(channel_dim, is_visual=is_visual))
            ]))
        return nn.ModuleList([a for a in adapter])

    @property
    def current_scale(self):
        # Return Novel scale if explicitly toggled, else Base
        return self.scale_novel if self.use_novel_scale else self.scale_base

    def return_text_adapter(self, index):
        active_scale = self.current_scale
        
        if torch.rand(1) > self.slow_fast_ratio and self.training:
            scale = active_scale * self.adapter_scale_factor
        else:
            scale = active_scale
        return self.text_adapter[index], scale, self.text_adapted_func
    
    def return_text_adapted_x(self, x, adapter, scale):
        if adapter is None: return x
        y = gradient_scale_layer(x, scale)
        y = adapter(y)
        y = gradient_scale_layer(y*scale, 1.0/scale)
        return x + y

    def return_visual_adapter(self, index):
        active_scale = self.current_scale
        
        if torch.rand(1) > self.slow_fast_ratio and self.training:
            scale = active_scale * self.adapter_scale_factor
        else:
            scale = active_scale
        return self.visual_adapter[index], scale, self.visual_adapted_func
    
    def return_visual_adapted_x(self, x, adapter, scale):
        if adapter is None: return x
        
        n_token = x.shape[0]
        if x.dim() == 3: 
            cls_token, x_patches = torch.split(x, [1, n_token-1], dim=0)
            
            y = gradient_scale_layer(x_patches, scale)
            y = adapter(y)
            y = gradient_scale_layer(y*scale, 1.0/scale)
            
            x_patches = x_patches + y
            x = torch.cat([cls_token, x_patches], dim=0)
        else:
            y = gradient_scale_layer(x, scale)
            y = adapter(y)
            y = gradient_scale_layer(y*scale, 1.0/scale)
            x = x + y
            
        return x

    def update_adapter_scale(self, scale_factor):
        # Manual scheduling update
        self.scale_base = self.scale_base * scale_factor
        self.scale_novel = self.scale_novel * scale_factor

    def forward(self):
        return self.text_adapter_parser, self.visual_adapter_parser

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.adapter_learner = AdapterLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        self.all_cls = torch.arange(0, len(classnames))
        self.neg_sampling_ratio = cfg.TRAINER.HeBA.NEG_SAMPLING_RATIO
        self.text_features_for_inference = None

    def encode_text(self, prompts, tk_prompts, adapter_parser=None):
        if adapter_parser is not None:
            text_features = self.text_encoder(prompts.type(self.dtype), tk_prompts.type(self.dtype), adapter_parser)
        else:
            text_features = self.text_encoder(prompts.type(self.dtype), tk_prompts.type(self.dtype))
        return text_features
    
    def encode_image(self, image, adapter_parser=None):
        if adapter_parser is not None:
            image_features = self.image_encoder([image.type(self.dtype), adapter_parser])
        else:
            image_features = self.image_encoder(image.type(self.dtype))
        return image_features

    def forward(self, image, label=None):
        text_adapter_parser, visual_adapter_parser = self.adapter_learner()

        if self.adapter_learner.training:
            tk_prompts = self.adapter_learner.tk_prompts
            n_cls, n_temp = tk_prompts.shape[0:2]

            if self.neg_sampling_ratio >= 0 and self.all_cls.shape[0] > image.size(0):
                self.all_cls = self.all_cls.to(label.device)
                pos_c, inversed_c = torch.unique(label, return_inverse=True)
                pos_prompts = tk_prompts[pos_c]

                if self.neg_sampling_ratio > 0:
                    neg_c = [c not in pos_c for c in self.all_cls]
                    neg_indices = torch.nonzero(torch.tensor(neg_c, device=label.device)).squeeze()
                    if neg_indices.numel() > 0:
                        neg_prompts = tk_prompts[neg_indices]
                        n_neg = min(neg_prompts.shape[0], len(pos_c) * self.neg_sampling_ratio)
                        i_neg = torch.randperm(neg_prompts.shape[0])[:n_neg]
                        neg_prompts = neg_prompts[i_neg]
                        tk_prompts = torch.cat([pos_prompts, neg_prompts], dim=0)
                    else:
                        tk_prompts = pos_prompts
                else:
                    tk_prompts = pos_prompts
                    
                n_cls = tk_prompts.shape[0]
                label = inversed_c

            iid = torch.randint(0, n_temp, (1, n_cls), dtype=torch.long)
            prompts = tk_prompts[torch.arange(n_cls), iid].squeeze(0)

            with torch.no_grad():
                embedding = self.token_embedding(prompts)
                
            text_features = self.encode_text(embedding, prompts, text_adapter_parser)
            image_features = self.encode_image(image, visual_adapter_parser)

            text_features = F.normalize(text_features, dim=-1)
            image_features = F.normalize(image_features, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            return logits, label
        else:
            if self.text_features_for_inference is not None:
                text_features = self.text_features_for_inference
            else:
                tk_prompts = self.adapter_learner.tk_prompts
                n_cls, n_temp = tk_prompts.shape[0:2]
                mean_text_features = 0
                for iid in range(n_temp):
                    prompts = tk_prompts[:, iid]
                    with torch.no_grad():
                        embedding = self.token_embedding(prompts)
                    text_features = self.encode_text(embedding, prompts, text_adapter_parser)
                    text_features = F.normalize(text_features, dim=-1)
                    mean_text_features += text_features
                mean_text_features /= n_temp
                self.text_features_for_inference = F.normalize(mean_text_features, dim=1)
                text_features = self.text_features_for_inference
            
            image_features = self.encode_image(image, visual_adapter_parser)
            image_features = F.normalize(image_features, dim=-1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            return logits

@TRAINER_REGISTRY.register()
class HeBA(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.HeBA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.HeBA.PREC == "fp32" or cfg.TRAINER.HeBA.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "text_adapter" not in name and "visual_adapter" not in name:
                param.requires_grad_(False)

        num_trainable_params = 0
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                num_trainable_params += param.data.nelement()
        print(f"Parameters to be updated: {enabled}")
        print(f"Number of trainable parameters: {num_trainable_params}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.adapter_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("adapter_learner", self.model.adapter_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.HeBA.PREC == "amp" else None
        
        self.criterion = LabelSmoothingCrossEntropy(eps=0.1)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.HeBA.PREC
        if prec == "amp":
            with autocast():
                logits, label = self.model(image, label)
                loss = self.criterion(logits, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits, label = self.model(image, label)
            loss = self.criterion(logits, label)
            self.model_backward_and_update(loss)
        loss_summary = {"loss": loss.item(), "n_cls": logits.shape[1]}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory: return
        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None: model_file = "model.pth.tar-" + str(epoch)
        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path): raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            if "text_features_for_inference" in state_dict: del state_dict["text_features_for_inference"]
            if "tk_prompts" in state_dict: del state_dict["tk_prompts"]
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)