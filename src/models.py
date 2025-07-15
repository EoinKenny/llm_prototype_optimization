import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.nn.init as init
import gc

from typing import Dict, Any, Optional, Tuple, Union

from transformers import (
    # ModernBertConfig, ModernBertModel,
    # BertConfig, BertModel, BertTokenizer,
    # ElectraConfig, ElectraModel, ElectraTokenizer,
    # RobertaConfig, RobertaModel, RobertaTokenizer,
    AutoConfig, AutoModel, AutoTokenizer
)


def clean_gpus() -> None:
    gc.collect()
    torch.cuda.empty_cache() 
clean_gpus()


# --- Polarized Classification Layer ---
class PolarizedClassificationLayer(nn.Module):
    def __init__(self, num_classes, num_protos):
        """
        Args:
            num_classes: number of output classes.
            num_protos: number of prototypes per class.
            total_prototypes: total number of prototypes (num_classes * num_protos).
        """
        super(PolarizedClassificationLayer, self).__init__()
        # weight shape: (num_classes, total_prototypes)

        total_prototypes = num_classes * num_protos
        self.weight = nn.Parameter(torch.empty(num_classes, total_prototypes))
        # Initialize weights with +1 for the correct class and -1 for the rest.
        sign_matrix = -torch.ones(num_classes, total_prototypes)
        for j in range(total_prototypes):
            assigned_class = j // num_protos
            sign_matrix[assigned_class, j] = 1.
        self.register_buffer("sign_matrix", sign_matrix)
        
        # Initialize weights with the sign values.
        nn.init.constant_(self.weight, 0)
        self.weight.data.copy_(self.sign_matrix.clone())
        # Bias is fixed to zero.
        self.bias = None

    def forward(self, x):
        # Enforce polarity constraints:
        # For positions where sign_matrix is positive, ensure weight >= 0,
        # and where negative, ensure weight <= 0.
        with torch.no_grad():
            pos_mask = self.sign_matrix > 0
            neg_mask = self.sign_matrix < 0
            self.weight.data[pos_mask] = self.weight.data[pos_mask].clamp(min=0)
            self.weight.data[neg_mask] = self.weight.data[neg_mask].clamp(max=0)
        # Linear mapping: x shape is (batch_size, total_prototypes)
        return F.linear(x, self.weight, self.bias)
        

class LMProtoNet(nn.Module):
    def __init__(self, backbone: nn.Module,
                 num_labels: int = 2,
                 num_protos_per_class: int = 5,
                 init_prototypes: torch.Tensor = None):
        super().__init__()

        self.backbone = backbone
        self.num_labels = num_labels
        self.num_total_prototypes = num_protos_per_class * num_labels
        self.device = next(self.backbone.parameters()).device
        latent_size = self.backbone.latent_size

        if not self.backbone.no_llm_head and self.backbone.model_type == 'llm':
            latent_size=self.backbone.prototype_dim
        
        
        if self.backbone.model_type == 'bert':
            prototype_dim = latent_size
        elif self.backbone.model_type == 'llm':
            if not self.backbone.no_llm_head:
                prototype_dim = self.backbone.prototype_dim
            else:
                prototype_dim = latent_size
        else:
            raise NameError('Wrong model_type in LMProtoNet')


        # ----- prototypes -----
        if init_prototypes is not None:
            # keep user-provided prototypes exactly as they are
            assert init_prototypes.shape == (self.num_total_prototypes, prototype_dim)
            self.prototypes = nn.Parameter(init_prototypes.to(self.device))
        else:
            # Xavier-uniform instead of torch.rand
            proto_tensor = torch.empty(
                self.num_total_prototypes, prototype_dim, device=self.device
            )
            init.xavier_uniform_(proto_tensor, gain=1.0)   # or xavier_normal_
            self.prototypes = nn.Parameter(proto_tensor)

        # ----- linear classifier over prototype activations -----
        # self.classfn_model = nn.Linear(self.num_total_prototypes, num_labels, bias=False)
        # self.classfn_model = nn.Linear(latent_size, num_labels, bias=False)
        self.classfn_model = PolarizedClassificationLayer(self.num_labels, num_protos_per_class)
        # init.xavier_uniform_(self.classfn_model.weight, gain=1.0)
        self.classfn_model.to(self.device)


    def forward(self, input_ids=None, attention_mask=None, llm_encodings=None, forward_type='train'):

        rep = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            llm_encodings=llm_encodings,
            forward_type=forward_type,
        )
        
        cls_rep_norm = F.normalize(rep, p=2, dim=1)              # [B, H]
        proto_norm   = F.normalize(self.prototypes, p=2, dim=1)  # [P, H]

        # Cosine similarities
        loss_acts = 1 - (cls_rep_norm @ proto_norm.T)
        # print('experiment with making acts positive only')
        # breakpoint()
        acts = cls_rep_norm @ proto_norm.T
        
        l_p1 = loss_acts.min(dim=0).values.mean() 
        l_p2 = loss_acts.min(dim=1).values.mean()   

        # Prototype separation (upper-triangular, no diagonal)
        proto_sim = proto_norm @ proto_norm.T
        mask = torch.triu(torch.ones_like(proto_sim, dtype=torch.bool), diagonal=1)
        l_p3 = (1 + proto_sim[mask]).mean()

        # Classification logits
        logits = self.classfn_model(acts)                          # [B, C]
        # logits = self.classfn_model(rep)                         # [B, C]
        return {
            "logits": logits,
            "acts": acts,
            "cls_rep_normalized": cls_rep_norm,
            "l_p1": l_p1,
            "l_p2": l_p2,
            "l_p3": l_p3,
        }


class ModelWrapper(nn.Module):
    """
    Wrapper over transformer backbones, refactored into encoder and trainable_head modules.
    Modes:
        - 'enc': run input through encoder
        - 'train': run frozen encodings through trainable_head
        - 'full': encoder + trainable_head
    """
    def __init__(self, model_name: str, latent_dim: int = None, max_length: int = 128, prototype_dim=128, device='cuda:0', no_llm_head=True):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.prototype_dim = prototype_dim
        self.device = device
        self.no_llm_head = no_llm_head

        # Load backbone model and tokenizer
        if model_name not in {"bert", "electra", "llama", "qwen", "modern_bert", 'roberta', 'mpnet'}:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # Load backbone model and tokenizer
        if model_name == "bert":
            base = 'bert-base-uncased'
            config = AutoConfig.from_pretrained(base)
            self.config = config
            clean_gpus()
            self.hugging_model = AutoModel.from_pretrained(
                base,
                # config=config,
                token=hugging_token,
            )
            self.hugging_model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(base,token=hugging_token)
            self.model_type = "bert"

            # ── freeze everything first ─────────────────────────────────────────────
            for name, p in self.hugging_model.named_parameters():
                p.requires_grad = False
            
            # ── un-freeze the parts we want to fine-tune ────────────────────────────
            for name, p in self.hugging_model.named_parameters():
                if (name.startswith("encoder.layer.11.") or 
                    name.startswith("pooler.") or 
                    name.startswith("classifier.") or  # Common name for classification head
                    name.startswith("cls.")):          # Alternative name
                    p.requires_grad = True

            self.encoder = partial(CLSWrapper, self.hugging_model)
            self.latent_size = 768

            
        elif model_name == "electra":
            base = "google/electra-base-discriminator"
            config = AutoConfig.from_pretrained(base)
            self.config = config
            clean_gpus()
            self.hugging_model = AutoModel.from_pretrained(
                base,
                # config=config,
                token=hugging_token,
            )
            self.hugging_model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(base,token=hugging_token)
            self.model_type = "bert"

            # Freeze all Electra layers except the last one.
            for name, param in self.hugging_model.named_parameters():
                if "encoder.layer.11" not in name:      # keep only the last block trainable
                    param.requires_grad = False
                    
            self.encoder = partial(CLSWrapper, self.hugging_model)
            self.latent_size = 768

        elif model_name == "modern_bert":
            base = "answerdotai/ModernBERT-base"
            config = AutoConfig.from_pretrained(base)
            self.config = config
            clean_gpus()
            self.hugging_model = AutoModel.from_pretrained(
                base,
                # config=config,
                token=hugging_token,
            )
            self.hugging_model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(base,token=hugging_token)
            self.model_type = "bert"

            # How many encoder layers are there?
            n_layers = len(self.hugging_model.layers)   # should be 22 → indices 0 … 21
            last_layer_idx = n_layers - 1                     # 21
            
            # 1. Freeze everything.
            for p in self.hugging_model.parameters():
                p.requires_grad = False
            
            # 2. Un-freeze the last transformer block.
            for p in self.hugging_model.layers[last_layer_idx].parameters():
                p.requires_grad = True
                                
            self.encoder = partial(CLSWrapper, self.hugging_model)
            self.latent_size = 768          


        # RoBERTa
        elif model_name == "roberta":
            base = "FacebookAI/roberta-base"
            config = AutoConfig.from_pretrained(base)
            self.config = config
            clean_gpus()
            self.hugging_model = AutoModel.from_pretrained(
                base,
                # config=config,
                token=hugging_token,
            )
            self.hugging_model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(base,token=hugging_token)
            self.model_type = "bert"

            for name, param in self.hugging_model.named_parameters():
                # keep only the last encoder block (layer 11) and pooler trainable
                if ("encoder.layer.11." in name) or name.startswith("pooler"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.encoder = partial(CLSWrapper, self.hugging_model)
            self.latent_size = config.hidden_size

        # ALBERT
        elif model_name == "mpnet":
            base = "sentence-transformers/all-mpnet-base-v2"
            config = AutoConfig.from_pretrained(base)
            self.config = config
            clean_gpus()
            self.hugging_model = AutoModel.from_pretrained(
                base,
                # config=config,
                token=hugging_token,
            )
            self.hugging_model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(base,token=hugging_token)
            self.model_type = "bert"
            
            for name, param in self.hugging_model.named_parameters():
                if ("encoder.layer.11." in name) or name.startswith("pooler"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.encoder = partial(CLSWrapper, self.hugging_model)
            self.latent_size = config.hidden_size
            
            
        elif model_name == "llama":
            base = "meta-llama/Llama-3.2-3B"
            config = AutoConfig.from_pretrained(base)
            self.config = config
            self.latent_size = 3072
            self.model_type = "llm"
            clean_gpus()
            self.hugging_model = AutoModel.from_pretrained(
                base, 
                output_hidden_states=True, 
                return_dict=True, 
                device_map=self.device, 
                token=hugging_token,
                # torch_dtype=torch.float16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base,legacy=False,token=hugging_token)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                            
            self.llm_encoder = LlamaResidualWrapper(self.hugging_model)
            self.trainable_head = nn.Sequential(
                nn.Linear(self.latent_size, self.prototype_dim),
                nn.InstanceNorm1d(self.prototype_dim),
                nn.ReLU(),
                nn.Linear(self.prototype_dim, self.prototype_dim),
            ).to(self.hugging_model.device)
            self.device = self.hugging_model.device
            
            # freeze parameters
            for p in self.llm_encoder.parameters():
                p.requires_grad = False

            # Decide if we are training an MLP in the LLM
            if not self.no_llm_head:
                self.encoder = LLMEncoder(self.llm_encoder, self.trainable_head)
            else:
                self.encoder = self.llm_encoder


            
        else:  # qwen
            base = "Qwen/Qwen3-8B"
            config = AutoConfig.from_pretrained(base)
            self.config = config
            clean_gpus()
            self.hugging_model = AutoModel.from_pretrained(
                base,
                device_map=self.device, 
                output_hidden_states=True,
                return_dict=True,
                token=hugging_token,
                torch_dtype=torch.float16,
            )
            self.latent_size = 4096
            
            # tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base, token=hugging_token,)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model_type = "llm"
            total = config.num_hidden_layers
            self.llm_encoder = Qwen3ResidualWrapper(self.hugging_model)
            self.trainable_head = nn.Sequential(
                nn.Linear(self.latent_size, self.prototype_dim),
                nn.InstanceNorm1d(self.prototype_dim),
                nn.ReLU(),
                nn.Linear(self.prototype_dim, self.prototype_dim),
            ).to(self.hugging_model.device)
            self.device = self.hugging_model.device
            
            
            # freeze params
            for p in self.llm_encoder.parameters():
                p.requires_grad = False

            # Decide if we are training an MLP in the LLM
            
            if not no_llm_head:
                self.encoder = LLMEncoder(self.llm_encoder, self.trainable_head)
            else:
                self.encoder = self.llm_encoder
            
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        llm_encodings: torch.Tensor = None,
        forward_type: str = 'full',
    ) -> torch.Tensor:
        
        # --- collect_llm_encodings MODE ---
        # This is just for saving LLM data encodings for training later on
        # The entire network is always frozen
        if forward_type == 'collect_llm_encodings':

            with torch.no_grad():

                if input_ids is None or attention_mask is None:
                    raise ValueError("enc mode requires input_ids and attention_mask")
    
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
    
                if self.model_name == 'bert' or self.model_name == 'electra':
                    raise TypeError('Do not use BERT model in encoder setting, only for saving data')
                    
                elif self.model_name == 'llama' or self.model_name == 'qwen':
                    return self.llm_encoder(input_ids)
                    
                else:
                    raise NameError('wrong model name')
                
                
        # --- TRAIN MODE ---
        if forward_type == 'train':
            if self.model_type == 'bert':
                return self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            elif self.model_type == 'llm':
                if not self.no_llm_head:
                    return self.trainable_head(llm_encodings)
                else:
                    return llm_encodings
            else:
                raise NameError('wrong model name')

        # --- FULL MODE ---
        # This is for the optimization down the line with the LLM to find the prototype
        if forward_type == 'full':
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask)                
        raise ValueError(f"Unknown forward_type: {forward_type}")


class LlamaResidualWrapper(nn.Module):
    def __init__(self, base_model, residual_layer_idx=None):
        super().__init__()
        self.base_model = base_model
        
        # Calculate 2/3 through the network if not specified
        if residual_layer_idx is None:
            num_layers = len(base_model.layers)
            residual_layer_idx = int(num_layers * 2 / 3)
        
        self.residual_layer_idx = residual_layer_idx
        self.residual_output = None
        
        # Register hook on the target layer
        self._register_hook()

        self.device = self.base_model.device
    
    def _register_hook(self):
        """Register forward hook to capture residual output"""
        def hook_fn(module, input, output):
            # Store the residual output (first element of input is the residual)
            self.residual_output = input[0].clone()
        
        # Hook into the target layer
        target_layer = self.base_model.layers[self.residual_layer_idx]
        self.hook_handle = target_layer.register_forward_hook(hook_fn)
    
    def forward(self, *args, **kwargs):
        # Reset residual output
        self.residual_output = None
        
        # Forward pass through base model
        output = self.base_model(*args, **kwargs)
        
        return self.residual_output[:, -1, :]
    
    def get_residual_only(self, *args, **kwargs):
        """Helper method to get only the residual output"""
        result = self.forward(*args, **kwargs)
        return result['residual_2_3']
    
    def __del__(self):
        # Clean up hook when object is destroyed
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()

        
class Qwen3ResidualWrapper(nn.Module):
    def __init__(self, base_model, residual_layer_idx=None):
        super().__init__()
        self.base_model = base_model
        
        # Calculate 2/3 through the network
        num_layers = len(base_model.layers)
        if residual_layer_idx is None:
            residual_layer_idx = int(num_layers * 2 / 3)
        
        self.residual_layer_idx = residual_layer_idx
        self.residual_output = None
        
        # Register hook on the target layer
        self.hook = base_model.layers[residual_layer_idx].register_forward_hook(
            self._capture_residual
        )

        self.device = self.base_model.device
    
    def _capture_residual(self, module, input, output):
        """Hook function to capture the residual output"""
        self.residual_output = output
    
    def forward(self, *args, **kwargs):
        # Reset residual output
        self.residual_output = None
        
        # Forward pass through base model
        final_output = self.base_model(*args, **kwargs)
        
        return self.residual_output[0].float()[:, -1, :]
    
    def __del__(self):
        # Clean up hook when object is destroyed
        if hasattr(self, 'hook'):
            self.hook.remove()


def CLSWrapper(base_model, input_ids: torch.Tensor,attention_mask: torch.Tensor,**kwargs: Dict[str, Any]):
    input_ids = input_ids.to(base_model.device)
    attention_mask = attention_mask.to(base_model.device)
    outputs = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        **kwargs
    )
    return outputs.last_hidden_state[:, 0, :]   # [B, H]


class LLMEncoder(nn.Module):
    """
    Wraps a *causal-LM feature extractor* (e.g. LlamaResidualWrapper,
    Qwen3ResidualWrapper) **plus** an optional projection / head so that the
    resulting module exposes **the same forward signature** as the
    BERT-family encoders used elsewhere in your codebase:

        >>> out = encoder(input_ids=batch["input_ids"],
        ...               attention_mask=batch["attention_mask"])

    The wrapper:ƒbreaa
      • pushes all tensors to the correct device  
      • accepts `attention_mask` even if the underlying extractor ignores it  
      • returns the head’s output (usually of size `prototype_dim`)  
      • keeps a reference to the feature extractor’s raw embeddings
        in `self.last_hidden` for debugging / analysis, should you need them.
    """
    def __init__(
        self,
        feature_extractor: nn.Module,   # e.g. LlamaResidualWrapper(...)
        head: nn.Module,               # your trainable projection head
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head              = head

        self.device = next(self.feature_extractor.parameters()).device
        self.head.to(self.device)

        # # Expose useful sizes for downstream sanity-checks
        # self.latent_size        = head[-1].in_features     # dim before head
        # self.projected_size     = head[-1].out_features    # dim after head
        # self.last_hidden: Optional[torch.Tensor] = None    # filled at fwd-time

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    @torch.no_grad()     # encoders stay frozen; comment out if you fine-tune
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # Move to the right device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # (The residual wrappers accept arbitrary **kwargs, so we
        #  forward both tensors – the extractor will simply ignore
        #  attention_mask if it doesn’t need it.)
        self.last_hidden = self.feature_extractor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )                                   # shape [B, latent_size]

        # Projection / trainable head
        return self.head(self.last_hidden)  # shape [B, projected_size]




















