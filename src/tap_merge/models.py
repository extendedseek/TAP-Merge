"""Small offline causal LM for integration checks and optional Hugging Face backbones."""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .archive import atomic_save, safe_load, cpu_tree, fingerprint


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x):
        b, s, h = x.shape
        q, k, v = self.qkv(x).reshape(b, s, 3, self.heads, h // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.out(y.transpose(1, 2).reshape(b, s, h))


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = Attention(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab_size=256, width=48, layers=2, heads=4, context=128):
        super().__init__()
        self.config_dict = dict(vocab_size=vocab_size, width=width, layers=layers, heads=heads, context=context)
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*[Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token(input_ids) + self.position(positions)
        return self.head(self.norm(self.blocks(x)))


def logits(model, x):
    output = model(input_ids=x)
    return output if isinstance(output, torch.Tensor) else output.logits


def create_base(output, kind="tiny", model_id=None, revision=None, seed=42, config_json=None):
    torch.manual_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.pt").exists():
        raise FileExistsError("Base exists; choose a new output directory")
    if kind == "tiny":
        settings = json.loads(Path(config_json).read_text()) if config_json else {}
        model = TinyLM(**settings)
        info = dict(kind="tiny", config=model.config_dict, seed=seed, purpose="offline smoke test")
    else:
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
        from huggingface_hub import model_info
        if config_json:
            config = AutoConfig.for_model(**json.loads(Path(config_json).read_text()))
            model = AutoModelForCausalLM.from_config(config)
            info = dict(kind="hf", config=model.config.to_dict(), seed=seed, initialization="random")
        else:
            if not model_id or not revision:
                raise ValueError("Supply model_id and pinned revision, or explicit architecture JSON")
            resolved = model_info(model_id, revision=revision).sha
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=resolved, torch_dtype=torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=resolved)
            tokenizer.save_pretrained(output / "tokenizer")
            info = dict(kind="hf", config=model.config.to_dict(), model_id=model_id,
                        model_revision=resolved, seed=seed, initialization="pretrained")
    model = model.float()
    if info["kind"] == "hf":
        model.config.use_cache = False
    atomic_save(cpu_tree(model.state_dict()), output / "model.pt")
    info["fingerprint"] = fingerprint(model.state_dict())
    (output / "model.json").write_text(json.dumps(info, indent=2))
    return info


def load_model(base, weights=None, device="cpu"):
    base = Path(base)
    info = json.loads((base / "model.json").read_text())
    if info["kind"] == "tiny":
        model = TinyLM(**info["config"])
    else:
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = dict(info["config"])
        kind = cfg.pop("model_type")
        model = AutoModelForCausalLM.from_config(AutoConfig.for_model(kind, **cfg))
        model.config.use_cache = False
    model.load_state_dict(safe_load(weights or base / "model.pt"), strict=True)
    return model.to(device).float(), info
