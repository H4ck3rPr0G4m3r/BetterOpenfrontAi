"""Policy network: autoregressive, intent-conditioned heads over the v2 layout.

Architecture (why each piece is shaped the way it is):

* Raster CNN over 11 planes at 64x128 -> 256-d summary. Keeps the
  full-resolution stem and the 16x32 feature map for the spatial head.
* Player slots and build sites are *sets*: each row goes through a shared
  MLP, then masked mean+max pooling. Slot order is not stable across steps
  (players move up and down the list), so nothing may depend on position.
* Heads are sampled intent-first. The target and build-site heads are
  pointer networks: they score each slot/site from its own embedding and
  the intent context, so "attack the weak neighbour that is attacking me"
  is a property of that player's features, not of slot index 3. The
  spatial head is a small conv decoder (translation equivariant) with one
  channel per spatial kind (boat landing, warship water).
* Only the heads an intent uses enter its log-probability and entropy. v1
  summed all five heads every step, so PPO's ratio was dominated by an
  8192-way spatial head that did nothing for most intents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import layout as L

NEG = -1e9


@dataclass
class ModelConfig:
    trunk: int = 256
    slot_dim: int = 64
    site_dim: int = 32
    context: int = 128
    channels: tuple = (32, 64, 64, 128)


def fp32(device: torch.device):
    """Disable autocast: heads and value run in fp32 even when the CNN runs
    in bf16. Per-step rewards are ~0.01, below bf16 resolution of a value
    or a log-ratio, so mixed precision there would be pure noise."""
    return torch.autocast(device_type=device.type, enabled=False)


def _mlp(*sizes: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers)


def masked_pool(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean and max over a set dimension, ignoring invalid rows."""
    w = valid.unsqueeze(-1).to(x.dtype)
    mean = (x * w).sum(1) / w.sum(1).clamp_min(1.0)
    mx = x.masked_fill(~valid.unsqueeze(-1), NEG).max(1).values
    mx = torch.where(valid.any(1, keepdim=True), mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=-1)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(~mask, NEG)


def categorical_stats(logits: torch.Tensor, mask: torch.Tensor, index: torch.Tensor):
    """Log-prob of ``index`` and entropy of the masked categorical."""
    logp = F.log_softmax(masked_logits(logits, mask), dim=-1)
    chosen = logp.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    p = logp.exp()
    entropy = -(p * logp.masked_fill(~mask, 0.0)).sum(-1)
    return chosen, entropy


def sample(logits: torch.Tensor, mask: torch.Tensor, deterministic: bool) -> torch.Tensor:
    masked = masked_logits(logits, mask)
    if deterministic:
        return masked.argmax(-1)
    return torch.distributions.Categorical(logits=masked).sample()


class OpenFrontPolicy(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        c0, c1, c2, c3 = c.channels
        conv = lambda a, b, s: nn.Sequential(nn.Conv2d(a, b, 3, stride=s, padding=1), nn.ReLU())
        self.stem = conv(L.RASTER_PLANES, c0, 1)  # 64x128
        self.down1 = conv(c0, c1, 2)  # 32x64
        self.down2 = conv(c1, c2, 2)  # 16x32
        self.down3 = conv(c2, c3, 2)  # 8x16
        self.raster_out = nn.Sequential(nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(), nn.Linear(c3 * 8, c.trunk), nn.ReLU())

        self.slot_mlp = _mlp(L.SLOT_FIELDS, c.slot_dim, c.slot_dim)
        self.site_mlp = _mlp(L.SITE_FIELDS, c.site_dim, c.site_dim)
        self.global_mlp = _mlp(L.NUM_GLOBALS, 64)
        joint = c.trunk + 2 * c.slot_dim + 2 * c.site_dim + 64
        self.trunk = _mlp(joint, c.trunk, c.trunk)

        self.value_head = nn.Sequential(_mlp(c.trunk, c.trunk), nn.Linear(c.trunk, 1))
        self.intent_head = nn.Linear(c.trunk, L.NUM_INTENTS)
        self.intent_embed = nn.Embedding(L.NUM_INTENTS, 32)
        self.context = _mlp(c.trunk + 32, c.context)
        self.ratio_head = nn.Linear(c.context, L.NUM_RATIOS)
        self.target_q = nn.Linear(c.context, c.slot_dim)
        self.target_k = nn.Linear(c.slot_dim, c.slot_dim)
        self.target_out = nn.Linear(c.slot_dim, 1)
        self.site_q = nn.Linear(c.context, c.site_dim)
        self.site_k = nn.Linear(c.site_dim, c.site_dim)
        self.site_out = nn.Linear(c.site_dim, 1)
        self.spatial_film = nn.Linear(c.trunk, c2)
        self.spatial_mid = conv(c2, c0, 1)
        self.spatial_out = nn.Conv2d(2 * c0, L.SPATIAL_KINDS, 1)

    # ----- encoding -------------------------------------------------------

    def encode(self, raster: torch.Tensor, features: torch.Tensor) -> dict:
        x = raster.float() * (1.0 / 255.0)
        s0 = self.stem(x)
        s2 = self.down2(self.down1(s0))
        s3 = self.down3(s2)
        with fp32(x.device):
            return self._encode_rest(s0.float(), s2.float(), s3.float(), features.float())

    def _encode_rest(self, s0, s2, s3, f) -> dict:
        r = self.raster_out(s3)
        slots = f[:, : L.SLOT_TABLE].reshape(-1, L.NUM_SLOTS, L.SLOT_FIELDS)
        sites = f[:, L.SLOT_TABLE : L.SLOT_TABLE + L.SITE_TABLE].reshape(-1, L.MAX_BUILD_SITES, L.SITE_FIELDS)
        glob = f[:, L.SLOT_TABLE + L.SITE_TABLE :]
        slot_e = self.slot_mlp(slots)
        site_e = self.site_mlp(sites)
        h = self.trunk(torch.cat([
            r,
            masked_pool(slot_e, slots[..., 0] > 0.5),
            masked_pool(site_e, sites[..., 0] > 0.5),
            self.global_mlp(glob),
        ], dim=-1))
        return {"h": h, "s0": s0, "s2": s2, "slot_e": slot_e, "site_e": site_e}

    def value(self, enc: dict) -> torch.Tensor:
        return self.value_head(enc["h"]).squeeze(-1)

    def _conditional_logits(self, enc: dict, intent: torch.Tensor) -> dict:
        ctx = self.context(torch.cat([enc["h"], self.intent_embed(intent)], dim=-1))
        q = self.target_q(ctx).unsqueeze(1)
        target = self.target_out(F.relu(q + self.target_k(enc["slot_e"]))).squeeze(-1)
        qs = self.site_q(ctx).unsqueeze(1)
        site = self.site_out(F.relu(qs + self.site_k(enc["site_e"]))).squeeze(-1)
        return {"target": target, "ratio": self.ratio_head(ctx), "site": site}

    def _spatial_logits(self, enc: dict) -> torch.Tensor:
        film = self.spatial_film(enc["h"])[:, :, None, None]
        mid = self.spatial_mid(F.relu(enc["s2"] + film))
        up = F.interpolate(mid, scale_factor=4, mode="nearest")
        return self.spatial_out(torch.cat([up, enc["s0"]], dim=1)).flatten(2)  # [B, kinds, cells]

    # ----- acting and evaluation -------------------------------------------

    @staticmethod
    def _rows(masks: torch.Tensor, intent: torch.Tensor) -> dict:
        """The conditional mask rows for each sample's intent."""
        b = torch.arange(masks.shape[0], device=masks.device)
        target = masks[:, L.MASK_TARGET : L.MASK_RATIO].view(-1, L.NUM_INTENTS, L.NUM_SLOTS)[b, intent]
        spatial_all = masks[:, L.MASK_SPATIAL : L.MASK_SITE].view(-1, L.SPATIAL_KINDS, L.RASTER_CELLS)
        site_all = masks[:, L.MASK_SITE :].view(-1, L.SITE_KINDS, L.MAX_BUILD_SITES)
        sk = SPATIAL_KIND.to(masks.device)[intent]
        tk = SITE_KIND.to(masks.device)[intent]
        spatial = spatial_all[b, sk.clamp_min(0)] & (sk >= 0).unsqueeze(-1)
        site = site_all[b, tk.clamp_min(0)] & (tk >= 0).unsqueeze(-1)
        return {
            "intent": masks[:, : L.MASK_TARGET],
            "target": target,
            "ratio": masks[:, L.MASK_RATIO : L.MASK_SPATIAL],
            "spatial": spatial,
            "site": site,
        }

    @torch.no_grad()
    def act(self, raster, features, masks, deterministic: bool = False) -> dict:
        """Sample actions. Returns actions [B,5], log-probs, values, and the
        conditional mask rows the update needs to re-evaluate them."""
        enc = self.encode(raster, features)
        with fp32(masks.device):
            return self._act(enc, masks, deterministic)

    def _act(self, enc, masks, deterministic):
        intent_mask = masks[:, : L.MASK_TARGET]
        intent = sample(self.intent_head(enc["h"]), intent_mask, deterministic)
        rows = self._rows(masks, intent)
        cond = self._conditional_logits(enc, intent)
        spatial_logits = self._spatial_logits(enc)
        sk = SPATIAL_KIND.to(masks.device)[intent].clamp_min(0)
        spatial_logits = spatial_logits[torch.arange(len(intent), device=masks.device), sk]
        # Unused heads still need a legal placeholder index to sample from.
        def safe(m):
            return torch.where(m.any(-1, keepdim=True), m, torch.ones_like(m))
        target = sample(cond["target"], safe(rows["target"]), deterministic)
        ratio = sample(cond["ratio"], safe(rows["ratio"]), deterministic)
        spatial = sample(spatial_logits, safe(rows["spatial"]), deterministic)
        site = sample(cond["site"], safe(rows["site"]), deterministic)
        use = usage(intent)
        actions = torch.stack([intent, target * use[0], ratio * use[1], spatial * use[2], site * use[3]], dim=-1)
        logp, _ = self._log_prob(enc, actions, rows, cond=cond, spatial_logits=spatial_logits)
        return {"actions": actions, "logp": logp, "value": self.value(enc), "rows": rows}

    def evaluate(self, raster, features, rows: dict, actions: torch.Tensor):
        """Log-prob, per-head entropy, and value of stored actions."""
        enc = self.encode(raster, features)
        with fp32(actions.device):
            logp, entropy = self._log_prob(enc, actions, rows)
            return logp, entropy, self.value(enc)

    def _log_prob(self, enc, actions, rows, cond=None, spatial_logits=None):
        intent = actions[:, 0]
        intent_logp, intent_ent = categorical_stats(self.intent_head(enc["h"]), rows["intent"], intent)
        if cond is None:
            cond = self._conditional_logits(enc, intent)
        if spatial_logits is None:
            sk = SPATIAL_KIND.to(intent.device)[intent].clamp_min(0)
            spatial_logits = self._spatial_logits(enc)[torch.arange(len(intent), device=intent.device), sk]
        use = usage(intent).float()
        safe = lambda m: torch.where(m.any(-1, keepdim=True), m, torch.ones_like(m))
        t_logp, t_ent = categorical_stats(cond["target"], safe(rows["target"]), actions[:, 1])
        r_logp, r_ent = categorical_stats(cond["ratio"], safe(rows["ratio"]), actions[:, 2])
        s_logp, s_ent = categorical_stats(spatial_logits, safe(rows["spatial"]), actions[:, 3])
        b_logp, b_ent = categorical_stats(cond["site"], safe(rows["site"]), actions[:, 4])
        logp = intent_logp + use[0] * t_logp + use[1] * r_logp + use[2] * s_logp + use[3] * b_logp
        entropy = {
            "intent": intent_ent,
            "target": use[0] * t_ent,
            "ratio": use[1] * r_ent,
            "spatial": use[2] * s_ent,
            "site": use[3] * b_ent,
        }
        return logp, entropy


USES_TARGET = torch.as_tensor(L.USES_TARGET)
USES_RATIO = torch.as_tensor(L.USES_RATIO)
SPATIAL_KIND = torch.as_tensor(L.SPATIAL_KIND)
SITE_KIND = torch.as_tensor(L.SITE_KIND)


def usage(intent: torch.Tensor) -> torch.Tensor:
    """[4, B] long: does each sample's intent use target, ratio, spatial, site."""
    d = intent.device
    return torch.stack([
        USES_TARGET.to(d)[intent],
        USES_RATIO.to(d)[intent],
        SPATIAL_KIND.to(d)[intent] >= 0,
        SITE_KIND.to(d)[intent] >= 0,
    ]).long()


# ----- checkpoints -----------------------------------------------------------

CHECKPOINT_FORMAT = "openfront-policy-v2"


def save_checkpoint(path, model: OpenFrontPolicy, meta: dict | None = None, optimizer=None) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "layout": L.layout_dict(),
        "model_config": asdict(model.config),
        "state_dict": model.state_dict(),
        "meta": meta or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, str(path))


def load_checkpoint(path, device="cpu") -> tuple[OpenFrontPolicy, dict]:
    """Load a policy checkpoint. Refuses v1 (SB3 .zip) and layout mismatches."""
    if str(path).endswith(".zip"):
        raise ValueError(f"{path} is a v1 SB3 checkpoint; the v2 policy cannot load it (retrain)")
    payload = torch.load(str(path), map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not an {CHECKPOINT_FORMAT} checkpoint")
    L.check_layout(payload["layout"])
    cfg = payload["model_config"]
    cfg["channels"] = tuple(cfg.get("channels", ModelConfig.channels))
    model = OpenFrontPolicy(ModelConfig(**cfg)).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, payload


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["ModelConfig", "OpenFrontPolicy", "load_checkpoint", "save_checkpoint", "parameter_count"]
