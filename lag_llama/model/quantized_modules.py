import torch
from torch import nn
from lag_llama.model.module import RMSNorm, CausalSelfAttention, LTSMConfig, MLP, find_multiple
from torch.nn import functional as F
from typing import List, Optional
from gluonts.torch.distributions import DistributionOutput
from gluonts.torch.scaler import MeanScaler, NOPScaler, StdScaler
from gluon_utils.scalers.robust_scaler import RobustScaler
import math

from gluonts.torch.util import lagged_sequence_values, unsqueeze_expand


class QuantizedBlock(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd_per_head * config.n_head)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd_per_head * config.n_head)
        self.mlp = QuantizedMLP(config)

    def forward(self, x: torch.Tensor, use_kv_cache: bool) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x), use_kv_cache)
        y = x + self.mlp(self.rms_2(x))
        return y
    

class QuantizedMLP(nn.Module):
    def __init__(self, config: LTSMConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd_per_head * config.n_head
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.c_fc1 = nn.Linear(
            config.n_embd_per_head * config.n_head, n_hidden, bias=False
        )
        self.c_fc2 = nn.Linear(
            config.n_embd_per_head * config.n_head, n_hidden, bias=False
        )
        self.c_proj = nn.Linear(
            n_hidden, config.n_embd_per_head * config.n_head, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return self.dequant(x)
    

class QuantizedLagLlamaModel(nn.Module):
    def __init__(
        self,
        context_length: int,
        max_context_length: int,
        scaling: str,
        input_size: int,
        n_layer: int,
        n_embd_per_head: int,
        n_head: int,
        lags_seq: List[int],
        distr_output: DistributionOutput,
        rope_scaling=None,
        num_parallel_samples: int = 100,
        time_feat: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.lags_seq = lags_seq
        if time_feat:
            feature_size = input_size * (len(self.lags_seq)) + 2 * input_size + 6
        else:
            feature_size = input_size * (len(self.lags_seq)) + 2 * input_size

        config = LTSMConfig(
            n_layer=n_layer,
            n_embd_per_head=n_embd_per_head,
            n_head=n_head,
            block_size=max_context_length,
            feature_size=feature_size,
            rope_scaling=rope_scaling,
            dropout=dropout,
        )
        self.num_parallel_samples = num_parallel_samples

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True, dim=1)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True, dim=1)
        elif scaling == "robust":
            self.scaler = RobustScaler(keepdim=True, dim=1)
        else:
            self.scaler = NOPScaler(keepdim=True, dim=1)

        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_args_proj(
            config.n_embd_per_head * config.n_head
        )

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(
                    config.feature_size, config.n_embd_per_head * config.n_head
                ),
                h=nn.ModuleList([QuantizedBlock(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd_per_head * config.n_head),
            )
        )
        self.y_cache = False  # used at time of inference when kv cached is used

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer)
            )

    def prepare_input(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ):
        scaled_past_target, loc, scale = self.scaler(
            past_target, past_observed_values
        )  # Data is standardized (past_observed_values is passed as "weights" parameter) # (bsz, context_length+max(self.lags_seq)

        # In the below code, instead of max(self.lags_seq), it was previously -self.context_length
        if future_target is not None:
            input = torch.cat(
                (
                    scaled_past_target[..., max(self.lags_seq) :],  # Just the context
                    (future_target[..., :-1] - loc)
                    / scale,  # Not sure about the -1 here. Maybe so since the last value isn't used in the model for prediction of any new values. also if the prediction length is 1, this doesn't really affect anything
                ),
                dim=-1,
            )  # Shape is (bsz, context_length+(pred_len-1))
        else:
            input = scaled_past_target[..., max(self.lags_seq) :]
        if (past_time_feat is not None) and (future_time_feat is not None):
            time_feat = (
                torch.cat(
                    (
                        past_time_feat[..., max(self.lags_seq) :, :],
                        future_time_feat[..., :-1, :],
                    ),
                    dim=1,
                )
                if future_time_feat is not None
                else past_time_feat[..., max(self.lags_seq) :, :]
            )

        prior_input = (
            past_target[..., : max(self.lags_seq)] - loc
        ) / scale  # This the history used to construct lags.  # bsz, max(self.lags_seq)

        lags = lagged_sequence_values(
            self.lags_seq, prior_input, input, dim=-1
        )  # Lags are added as an extra dim. Shape is (bsz, context_length+(pred_len-1), len(self.lags_seq))

        static_feat = torch.cat(
            (loc.abs().log1p(), scale.log()), dim=-1
        )  # (bsz, 2) (loc and scale are concatenated)
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=-2, size=lags.shape[-2]
        )  # (bsz, context_length+(pred_len-1), 2)
        # expanded_static_feat: (bsz, context_length+(pred_len-1), len(self.lags_seq) + 2); (bsz, 1); (bsz, 1)

        if past_time_feat is not None:
            return (
                torch.cat((lags, expanded_static_feat, time_feat), dim=-1),
                loc,
                scale,
            )
        else:
            return torch.cat((lags, expanded_static_feat), dim=-1), loc, scale

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
    ) -> torch.Tensor:
        # if past_time_feat is not None:
        transformer_input, loc, scale = self.prepare_input(
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_target=future_target,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
        )  # return: (bsz, context_length+(pred_len-1), len(self.lags_seq) + 2); (bsz, 1); (bsz, 1)
        # To use kv cache for inference and pass recent token to transformer
        if use_kv_cache and self.y_cache:
            # Only use the most recent one, rest is in cache
            transformer_input = transformer_input[:, -1:]

        # forward the LLaMA model itself
        x = self.transformer.wte(
            transformer_input
        )  # token embeddings of shape (b, t, n_embd_per_head*n_head) # (bsz, context_length+(pred_len-1), n_embd_per_head*n_head)

        for block in self.transformer.h:
            x = block(x, use_kv_cache)
        x = self.transformer.ln_f(
            x
        )  # (bsz, context_length+(pred_len-1), n_embd_per_head*n_head)
        if use_kv_cache:
            self.y_cache = True
        params = self.param_proj(
            x
        )  # (bsz, context_length+(pred_len-1)) ; (bsz, context_length+(pred_len-1))
        return params, loc, scale

    def reset_cache(self) -> None:
        """
        Resets all cached key-values in attention.
        Has to be called after prediction loop in predictor
        """
        for block in self.transformer.h:
            block.y_cache = None
            block.attn.kv_cache = None
