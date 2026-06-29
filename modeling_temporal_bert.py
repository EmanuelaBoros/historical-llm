from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

# ---------------------------------------------------------------------
# Period mapping
# ---------------------------------------------------------------------


PERIODS = [
    "pre_1850",
    "1850_1899",
    "1900_1938",
    "1939_1945",
    "post_1945",
]


def year_to_period_id(year: int) -> int:
    if year < 1850:
        return 0
    if 1850 <= year <= 1899:
        return 1
    if 1900 <= year <= 1938:
        return 2
    if 1939 <= year <= 1945:
        return 3
    return 4


# ---------------------------------------------------------------------
# Temporal adapter
# ---------------------------------------------------------------------


class TemporalAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_size, hidden_size)

        # Start close to identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.up(
            self.dropout(self.activation(self.down(hidden_states)))
        )


class TemporalAdapterBank(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_periods: int,
        bottleneck_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.adapters = nn.ModuleList(
            [
                TemporalAdapter(
                    hidden_size=hidden_size,
                    bottleneck_size=bottleneck_size,
                    dropout=dropout,
                )
                for _ in range(num_periods)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        period_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden_states: [batch, seq_len, hidden]
        period_ids: [batch]
        """

        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)

        if period_ids.dim() == 0:
            period_ids = period_ids.unsqueeze(0)

        period_ids = period_ids.to(hidden_states.device).long()

        batch_size = hidden_states.shape[0]

        if period_ids.shape[0] != batch_size:
            raise ValueError(
                f"period_ids batch size does not match hidden_states batch size: "
                f"period_ids={tuple(period_ids.shape)}, "
                f"hidden_states={tuple(hidden_states.shape)}"
            )

        output = hidden_states.clone()

        for period_id, adapter in enumerate(self.adapters):
            mask = period_ids == period_id

            if mask.any():
                output[mask, :, :] = adapter(hidden_states[mask, :, :])

        return output


# ---------------------------------------------------------------------
# Historical Temporal BERT
# ---------------------------------------------------------------------


class HistoricalTemporalBertForMLM(nn.Module):
    """
    Simpler stable version:

    input
      -> BERT encoder
      -> temporal adapter selected by document period
      -> MLM head

    This avoids manually looping over BERT layers.
    """

    def __init__(
        self,
        base_model_name: str = "dbmdz/bert-base-french-europeana-cased",
        num_periods: int = len(PERIODS),
        adapter_bottleneck_size: int = 64,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()

        self.base = AutoModelForMaskedLM.from_pretrained(base_model_name)

        hidden_size = self.base.config.hidden_size

        self.temporal_adapter_bank = TemporalAdapterBank(
            hidden_size=hidden_size,
            num_periods=num_periods,
            bottleneck_size=adapter_bottleneck_size,
            dropout=adapter_dropout,
        )

        self.num_periods = num_periods

    @property
    def config(self):
        return self.base.config

    def get_input_embeddings(self):
        return self.base.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.base.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        period_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> MaskedLMOutput:

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        if labels is not None and labels.dim() == 1:
            labels = labels.unsqueeze(0)

        batch_size = input_ids.shape[0]

        if period_ids is None:
            period_ids = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=input_ids.device,
            )

        if period_ids.dim() == 0:
            period_ids = period_ids.unsqueeze(0)

        period_ids = period_ids.to(input_ids.device).long()

        # Official BERT forward pass
        bert_outputs = self.base.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        sequence_output = bert_outputs.last_hidden_state

        # Apply temporal adapter after BERT encoder
        sequence_output = self.temporal_adapter_bank(
            hidden_states=sequence_output,
            period_ids=period_ids,
        )

        prediction_scores = self.base.cls(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=None,
            attentions=None,
        )
