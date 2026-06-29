from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling


@dataclass
class TemporalDataCollatorForMLM:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15

    def __post_init__(self):
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )

    def __call__(self, examples: list[dict]) -> dict:
        period_ids = torch.tensor(
            [example.pop("period_ids") for example in examples],
            dtype=torch.long,
        )

        batch = self.mlm_collator(examples)
        batch["period_ids"] = period_ids

        return batch
