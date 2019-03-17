import torch.nn as nn
from typing import Any, Mapping
from catalyst.dl.experiments import SupervisedRunner


class CustomRunner(SupervisedRunner):
    def __init__(
        self,
        model: nn.Module = None,
        device=None,
        input_key: str = "features",
        output_key: str = "logits",
        align_key: str = "alignments",
        input_target_key: str = "targets",
    ):
        super().__init__(
            model=model,
            input_key=input_key,
            output_key=output_key,
            input_target_key=input_target_key,
            device=device,
        )
        self.align_key = align_key

    def _batch2device(self, batch: Mapping[str, Any], device):
        if isinstance(batch, (tuple, list)):
            assert len(batch) == 3
            batch = {
                self.input_key: batch[0],
                self.align_key: batch[1],
                self.target_key: batch[2],
            }
        batch = super()._batch2device(batch, device)
        return batch

    def predict_batch(self, batch: Mapping[str, Any]):
        output = self.model(
            (
                batch[self.input_key][0].to(self.device),
                batch[self.input_key][1].to(self.device),
                batch[self.align_key].to(self.device),
            )
        )
        output = {self.output_key: output}
        return output
