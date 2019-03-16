from typing import Any, Mapping, Dict, List
from catalyst.dl.experiments import SupervisedRunner


class CustomRunner(SupervisedRunner):
    def predict_batch(self, batch: Mapping[str, Any]):
        output = self.model(
            (
                batch[self.input_key][0].to(self.device),
                batch[self.input_key][1].to(self.device),
            )
        )
        output = {self.output_key: output}
        return output
