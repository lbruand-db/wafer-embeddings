"""MLflow pyfunc wrapper around :class:`WaferEncoder` (SPECS.md §10, §11.4).

Input: a DataFrame with ``height``, ``width`` and ``wafer_map`` (the flattened
row-major map, cells in {0 = no die, 1 = pass, 2 = fail}) — the Delta/parquet layout.
Output: an ``embedding`` column of L2-normalized float vectors. All logic lives in
``WaferEncoder`` (CI-tested); this file is the thin MLflow adapter.
"""

from __future__ import annotations

import mlflow.pyfunc  # ty: ignore[unresolved-import]

CHECKPOINT = "checkpoint"  # artifact key holding the training checkpoint (.pt)


class WaferEncoderModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:
        from wafer_embeddings.serving.encoder import WaferEncoder

        self.encoder = WaferEncoder.from_checkpoint(context.artifacts[CHECKPOINT])

    def predict(self, context, model_input, params=None):
        import pandas as pd

        emb = self.encoder.embed_rows(
            model_input["height"].tolist(),
            model_input["width"].tolist(),
            model_input["wafer_map"].tolist(),
        )
        return pd.DataFrame({"embedding": [row.tolist() for row in emb]})
