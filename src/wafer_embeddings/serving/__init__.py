"""Inference-side packaging: checkpoint -> embeddings (batch, pyfunc, app)."""

from wafer_embeddings.serving.encoder import WaferEncoder, build_encoder, encode_map, decode_map

__all__ = ["WaferEncoder", "build_encoder", "encode_map", "decode_map"]
