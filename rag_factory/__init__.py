"""
rag_factory — AI RAG Factory
NVIDIA-inspired, Temporal-ready pipeline engine over 33 AWS Bedrock + Qdrant RAG patterns.
"""
from .spec       import PipelineSpec, MANIFEST, VALIDATOR
from .assembler  import Assembler

__version__ = "0.1.0"
__all__     = ["PipelineSpec", "MANIFEST", "VALIDATOR", "Assembler"]
