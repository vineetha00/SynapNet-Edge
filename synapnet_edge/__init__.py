"""SynapNet-Edge: Hybrid long-context architecture for consumer hardware."""
from synapnet_edge.models.synapnet_edge_model import SynapNetEdge, SynapNetEdgeConfig
from synapnet_edge.quantization.cajq import apply_cajq
from synapnet_edge.memory.baee import BAEEMemoryManager

__version__ = "0.1.0"
__all__ = ["SynapNetEdge", "SynapNetEdgeConfig", "apply_cajq", "BAEEMemoryManager"]
