"""
图神经网络模型实现
"""

# Lazy imports to allow using non-PyG modules without installing PyG
def __getattr__(name):
    if name == "GCN":
        from .gcn import GCN
        return GCN
    if name == "GAT":
        from .gat import GAT
        return GAT
    if name == "GraphSAGE":
        from .sage import GraphSAGE
        return GraphSAGE
    if name == "HeteroGNN":
        from .hetero_gnn import HeteroGNN
        return HeteroGNN
    raise AttributeError(f"module 'src.models' has no attribute '{name}'")

__all__ = ["GCN", "GAT", "GraphSAGE", "HeteroGNN"]
