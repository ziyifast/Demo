"""
数据处理模块
"""

# Lazy imports to allow using non-PyG modules without installing PyG
def __getattr__(name):
    if name in ("load_cora", "load_fraud_dataset"):
        from .dataset import load_cora, load_fraud_dataset
        return locals()[name]
    if name in ("build_homogeneous_graph", "build_heterogeneous_graph"):
        from .graph_builder import build_homogeneous_graph, build_heterogeneous_graph
        return locals()[name]
    if name == "generate_fraud_dataset":
        from .generate_fraud_data import generate_fraud_dataset
        return generate_fraud_dataset
    raise AttributeError(f"module 'src.data' has no attribute '{name}'")

__all__ = [
    "load_cora",
    "load_fraud_dataset",
    "build_homogeneous_graph",
    "build_heterogeneous_graph",
    "generate_fraud_dataset",
]
