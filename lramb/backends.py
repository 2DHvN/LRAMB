"""Resolve an optional accelerator without importing PyTorch for NumPy runs."""


def backend_options(backend="auto",device="auto",dtype="float64"):
    if backend not in {"auto","numpy","torch"}:
        raise ValueError("backend must be auto, numpy or torch")
    if dtype not in {"float64","float32"}:
        raise ValueError("dtype must be float64 or float32")
    if backend=="numpy":
        return dict(backend="numpy",device="cpu",dtype="float64")
    try:
        import torch
    except ImportError:
        if backend=="auto":
            return dict(backend="numpy",device="cpu",dtype="float64")
        raise ImportError('PyTorch backend requires PyTorch: pip install -e ".[torch]"') from None
    if backend=="auto":
        backend="torch" if torch.cuda.is_available() else "numpy"
    if backend=="numpy":
        return dict(backend="numpy",device="cpu",dtype="float64")
    if device=="auto":
        device="cuda" if torch.cuda.is_available() else "cpu"
    target=torch.device(device)
    if target.type not in {"cpu","cuda"}:
        raise ValueError("The torch simulator supports CPU and CUDA devices")
    if target.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch installation has no available CUDA device")
    return dict(backend="torch",device=str(target),dtype=dtype)
