from aiomap.core.types import Device


def resolve_device(device: Device) -> Device:
    if device == 'cpu':
        return device
    else:
        from torch.cuda import is_available

        has_gpu = is_available()
        if device == 'auto':
            device = 'cuda' if has_gpu else 'cpu'
        elif device =='cuda' and not has_gpu:
            raise RuntimeError(
                "Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and CUDA toolkit installed."
            )
