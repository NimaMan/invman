
from .es_module import ESModule


def load_model(directory, device="cpu", strict=True):
    return ESModule.load(directory, device=device, strict=strict)
