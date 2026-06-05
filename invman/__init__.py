def get_config(*args, **kwargs):
    from invman.config import get_config as _get_config

    return _get_config(*args, **kwargs)

__all__ = ["get_config"]
