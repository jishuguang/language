from yacs.config import CfgNode


_C = CfgNode(new_allowed=True)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    return _C.clone()
