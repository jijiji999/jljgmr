"""A00 retargeting, adapted from General Motion Retargeting (see LICENSE-GMR)."""
__all__ = ["A00Retargeter"]


def __getattr__(name):
    if name == "A00Retargeter":
        from .retarget import A00Retargeter
        return A00Retargeter
    raise AttributeError(name)
