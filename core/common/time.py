"""Time-related helpers."""


def _format_time(t: float) -> str:
    """mm:ss.mmm"""
    negative = False
    if t < 0:
        negative = True
        t = abs(t)

    mm, ss = divmod(int(t), 60)
    mmm = int((t - int(t)) * 1_000)

    if negative:
        return f"-{mm:02d}:{ss:02d}.{mmm:03d}"
    else:
        return f"{mm:02d}:{ss:02d}.{mmm:03d}"


__all__ = ["_format_time"]
