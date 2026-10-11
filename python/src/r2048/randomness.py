"""随机种子在运行边界兑现；断点恢复使用已保存的实际值。"""

import secrets
from pathlib import Path

from r2048.fileio import read_json, write_json_atomic
from r2048.identity import file_lock


def resolve_seed(seed: int) -> int:
    if seed < -1:
        raise ValueError("随机种子必须为 -1 或非负整数")
    return secrets.randbits(63) if seed == -1 else seed


def persist_seeds(path: Path, settings: dict, names: tuple[str, ...]) -> dict:
    with file_lock(path.with_suffix(".lock")):
        saved = read_json(path) if path.exists() else {}
        for name in names:
            seed = int(settings[name])
            if seed == -1 and name in saved:
                settings[name] = saved[name]
            else:
                settings[name] = resolve_seed(seed)
        write_json_atomic(path, {name: settings[name] for name in names})
    return settings
