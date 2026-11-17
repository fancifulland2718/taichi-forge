"""Cold build identities, independent of artifact storage and compile purpose.

Only immutable input digests cross this boundary. A frontend/producer may use
different stages without retaining Python owners, IR, or resource lifetimes.
There is deliberately no lookup on the ordinary JIT invocation path.
"""

from dataclasses import dataclass
import hashlib
import json


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode("utf8")).hexdigest()


def unchanged_file_digest(path, previous, current):
    """Reuse a local development toolchain digest while its file stamp agrees.

    Artifact admission still checks actual bytes. This memo is for trusted local
    compiler inputs, not a signature verifier or an untrusted cache protocol.
    """
    from pathlib import Path

    # The stamp contains the file identity; path aliases may miss the memo but
    # cannot turn a replacement file into the previous input. Avoid an extra
    # filesystem canonicalization for every source file in a warm build.
    path = Path(path).absolute()
    stat = path.stat()
    stamp = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    old = previous.get(str(path), {}) if isinstance(previous, dict) else {}
    if not isinstance(old, dict):
        old = {}
    digest = old.get("sha256") if old.get("stamp") == stamp else None
    if not isinstance(digest, str) or len(digest) != 64:
        digest = file_digest(path)
    current[str(path)] = dict(stamp=stamp, sha256=digest)
    return digest


@dataclass(frozen=True)
class BuildIdentity:
    stage: str
    policy: int
    inputs: tuple

    @classmethod
    def create(cls, stage, policy, inputs):
        return cls(stage, policy, tuple(sorted(inputs.items())))

    @property
    def key(self):
        return value_digest((self.stage, self.policy, self.inputs))
