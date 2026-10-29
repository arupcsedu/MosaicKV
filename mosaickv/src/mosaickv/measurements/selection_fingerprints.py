"""Compact sideband fingerprints of exact selected positions, never model inputs."""

import hashlib
from collections import defaultdict


def selection_fingerprints(records):
    """One streaming pass; input is the canonical packed slot order.

    Source IDs and raw K/V are deliberately excluded. Layer/head/slot/logical
    position and tier define the selected exact cache independently of catalogs.
    Storage identity and bytes are measured separately, never inferred here.
    """
    hashes, counts = {}, defaultdict(int)
    for row in records:
        layer, head = int(row["layer"]), int(row["kv_head"])
        if row["tier"] != "exact":
            raise ValueError("selection fingerprint diagnostic supports exact slots only")
        key = (layer, head)
        digest = hashes.setdefault(key, hashlib.sha256())
        digest.update(f"{int(row['slot'])}:{int(row['logical_position'])}\n".encode())
        counts[key] += 1
    return [
        {
            "layer": layer,
            "kv_head": head,
            "positions": counts[layer, head],
            "sha256": digest.hexdigest(),
        }
        for (layer, head), digest in sorted(hashes.items())
    ]
