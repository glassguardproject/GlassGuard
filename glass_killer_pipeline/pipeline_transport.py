#!/usr/bin/env python3
"""Same-machine pipeline transport for the 3-node Glass Killer build.

FAT payloads (masks, clouds) exceed the 16 MB UDP DDS buffer (no-SHM profile), so we
do NOT push them through a ROS topic. Instead each stage writes its payload dict as a
pickle into /dev/shm (RAM-backed, ~zero-copy fast) and publishes ONLY a tiny
std_msgs/String on a topic: "<stamp_ns>\t<path>". The next node reads + unlinks it.

- bool masks are np.packbits-ed (÷8 RAM/IO) by the caller via pack_masks()/unpack_masks().
- drop-to-latest: newer notifications supersede older; readers unlink stale files.
"""
import os
import pickle
import glob
import numpy as np

SHM_DIR = "/dev/shm/gkpipe"
os.makedirs(SHM_DIR, exist_ok=True)


def stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def pack_masks(masks_bool_list):
    """[(H,W) bool] -> [(shape, packed_uint8)] for compact pickling."""
    out = []
    for m in masks_bool_list:
        m = np.ascontiguousarray(np.asarray(m, bool))
        out.append((m.shape, np.packbits(m)))
    return out


def unpack_masks(packed):
    out = []
    for shape, packed_u8 in packed:
        n = int(np.prod(shape))
        out.append(np.unpackbits(packed_u8, count=n).astype(bool).reshape(shape))
    return out


def write_payload(pub, header, payload: dict, tag: str):
    """Pickle payload to /dev/shm and publish '<stamp>\\t<path>' on `pub` (String)."""
    from std_msgs.msg import String
    sns = stamp_ns(header)
    path = os.path.join(SHM_DIR, f"{tag}_{sns}.pkl")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)                      # atomic: reader never sees a half file
    msg = String()
    msg.data = f"{sns}\t{path}"
    pub.publish(msg)
    _reap(tag, keep_stamp=sns)                 # drop our own stale outputs


def read_payload(msg, tag: str):
    """Parse a String notification, load + unlink the pickle. Returns (stamp_ns, dict)
    or (None, None) if the file vanished (superseded)."""
    try:
        sns_s, path = msg.data.split("\t", 1)
        sns = int(sns_s)
    except Exception:
        return None, None
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None, None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    _reap(tag, keep_stamp=sns)
    return sns, payload


def _reap(tag: str, keep_stamp: int):
    """Unlink stale <tag>_*.pkl older than keep_stamp so RAM can't grow unbounded."""
    for p in glob.glob(os.path.join(SHM_DIR, f"{tag}_*.pkl")):
        try:
            s = int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0])
            if s < keep_stamp:
                os.unlink(p)
        except (ValueError, IndexError, OSError):
            pass
