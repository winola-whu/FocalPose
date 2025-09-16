# focalpose/lib3d/label_resolver.py
import re
from difflib import get_close_matches

def _tokenize(s: str):
    return [t for t in re.split(r'[_\\-\\s]+', s.lower()) if t]

def resolve_single(name: str, registry_keys):
    """
    Map a dataset label -> a key in 'registry_keys' (list of strings).
    Returns the canonical key if found, else None.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    keys = list(registry_keys)
    lower2canon = {k.lower(): k for k in keys}
    reg_lc = list(lower2canon.keys())

    sl = name.strip().lower()

    # 0) exact
    if sl in lower2canon:
        return lower2canon[sl]

    # 1) dash/underscore variants
    v1, v2 = sl.replace('-', '_'), sl.replace('_', '-')
    if v1 in lower2canon: return lower2canon[v1]
    if v2 in lower2canon: return lower2canon[v2]

    # 2) prefix completion: "shoe-sky" -> any "shoe-sky_*"
    pref = [k for k in keys if k.lower().startswith(sl + '_') or k.lower().startswith(sl + '-')]
    if len(pref) == 1: return pref[0]
    if len(pref) > 1:  return min(pref, key=len)

    # 3) substring anywhere
    sub = [k for k in keys if sl in k.lower()]
    if len(sub) == 1: return sub[0]
    if len(sub) > 1:  return min(sub, key=len)

    # 4) token-overlap (>= 60%)
    st = set(_tokenize(sl))
    if st:
        best, score = None, 0.0
        for k in keys:
            kt = set(_tokenize(k))
            if not kt: continue
            sc = len(st & kt) / float(len(st))
            if sc > score:
                best, score = k, sc
        if best is not None and score >= 0.6:
            return best

    # 5) difflib as a last resort
    m = get_close_matches(sl, reg_lc, n=1, cutoff=0.8)
    if m:
        return lower2canon[m[0]]

    return None
