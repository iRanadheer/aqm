import re

_CODE = re.compile(r"\d+(?:_\d+)+")          # e.g. 0_0_0, 4_2_7_1, 7_2_1


def parse_claims(content: str) -> list[str]:
    """Extract claim codes from a model `content` field (YAML answer only;
    reasoning lives in the separate `reasoning_content` column).

    Returns codes in order, deduped. Empty list for `categories: []`,
    a no-claim response, or unparseable/None content.
    """
    if not isinstance(content, str):
        return []
    m = re.search(r"```(?:yaml)?\s*(.*?)```", content, re.DOTALL)
    body = m.group(1) if m else content                     # prefer fenced block
    cm = re.search(r"categories:\s*(\[\s*\]|(?:\s*-\s*.+)+)", body)
    if not cm:
        return []
    seen, out = set(), []
    for c in _CODE.findall(cm.group(1)):
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def super_claims(content: str) -> list[str]:
    """Super-claims for a row: the leading number of each code (x_x_x -> x),
    deduped (e.g. 4_2_7_1 -> '4'). One per super-claim present in the row."""
    seen, out = set(), []
    for c in parse_claims(content):
        s = c.split("_", 1)[0]
        if s not in seen:
            seen.add(s); out.append(s)
    return out
