import re

VOLUME_RE = re.compile(r"(?<!\w)(\d+[.,]?\d*)\s?(л|л\.|литр\w*|мл|ml|г|кг|шт|уп\w*|пак\w*|бут\w*)(?!\w)", re.IGNORECASE)
PERCENT_RE = re.compile(r"(?<!\w)(\d+[.,]?\d*)\s?%|процент\w*", re.IGNORECASE)

def rule_spans(text):
    spans = []
    for m in VOLUME_RE.finditer(text):
        spans.append((m.start(), m.end(), "VOLUME"))
    for m in PERCENT_RE.finditer(text):
        spans.append((m.start(), m.end(), "PERCENT"))
    return spans

def apply_priors(text, words, probs, beta=2.0):
    offsets = []
    char_idx = 0
    for w in words:
        start = text.find(w, char_idx)
        end = start + len(w)
        offsets.append((start, end))
        char_idx = end + 1

    for s, e, label in rule_spans(text):
        for i, (w_start, w_end) in enumerate(offsets):
            if max(s, w_start) < min(e, w_end):
                if label == "VOLUME":
                    probs[i][3] += beta
                    probs[i][7] += beta
                elif label == "PERCENT":
                    probs[i][1] += beta
                    probs[i][5] += beta
    return probs
