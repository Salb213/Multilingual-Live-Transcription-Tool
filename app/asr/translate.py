from functools import lru_cache
from transformers import MarianMTModel, MarianTokenizer

@lru_cache(maxsize=2)
def _load(name: str):
    tok = MarianTokenizer.from_pretrained(name)
    mod = MarianMTModel.from_pretrained(name)
    return tok, mod

def translate_to_de(text: str, src_lang: str) -> str:
    if not text.strip():
        return ""
    s = (src_lang or "").lower()
    if s.startswith("pl"):
        model = "Helsinki-NLP/opus-mt-pl-de"
    elif s.startswith("sk"):
        model = "Helsinki-NLP/opus-mt-sk-de"
    else:
        return ""
    tok, mod = _load(model)
    batch = tok([text], return_tensors="pt", padding=True, truncation=True)
    out = mod.generate(**batch, max_new_tokens=256)
    return tok.batch_decode(out, skip_special_tokens=True)[0]
