"""Tokenizers: a dependency-free character tokenizer plus a HF adapter."""

from __future__ import annotations

from typing import List, Sequence, Optional
import json
import os


class BaseTokenizer:
    name = "base"

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def save(self, path: str) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer. Zero dependencies, deterministic, and good
    enough for smoke tests and TinyStories-scale demos."""

    name = "char"

    def __init__(self, alphabet: Optional[str] = None):
        # Default alphabet covers the synthetic corpus and ordinary English.
        if alphabet is None:
            alphabet = (
                "\n !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
            )
        itos = ["<pad>"] + list(dict.fromkeys(alphabet))  # id 0 = pad
        self.itos: List[str] = itos
        self.stoi = {s: i for i, s in enumerate(itos)}

    def encode(self, text: str) -> List[int]:
        unk = self.stoi.get("<pad>")
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[i] if 0 <= i < len(self.itos) else "?" for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "alphabet": "".join(self.itos[1:])}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(alphabet=d["alphabet"])


class HFTokenizerAdapter(BaseTokenizer):
    """Adapter around ``transformers.AutoTokenizer`` (optional dependency)."""

    name = "hf"

    def __init__(self, hf_name: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "transformers is required for tokenizer='" + hf_name + "'. "
                "Install with: pip install 'shimballm[hf]'"
            ) from e
        self.tok = AutoTokenizer.from_pretrained(hf_name)

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    def save(self, path: str) -> None:
        self.tok.save_pretrained(path)


def build_tokenizer(name: str) -> BaseTokenizer:
    if name == "char":
        return CharTokenizer()
    return HFTokenizerAdapter(name)
