"""Tokenizer BPE byte-level (murni Python, tanpa dependensi).

ID 0-255 = byte mentah, 256=<pad>, 257=<bos>, 258=<eos>, 259+ = hasil merge.
Teks dipecah dulu menjadi 'kata' (pre-tokenisasi) agar training cepat dan merge
tidak melintasi batas kata. Format file kompatibel dengan versi lama proyek.
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PAD_ID, BOS_ID, EOS_ID = 256, 257, 258
BASE_VOCAB = 259
SPECIAL_NAMES = {PAD_ID: "<pad>", BOS_ID: "<bos>", EOS_ID: "<eos>"}

_PRETOKEN = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+", re.UNICODE)


def _pretokenize(text: str):
    return _PRETOKEN.findall(text)


class ByteBPETokenizer:
    def __init__(self, merges=None):
        self.merges = [tuple(m) for m in (merges or [])]
        self.ranks = {pair: r for r, pair in enumerate(self.merges)}
        self.vocab_bytes = [bytes([i]) for i in range(256)] + [b"", b"", b""]
        for a, b in self.merges:
            self.vocab_bytes.append(self.vocab_bytes[a] + self.vocab_bytes[b])
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_bytes)

    # ---------- training ----------
    @classmethod
    def train(cls, text: str, vocab_size: int = 4096, min_frequency: int = 2):
        word_counts = Counter(_pretokenize(text))
        words = [list(w.encode("utf-8")) for w in word_counts]
        freqs = list(word_counts.values())
        pair_counts: Counter = Counter()
        where = defaultdict(set)

        def add(i):
            w, f = words[i], freqs[i]
            for pair in zip(w, w[1:]):
                pair_counts[pair] += f
                where[pair].add(i)

        def remove(i):
            w, f = words[i], freqs[i]
            for pair in zip(w, w[1:]):
                pair_counts[pair] -= f

        for i in range(len(words)):
            add(i)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)
        merges = []
        while BASE_VOCAB + len(merges) < vocab_size and heap:
            neg, pair = heapq.heappop(heap)
            cur = pair_counts.get(pair, 0)
            if cur != -neg:
                if cur > 0:
                    heapq.heappush(heap, (-cur, pair))
                continue
            if cur < min_frequency:
                break
            new_id = BASE_VOCAB + len(merges)
            merges.append(pair)
            touched = list(where[pair])
            for i in touched:
                remove(i)
                w, out, j = words[i], [], 0
                while j < len(w):
                    if j + 1 < len(w) and (w[j], w[j + 1]) == pair:
                        out.append(new_id)
                        j += 2
                    else:
                        out.append(w[j])
                        j += 1
                words[i] = out
                add(i)
            for i in touched:
                for p in zip(words[i], words[i][1:]):
                    if pair_counts[p] > 0:
                        heapq.heappush(heap, (-pair_counts[p], p))
            pair_counts.pop(pair, None)
        return cls(merges)

    # ---------- encode / decode ----------
    def _bpe_word(self, word: str):
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids = list(word.encode("utf-8"))
        while len(ids) > 1:
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            new_id, out, j = BASE_VOCAB + best_rank, [], 0
            while j < len(ids):
                if j + 1 < len(ids) and (ids[j], ids[j + 1]) == best:
                    out.append(new_id)
                    j += 2
                else:
                    out.append(ids[j])
                    j += 1
            ids = out
        if len(self._cache) < 200_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        ids = [BOS_ID] if add_bos else []
        for word in _pretokenize(text):
            ids.extend(self._bpe_word(word))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids):
        raw = bytearray()
        for i in ids:
            if i in SPECIAL_NAMES:
                continue
            if 0 <= i < len(self.vocab_bytes):
                raw.extend(self.vocab_bytes[i])
        return raw.decode("utf-8", errors="replace")

    # ---------- simpan / muat ----------
    def save(self, path):
        payload = {
            "type": "byte-bpe",
            "merges": [list(m) for m in self.merges],
            "special_tokens": {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID},
            "vocab_size": self.vocab_size,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["merges"])


def main():
    parser = argparse.ArgumentParser(description="Latih tokenizer BPE byte-level")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("input")
    parser.add_argument("--output", default="outputs/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=4096)
    args = parser.parse_args()
    tok = ByteBPETokenizer.train(Path(args.input).read_text(encoding="utf-8"), args.vocab_size)
    tok.save(args.output)
    print(f"Tokenizer disimpan: {tok.vocab_size} ID token -> {args.output}")


if __name__ == "__main__":
    main()
