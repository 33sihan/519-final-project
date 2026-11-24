"""
model.py  —  CIS 5190 Project B (News Source Classification), v4

Backend contract:
    * class NewsClassifier / Model instantiable with no arguments, or get_model()
    * model.pt is loaded as a state_dict into the model
    * model.predict(batch) returns one prediction per input (0 = Fox, 1 = NBC)

Design ("wide & deep" fastText-style neural network):
  headline -> hashed features:
      word unigrams, bigrams, trigrams, character 2-5-grams, first/last word,
      headline length
  * deep part: EmbeddingBag(mean) over all features -> MLP -> 2 logits
  * wide part: one learned weight per feature per class, summed (= logistic
    regression on the same features)
  logits = deep + wide
Features are hashed with crc32 (deterministic), so there is no vocabulary file:
everything lives in model.pt as a plain state_dict. The model is trained on the
4,800 provided links plus ~2.1M scraped Fox/NBC links (see train_big.py).
"""
import zlib

import torch
import torch.nn as nn

# Index 0 = Fox News, index 1 = NBC (same order as preprocess.LABELS).
# OUTPUT_INTS = True returns 0 (Fox) / 1 (NBC); the backend maps ints to its labels.
LABELS = ("FoxNews", "NBC")
OUTPUT_INTS = True

DEEP_BUCKETS = 2 ** 18
WIDE_BUCKETS = 2 ** 22
EMB_DIM = 64
HIDDEN = 256
CHAR_NGRAMS = (2, 3, 4, 5)

# Optional third branch: CNN over the hashed word sequence (phrases / word order).
USE_CNN = False
WORD_BUCKETS = 2 ** 17
WORD_DIM = 96
CNN_CHANNELS = 128
CNN_KERNELS = (1, 2, 3)
MAX_WORDS = 30


def _crc(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def text_to_features(text: str):
    words = str(text).lower().split()
    feats = ["<bias>", "len:%d" % min(len(words), 25)]
    if words:
        feats += ["first:" + words[0], "last:" + words[-1]]
    feats += ["w:" + w for w in words]
    feats += ["b:" + a + "_" + b for a, b in zip(words, words[1:])]
    feats += ["t:" + a + "_" + b + "_" + c for a, b, c in zip(words, words[1:], words[2:])]
    for w in words:
        padded = "<" + w + ">"
        for n in CHAR_NGRAMS:
            if len(padded) >= n:
                feats += ["c:" + padded[i:i + n] for i in range(len(padded) - n + 1)]
    return [_crc(f) for f in feats]


def text_to_wordids(text: str):
    """Hashed word ids for the CNN branch (0 is padding)."""
    words = str(text).lower().split()[:MAX_WORDS]
    return [_crc("w:" + w) % WORD_BUCKETS + 1 for w in words]


def pad_wordids(seqs):
    width = max(max((len(s) for s in seqs), default=0), max(CNN_KERNELS))
    out = torch.zeros(len(seqs), width, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def _to_text_list(batch):
    if isinstance(batch, str):
        return [batch]
    if isinstance(batch, torch.Tensor):
        raise TypeError("NewsClassifier expects a list of headline strings")
    return ["" if t is None else str(t) for t in list(batch)]


class NewsClassifier(nn.Module):
    def __init__(self, emb_dim: int = EMB_DIM, hidden: int = HIDDEN, dropout: float = 0.3,
                 sparse: bool = False):
        super().__init__()
        self.emb = nn.EmbeddingBag(DEEP_BUCKETS, emb_dim, mode="mean", sparse=sparse)
        self.deep = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.wide = nn.EmbeddingBag(WIDE_BUCKETS, 2, mode="sum", sparse=sparse)
        nn.init.normal_(self.emb.weight, std=0.05)
        nn.init.zeros_(self.wide.weight)
        self.use_cnn = USE_CNN
        if self.use_cnn:
            self.wemb = nn.Embedding(WORD_BUCKETS + 1, WORD_DIM, padding_idx=0, sparse=sparse)
            nn.init.normal_(self.wemb.weight, std=0.05)
            self.convs = nn.ModuleList(nn.Conv1d(WORD_DIM, CNN_CHANNELS, k) for k in CNN_KERNELS)
            self.cnn_out = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(CNN_CHANNELS * len(CNN_KERNELS), 2))

    def encode(self, texts):
        hashes, offsets = [], []
        for t in texts:
            offsets.append(len(hashes))
            hashes.extend(text_to_features(t))
        device = self.emb.weight.device
        h = torch.tensor(hashes, dtype=torch.long, device=device)
        o = torch.tensor(offsets, dtype=torch.long, device=device)
        if self.use_cnn:
            return h, o, pad_wordids([text_to_wordids(t) for t in texts]).to(device)
        return h, o

    def forward_encoded(self, h, offsets, wids=None):
        deep = self.deep(self.emb(h % DEEP_BUCKETS, offsets))
        wide = self.wide(h % WIDE_BUCKETS, offsets)
        out = deep + wide
        if self.use_cnn and wids is not None:
            x = self.wemb(wids).transpose(1, 2)                        # [N, D, T]
            pooled = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
            out = out + self.cnn_out(torch.cat(pooled, dim=1))
        return out

    def forward(self, batch):
        """batch: list of headline strings -> logits [N, 2]."""
        return self.forward_encoded(*self.encode(_to_text_list(batch)))

    @torch.no_grad()
    def predict(self, batch):
        self.eval()
        texts = _to_text_list(batch)
        preds = []
        for i in range(0, len(texts), 2048):
            preds.extend(self.forward(texts[i:i + 2048]).argmax(dim=1).tolist())
        if OUTPUT_INTS:
            return preds
        return [LABELS[p] for p in preds]


# Aliases the backend looks for
Model = NewsClassifier


def get_model():
    return NewsClassifier()
