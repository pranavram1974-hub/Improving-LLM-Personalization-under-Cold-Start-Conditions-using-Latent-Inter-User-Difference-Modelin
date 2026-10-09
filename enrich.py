"""Rule-based cold-start preference enrichment (proposed module).

Turns the 1-5 reviews that are available for a cold-start user into a short,
explicit preference profile, e.g.

    [User Preference Profile]:
    - Genre/theme preference: science fiction
    - Liked aspects: complex storyline
    - Rating tendency: mostly highly positive (average 4.5/5)
    - Writing style: brief (about 9 words per review)

Design choices (see SOP, Sec. III-C):
  * deterministic, CPU-only, no extra LLM and no paid API;
  * uses only the reviews it is given (plus metadata of the items those
    reviews are about), so it cannot leak information from the rest of the
    user's history;
  * fast enough to run on the whole test set in a few seconds.

The lexicons below are intentionally small and easy to inspect/extend.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ----------------------------------------------------------------------------
# tiny text utilities
# ----------------------------------------------------------------------------
def _stem(tok: str) -> str:
    """Very light plural stripping, applied to text AND lexicons alike."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokenize(sentence: str) -> List[str]:
    toks = re.findall(r"[a-z]+(?:'[a-z]+)?", sentence.lower())
    return [t.replace("'", "") for t in toks]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p for p in (s.strip() for s in parts) if p]


# ----------------------------------------------------------------------------
# lexicons
# ----------------------------------------------------------------------------
POS_WORDS = {
    "love", "loved", "great", "excellent", "amazing", "wonderful", "fantastic",
    "awesome", "perfect", "best", "brilliant", "beautiful", "enjoy", "enjoyed",
    "favorite", "superb", "outstanding", "gripping", "compelling", "captivating",
    "engaging", "entertaining", "masterpiece", "recommend", "recommended",
    "impressive", "delightful", "fun", "good", "solid", "powerful", "moving",
    "touching", "heartwarming", "witty", "clever", "hilarious", "stunning",
    "fascinating", "worth", "satisfying", "terrific", "incredible",
    "magnificent", "pleasant", "charming", "riveting", "strong", "rich",
    "soulful", "catchy", "melodic", "polished", "crisp", "vivid", "thrilling",
}
NEG_WORDS = {
    "hate", "hated", "bad", "boring", "dull", "terrible", "awful", "worst",
    "poor", "disappointing", "disappointed", "waste", "weak", "predictable",
    "slow", "tedious", "confusing", "annoying", "mediocre", "bland", "flat",
    "lame", "forgettable", "overrated", "cheesy", "repetitive", "shallow",
    "unrealistic", "rushed", "dragged", "messy", "horrible", "disappoint",
    "fail", "failed", "lacking", "lack", "cliche", "cliched", "unoriginal",
    "clumsy", "sloppy",
}
NEGATORS = {
    "not", "no", "never", "hardly", "barely", "without", "cannot", "cant",
    "isnt", "wasnt", "dont", "didnt", "doesnt", "wont", "couldnt", "wouldnt",
    "nor", "arent", "werent",
}
# adjectives we are willing to attach to an aspect ("complex storyline")
DESCRIPTORS = {
    "complex", "gripping", "compelling", "captivating", "engaging", "original",
    "creative", "beautiful", "powerful", "emotional", "witty", "clever", "dark",
    "realistic", "detailed", "deep", "thoughtful", "fast", "slow",
    "predictable", "boring", "shallow", "repetitive", "tight", "smooth", "rich",
    "fresh", "fun", "funny", "intense", "moving", "touching", "soulful",
    "catchy", "melodic", "raw", "polished", "crisp", "clear", "poetic", "vivid",
    "charming", "hilarious", "suspenseful", "thrilling", "strong", "weak",
}

# aspect label -> trigger words
ASPECTS: Dict[str, Sequence[str]] = {
    "storyline/plot": ["story", "storyline", "plot", "narrative", "twist",
                       "ending", "premise", "tale"],
    "characters": ["character", "protagonist", "hero", "heroine", "villain",
                   "relationship"],
    "writing style": ["writing", "prose", "writer", "author", "style",
                      "written", "dialogue", "narration"],
    "pacing": ["pace", "pacing", "slow", "fast", "dragged", "rushed"],
    "acting": ["acting", "actor", "actress", "performance", "cast", "role"],
    "direction/visuals": ["director", "direction", "cinematography", "visual",
                          "effect", "scenery", "animation", "shot"],
    "music/sound": ["music", "soundtrack", "sound", "melody", "song", "track",
                    "album", "guitar", "beat", "instrumental", "lyric",
                    "vocal", "voice", "singing", "singer"],
    "production quality": ["production", "quality", "mixing", "recording",
                           "remaster", "transfer", "picture", "audio"],
    "emotional impact": ["emotional", "moving", "touching", "heartfelt",
                         "emotion", "tear", "powerful"],
    "humor": ["funny", "humor", "hilarious", "laugh", "witty"],
    "suspense/tension": ["suspense", "tension", "thrilling", "gripping",
                         "suspenseful"],
    # note: "complex"/"detailed" are *descriptors* (they modify another aspect),
    # not triggers, otherwise "complex storyline" would also fire this aspect.
    "depth/complexity": ["depth", "complexity", "insightful", "informative"],
    "value": ["price", "value", "cheap", "bargain"],
}
_ASPECT_INDEX: Dict[str, str] = {}
for _label, _triggers in ASPECTS.items():
    for _t in _triggers:
        _ASPECT_INDEX.setdefault(_stem(_t), _label)

_POS = {_stem(w) for w in POS_WORDS}
_NEG = {_stem(w) for w in NEG_WORDS}
_DESC = {_stem(w) for w in DESCRIPTORS}

# genre / theme patterns per Amazon category used by DEP
_G = re.compile
GENRES: Dict[str, Dict[str, "re.Pattern"]] = {
    "Books": {
        "science fiction": _G(r"sci[- ]?fi|science[- ]fiction|space opera|alien|dystopi|cyberpunk"),
        "fantasy": _G(r"fantasy|wizard|dragon|\belf\b|\belves\b|sword"),
        "mystery/crime": _G(r"mystery|detective|whodunit|murder|crime|thriller|\bnoir\b"),
        "romance": _G(r"romance|romantic|love story"),
        "horror": _G(r"horror|scary|zombie|vampire|ghost"),
        "historical": _G(r"histor|world war|\bwwii\b|civil war|medieval"),
        "biography/memoir": _G(r"biograph|memoir|autobiograph"),
        "self-help/non-fiction": _G(r"self[- ]help|non[- ]?fiction|how[- ]to|\bguide\b|business"),
        "children/young adult": _G(r"children|\bkids?\b|young adult|\bya\b|teen"),
    },
    "Movies_and_TV": {
        "science fiction": _G(r"sci[- ]?fi|science[- ]fiction|space|alien|dystopi|cyberpunk"),
        "fantasy": _G(r"fantasy|wizard|dragon|magic"),
        "mystery/crime": _G(r"mystery|detective|murder|crime|thriller|\bnoir\b"),
        "romance": _G(r"romance|romantic|love story"),
        "horror": _G(r"horror|scary|zombie|vampire|ghost|slasher"),
        "historical/war": _G(r"histor|world war|\bwwii\b|civil war|\bwar film|war movie"),
        "comedy": _G(r"comedy|sitcom|hilarious|funny"),
        "drama": _G(r"\bdrama"),
        "action/adventure": _G(r"action|adventure|martial arts|battle"),
        "documentary": _G(r"documentar"),
        "animation/family": _G(r"anime|animat|cartoon|family|children|\bkids?\b"),
    },
    "CDs_and_Vinyl": {
        "rock": _G(r"\brock\b|punk|grunge|alternative"),
        "pop": _G(r"\bpop\b"),
        "hip-hop/rap": _G(r"hip[- ]?hop|\brap\b"),
        "jazz/blues": _G(r"jazz|blues"),
        "classical": _G(r"classical|symphon|orchestra|baroque|\bopera\b"),
        "country/folk": _G(r"country|folk|bluegrass"),
        "metal": _G(r"\bmetal\b"),
        "r&b/soul/funk": _G(r"r&b|\bsoul\b|funk"),
        "electronic": _G(r"electronic|techno|\bhouse music|\bedm\b|ambient"),
        "gospel/christian": _G(r"gospel|christian|worship"),
        "soundtrack": _G(r"soundtrack|\bost\b|film score"),
    },
}


# ----------------------------------------------------------------------------
# output container
# ----------------------------------------------------------------------------
@dataclass
class PreferenceProfile:
    n_reviews: int
    avg_rating: Optional[float]
    rating_tendency: str
    genres: List[str] = field(default_factory=list)
    liked: List[str] = field(default_factory=list)
    disliked: List[str] = field(default_factory=list)
    style: str = ""

    def to_text(self) -> str:
        lines = []
        if self.genres:
            lines.append(f"- Genre/theme preference: {', '.join(self.genres)}")
        if self.liked:
            lines.append(f"- Liked aspects: {', '.join(self.liked)}")
        if self.disliked:
            lines.append(f"- Disliked aspects: {', '.join(self.disliked)}")
        if self.avg_rating is not None:
            lines.append(
                f"- Rating tendency: {self.rating_tendency} "
                f"(average {self.avg_rating:.1f}/5)")
        if self.style:
            lines.append(f"- Writing style: {self.style}")
        return "[User Preference Profile]:\n" + "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ----------------------------------------------------------------------------
# the enricher
# ----------------------------------------------------------------------------
class PreferenceEnricher:
    """Extract a compact preference profile from a handful of reviews.

    Parameters
    ----------
    category : one of "Books", "Movies_and_TV", "CDs_and_Vinyl".
    max_items : max number of liked / disliked aspects reported.
    desc_chars : how much of each item description is used for genre cues.
    """

    def __init__(self, category: str, max_items: int = 4,
                 max_genres: int = 2, desc_chars: int = 400):
        if category not in GENRES:
            raise ValueError(f"unknown category {category!r}")
        self.category = category
        self.max_items = max_items
        self.max_genres = max_genres
        self.desc_chars = desc_chars

    # -- polarity ---------------------------------------------------------
    @staticmethod
    def _sentence_polarity(tokens: List[str]) -> int:
        score = 0
        for i, tok in enumerate(tokens):
            s = _stem(tok)
            val = 1 if s in _POS else (-1 if s in _NEG else 0)
            if val and any(t in NEGATORS for t in tokens[max(0, i - 3):i]):
                val = -val
            score += val
        return (score > 0) - (score < 0)

    @staticmethod
    def _rating_prior(rating) -> int:
        try:
            r = float(rating)
        except (TypeError, ValueError):
            return 0
        return 1 if r >= 4 else (-1 if r <= 2 else 0)

    # -- main entry -------------------------------------------------------
    def enrich(self, reviews: Sequence[dict]) -> PreferenceProfile:
        """reviews: dicts with keys title, text, rating, item_title, item_desc."""
        pos_cnt: Counter = Counter()
        neg_cnt: Counter = Counter()
        phrases: Dict[str, Counter] = defaultdict(Counter)
        genre_cnt: Counter = Counter()
        ratings, word_counts, exclaim, recommends = [], [], 0, 0

        for rev in reviews:
            prior = self._rating_prior(rev.get("rating"))
            try:
                ratings.append(float(rev["rating"]))
            except (KeyError, TypeError, ValueError):
                pass
            title = rev.get("title") or ""
            text = rev.get("text") or ""
            word_counts.append(len(text.split()))
            exclaim += int("!" in text or "!" in title)
            recommends += int("recommend" in text.lower())

            # sentence level aspect/opinion mining
            for sent in _split_sentences(title) + _split_sentences(text):
                toks = _tokenize(sent)
                stems = [_stem(t) for t in toks]
                pol = self._sentence_polarity(toks) or prior
                if pol == 0:
                    continue
                for j, st in enumerate(stems):
                    label = _ASPECT_INDEX.get(st)
                    if label is None:
                        continue
                    (pos_cnt if pol > 0 else neg_cnt)[label] += 1
                    window = range(max(0, j - 3), min(len(toks), j + 4))
                    for k in window:
                        if k != j and stems[k] in _DESC and stems[k] != st:
                            phrases[label][f"{toks[k]} {toks[j]}"] += 1
                            break

            # genre cues: item metadata + the review itself
            blob = " ".join([
                str(rev.get("item_title") or ""),
                str(rev.get("item_desc") or "")[: self.desc_chars],
                title, text,
            ]).lower()
            for g, pat in GENRES[self.category].items():
                genre_cnt[g] += len(pat.findall(blob))

        def best(label: str) -> str:
            return phrases[label].most_common(1)[0][0] if phrases[label] else label

        liked = [a for a, n in (pos_cnt - neg_cnt).most_common(self.max_items)]
        disliked = [a for a, n in (neg_cnt - pos_cnt).most_common(self.max_items)]
        genres = [g for g, n in genre_cnt.most_common(self.max_genres) if n > 0]

        avg = sum(ratings) / len(ratings) if ratings else None
        if avg is None:
            tendency = ""
        elif avg >= 4.5:
            tendency = "mostly highly positive"
        elif avg >= 3.5:
            tendency = "generally positive"
        elif avg >= 2.5:
            tendency = "mixed / moderate"
        else:
            tendency = "critical"

        n = max(len(reviews), 1)
        avg_words = sum(word_counts) / n if word_counts else 0
        size = ("brief" if avg_words < 30
                else "moderately detailed" if avg_words < 100 else "detailed")
        style = f"{size} (about {int(round(avg_words))} words per review)"
        if exclaim * 2 >= n:
            style += ", enthusiastic tone"
        if recommends:
            style += ", tends to recommend"

        return PreferenceProfile(
            n_reviews=len(reviews),
            avg_rating=avg,
            rating_tendency=tendency,
            genres=genres,
            liked=[best(a) for a in liked],
            disliked=[best(a) for a in disliked],
            style=style,
        )


if __name__ == "__main__":  # tiny demo from the SOP
    demo = [
        dict(title="Great read", text="I loved the science-fiction story.",
             rating=5, item_title="Space Saga", item_desc=""),
        dict(title="", text="The complex storyline was excellent.",
             rating=4, item_title="Orbit", item_desc=""),
    ]
    print(PreferenceEnricher("Books").enrich(demo).to_text())
