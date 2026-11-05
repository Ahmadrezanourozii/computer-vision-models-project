"""Balanced stimulus design for the 100 portraits.

Factors (crossed with offsets so that no two factors are confounded):
  light  : left / right / top / bottom              (25 each)  -> main experimental factor
  gender : female / male                             (50 / 50)
  age    : young adult / middle-aged / elderly       (~33 each)
  people : 6 ethnic backgrounds                      (~17 each, only for diversity)
  pose   : frontal / three-quarter                   (~50 / 50)
  extras : glasses / beard / hair styles             (sprinkled for variety)
Each design cell gets K candidate seeds; quality control (qc_select.py) keeps the best one.
"""
import csv
import itertools
import random

LIGHTS = ["left", "right", "top", "bottom"]
GENDERS = ["female", "male"]
AGES = ["young adult", "middle-aged", "elderly"]
PEOPLE = ["East Asian", "South Asian", "Black", "white", "Middle Eastern", "Latino"]
POSES = ["frontal", "three-quarter"]
HAIR = {"female": ["long straight hair", "short bob haircut", "hair tied back in a bun", "curly shoulder-length hair",
                   "wavy hair", "pixie cut"],
        "male": ["short hair", "buzz cut", "slicked-back hair", "curly short hair", "receding hairline", "wavy hair"]}
EXTRAS = {"female": ["", "", "", "wearing thin-rimmed glasses", "", "small earrings"],
          "male": ["", "with a full beard", "", "wearing thin-rimmed glasses", "with a short stubble beard", "with a mustache"]}


def build(n_per_light=25, k=2, seed=7):
    rng = random.Random(seed)
    rows, idx = [], 0
    for li, light in enumerate(LIGHTS):
        for j in range(n_per_light):
            g = GENDERS[(j + li) % 2]
            age = AGES[(j + 2 * li) % 3]
            ppl = PEOPLE[(j + 3 * li) % 6]
            pose = POSES[(j // 2 + li) % 2]
            hair = HAIR[g][rng.randrange(6)]
            extra = EXTRAS[g][rng.randrange(6)]
            if age == "elderly":
                hair = {"female": "grey hair", "male": "grey hair"}[g] if rng.random() < 0.7 else hair
            idx += 1
            for c in range(k):
                rows.append(dict(cell=f"c{idx:03d}", cand=c, seed=10000 + 100 * idx + c, light=light, gender=g, age=age,
                                 people=ppl, pose=pose, hair=hair, extra=extra))
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("-o", "--out", default="design.csv")
    a = ap.parse_args()
    rows = build(k=a.k)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", a.out)
    from collections import Counter
    first = [r for r in rows if r["cand"] == 0]
    for f in ("light", "gender", "age", "people", "pose"):
        print(f, dict(Counter(r[f] for r in first)))
    print("light x gender", dict(Counter((r["light"], r["gender"]) for r in first)))
    print("light x age", dict(Counter((r["light"], r["age"]) for r in first)))
