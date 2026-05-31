"""Normalise extracted entities — filter noise, clean institution names."""
import json
import re
from pathlib import Path


INPUT_FILE = Path("data/processed/entities.jsonl")
OUTPUT_FILE = Path("data/processed/entities_clean.jsonl")
LOG_FILE = Path("data/processed/normalisation_log.jsonl")

INSTITUTION_ALIASES: dict[str, str] = {
    "google brain": "Google Brain",
    "google deepmind": "Google DeepMind",
    "deepmind": "DeepMind",
    "openai": "OpenAI",
    "microsoft research": "Microsoft Research",
    "microsoft": "Microsoft",
    "meta ai": "Meta AI",
    "meta": "Meta",
    "facebook ai": "Meta AI",
    "fair": "Meta AI",
    "amazon": "Amazon",
    "apple": "Apple",
    "nvidia": "NVIDIA",
    "mit": "MIT",
    "massachusetts institute of technology": "MIT",
    "stanford": "Stanford University",
    "stanford university": "Stanford University",
    "cmu": "Carnegie Mellon University",
    "carnegie mellon": "Carnegie Mellon University",
    "carnegie mellon university": "Carnegie Mellon University",
    "uc berkeley": "UC Berkeley",
    "university of california berkeley": "UC Berkeley",
    "berkeley": "UC Berkeley",
    "oxford": "University of Oxford",
    "university of oxford": "University of Oxford",
    "cambridge": "University of Cambridge",
    "university of cambridge": "University of Cambridge",
    "eth zurich": "ETH Zurich",
    "eth zürich": "ETH Zurich",
    "tsinghua": "Tsinghua University",
    "tsinghua university": "Tsinghua University",
    "peking university": "Peking University",
    "pku": "Peking University",
    "chinese academy of sciences": "Chinese Academy of Sciences",
    "cas": "Chinese Academy of Sciences",
    "huawei": "Huawei",
    "alibaba": "Alibaba",
    "baidu": "Baidu",
    "ibm": "IBM",
    "ibm research": "IBM Research",
    "allen institute": "Allen Institute for AI",
    "ai2": "Allen Institute for AI",
}


def is_noise(text: str) -> bool:
    """Return True if text is LaTeX, a number, or otherwise not a real entity."""
    if not text or len(text) < 3 or len(text) > 80:
        return True
    # LaTeX markers
    if any(c in text for c in ["\\", "$", "^", "{", "}", "_", "\\"]):
        return True
    # Starts with number or symbol
    if re.match(r"^[\d\+\-\.\%]", text):
        return True
    # Mostly non-alphabetic
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.5:
        return True
    # URL
    if "http" in text or "www." in text:
        return True
    return False


def normalise_org(text: str) -> str:
    key = text.lower().strip().rstrip(".,;:")
    if key in INSTITUTION_ALIASES:
        canonical = INSTITUTION_ALIASES[key]
        return canonical
    # Title-case as fallback
    return text.strip().rstrip(".,;:")


def normalise_person(text: str) -> str:
    return text.strip().rstrip(".,;:")


def normalise_entities() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results = []
    logs = []

    with open(INPUT_FILE, encoding="utf-8") as f:
        raw_entries = [json.loads(line) for line in f]

    for entry in raw_entries:
        clean_orgs = []
        for org in entry["orgs"]:
            if is_noise(org):
                logs.append({"arxiv_id": entry["arxiv_id"], "type": "org_filtered", "original": org})
                continue
            normalised = normalise_org(org)
            if normalised != org:
                logs.append({"arxiv_id": entry["arxiv_id"], "type": "org_normalised", "original": org, "canonical": normalised})
            clean_orgs.append(normalised)

        clean_persons = []
        for person in entry["persons"]:
            if is_noise(person):
                logs.append({"arxiv_id": entry["arxiv_id"], "type": "person_filtered", "original": person})
                continue
            clean_persons.append(normalise_person(person))

        results.append({
            "arxiv_id": entry["arxiv_id"],
            "persons": list(dict.fromkeys(clean_persons)),
            "orgs": list(dict.fromkeys(clean_orgs)),
            "methods": entry["methods"],
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"Done. {len(results)} papers cleaned.")
    print(f"Normalisation log: {len(logs)} decisions → {LOG_FILE}")
    print(f"Sample: {results[0]}")


if __name__ == "__main__":
    normalise_entities()
