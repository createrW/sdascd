"""Shared label definitions. All three groups must import from here."""

# README 3.3: 4-class `sarcasm` -> binary `sarcasm_binary`
SARCASM_TO_BINARY = {
    "not_sarcastic": 0,
    "general": 1,
    "twisted_meaning": 1,
    "very_twisted": 1,
}

LABEL_NAMES = {0: "not_sarcastic", 1: "sarcastic"}

# README 3.2 / 3.3 / 5.2: expected counts, used by scripts/verify_dataset.py
EXPECTED_SPLIT_SIZES = {"train": 5593, "validation": 699, "test": 700}
EXPECTED_TRAIN_LABEL_COUNTS = {0: 1226, 1: 4367}
EXPECTED_TOTAL_LABEL_COUNTS = {0: 1544, 1: 5448}
