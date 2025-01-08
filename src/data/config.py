"""Configurations for data processing."""

from typing import Final

COLS_MAP: Final[dict[str, dict[str, int | str]]] = {
    "gender": {"M": 0, "F": 1},
    "route": {"SL": 1, "ORAL": 0},
    "state": {"home": 0, "hosp": 1, "icu": 2},
    "race": {
        "White": "white",
        "Black/African American": "black",
        "Hispanic/Latino": "latin",
        "Asian": "asian",
        "Other": "other",
        "Native American": "other",
        "Multiple": "other",
        "Pacific Islander": "other",
    },
}
DROP_COLS: Final[list[str]] = [
    "pharmacy_id",
    "previous_route",
    "previous_formulation",
    "previous_dose_timediff",
    "alp",
    "inr",
    "bun",
]


LAB_COLS: Final[list[str]] = [
    "ast",
    "alt",
    "bilirubin",
    "albumin",
    "creatinine",
    "sodium",
    "potassium",
    "hemoglobin",
    "hematocrit",
]
LAB_SKEW_COLS: Final[list[str]] = ["ast", "alt", "bilirubin", "creatinine"]
LAB_NO_SKEW_COLS: Final[list[str]] = [lab for lab in LAB_COLS if lab not in LAB_SKEW_COLS]
GROUP_COLS: Final[list[str]] = ["gender", "age_group"]
TARGET_COLS: Final[list[str]] = ["weight", "height"]
KEEP_COLS: Final[list[str]] = ["gender"]
