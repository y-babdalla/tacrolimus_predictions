NUMERIC_COLS = [
    "subject_id",
    "age",
    "weight",
    "height",
    "ast",
    "alt",
    "alp",
    "bilirubin_total",
    "albumin",
    "bun",
    "creatinine",
    "sodium",
    "potassium",
    "inr",
    "hemoglobin",
    "hematocrit",
    "pgp_inhibit",
    "pgp_induce",
    "cyp3a4_inhibit",
    "cyp3a4_induce",
    "tac_level",
    "dose",
    "doses_per_24_hrs",
    "level_dose_timediff",
]
CATEGORICAL_COLS = ["gender", "race", "state", "formulation", "route"]
TIME_COLS = ["level_time", "dose_time"]
TIME_ORDERING_COL = "level_time"
TARGET = "tac_level"
