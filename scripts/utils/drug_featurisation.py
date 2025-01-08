"""Example usage of drug featurisation for patients.

This is implemented for separate drugs to Pgp inhibitors and inducers, CPY3A4 inhibitors
and inducers.

Sources:
3a4: https://www.uptodate.com/contents/image?imageKey=DRUG/76992
pgp: https://www.uptodate.com/contents/image?imageKey=DRUG/7332
"""
import logging

import pandas as pd


def featurise_drug(
    drug_info: pd.DataFrame, drugs: list[str], print_no_category: bool = False
) -> dict:
    """Evaluates a list of drugs and counts their occurrences as Pgp and CYP3A4.

    Args:
        drug_info (pd.DataFrame): DataFrame with drug information.
        drugs (list[str]): List of drugs to evaluate.
        print_no_category (bool): Whether to print drugs that don't fit any category.

    Returns:
        dict: Dictionary with the counts for each category.
    """
    scores = {"cyp_inducers": 0, "cyp_inhibitors": 0, "pgp_inducers": 0, "pgp_inhibitors": 0}
    no_category = []

    cyp_ind = drug_info["3a4_inducers"].dropna().tolist()
    cyp_inh = drug_info["3a4_inhibitors"].dropna().tolist()
    pgp_ind = drug_info["pgp_inducers"].dropna().tolist()
    pgp_inh = drug_info["pgp_inhibitors"].dropna().tolist()

    for drug in drugs:
        drug_lower = drug.lower()
        found = False

        if any(drug_lower in cyp.lower() for cyp in cyp_ind):
            scores["cyp_inducers"] += 1
            found = True
        if any(drug_lower in cyp.lower() for cyp in cyp_inh):
            scores["cyp_inhibitors"] += 1
            found = True
        if any(drug_lower in pgp.lower() for pgp in pgp_ind):
            scores["pgp_inducers"] += 1
            found = True
        if any(drug_lower in pgp.lower() for pgp in pgp_inh):
            scores["pgp_inhibitors"] += 1
            found = True

        if not found:
            no_category.append(drug)

    if print_no_category:
        logging.info(f"No category found: {no_category}")

    return scores


if __name__ == "__main__":
    example_patient = [
        "Bexarotene",
        "Efavirenz",
        "Modafinil",
        "Rifabutin",
        "Apalutamide",
        "Atazanavir",
        "Clarithromycin",
        "Ketoconazole",
        "Ritonavir",
        "Verapamil",
        "Amiodarone",
        "Cyclosporine",
        "Erythromycin",
        "Itraconazole",
        "Vemurafenib",
        "Apalutamide",
        "Green tea",
        "Phenytoin",
        "St. John's wort",
        "Ibuprofen",
        "Paracetamol",
        "Cetirizine",
        "Metformin",
        "Lisinopril",
    ]

    interactions = pd.read_csv("utils/drug_interactions.csv")

    results = featurise_drug(interactions, example_patient, print_no_category=True)
    logging.info(results)
