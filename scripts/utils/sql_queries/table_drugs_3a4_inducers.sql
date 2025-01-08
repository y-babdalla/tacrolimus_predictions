-- Save 3a4_inducers
CREATE TABLE drugs_3a4_inducers AS
SELECT DISTINCT
	drug,
	formulary_drug_cd,
	gsn,
	ndc,
	-- prod_strength,
	form_unit_disp,
	route
FROM mimiciv_hosp.prescriptions
WHERE subject_id IN (SELECT subject_id FROM tac_patients_presc)
	AND (
        LOWER(drug) LIKE '%bexarotene%' OR
        LOWER(drug) LIKE '%bosentan%' OR
        LOWER(drug) LIKE '%cenobamate%' OR
        LOWER(drug) LIKE '%dabrafenib%' OR
        LOWER(drug) LIKE '%dexamethasone%' OR
        LOWER(drug) LIKE '%dipyrone%' OR
        LOWER(drug) LIKE '%efavirenz%' OR
        LOWER(drug) LIKE '%elagolix%' AND LOWER(drug) LIKE '%estradiol%' AND LOWER(drug) LIKE '%norethindrone%' OR
        LOWER(drug) LIKE '%eslicarbazepine%' OR
        LOWER(drug) LIKE '%etravirine%' OR
        LOWER(drug) LIKE '%lorlatinib%' OR
        LOWER(drug) LIKE '%mitapivat%' OR
        LOWER(drug) LIKE '%modafinil%' OR
        LOWER(drug) LIKE '%nafcillin%' OR
        LOWER(drug) LIKE '%pexidartinib%' OR
        LOWER(drug) LIKE '%repotrectinib%' OR
        LOWER(drug) LIKE '%rifabutin%' OR
        LOWER(drug) LIKE '%rifapentine%' OR
        LOWER(drug) LIKE '%sotorasib%' OR
        LOWER(drug) LIKE '%john%' OR
        LOWER(drug) LIKE '%apalutamide%' OR
        LOWER(drug) LIKE '%carbamazepine%' OR
        LOWER(drug) LIKE '%encorafenib%' OR
        LOWER(drug) LIKE '%enzalutamide%' OR
        LOWER(drug) LIKE '%fosphenytoin%' OR
        LOWER(drug) LIKE '%lumacaftor%' OR
        LOWER(drug) LIKE '%mitotane%' OR
        LOWER(drug) LIKE '%phenobarbital%' OR
        LOWER(drug) LIKE '%phenytoin%' OR
        LOWER(drug) LIKE '%primidone%' OR
        LOWER(drug) LIKE '%rifampin%' OR
		LOWER(drug) LIKE '%rifampicin%' OR
        LOWER(drug) LIKE '%bosentan%' OR
		LOWER(drug) LIKE '%tracleer%' OR
        LOWER(drug) LIKE '%carbamazepine%' OR
        LOWER(drug) LIKE '%dexamethasone%' OR
        LOWER(drug) LIKE '%dexamethasone sod phosphate%' OR
        LOWER(drug) LIKE '%phenobarbital%' OR
        LOWER(drug) LIKE '%phenytoin%'
    );

SELECT *
FROM drugs_3a4_inducers