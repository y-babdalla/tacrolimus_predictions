-- Save pgp_inducers
CREATE TABLE drugs_pgp_inducers AS
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
        LOWER(drug) LIKE '%apalutamide%' OR
        LOWER(drug) LIKE '%carbamazepine%' OR
        LOWER(drug) LIKE '%fosphenytoin%' OR
        LOWER(drug) LIKE '%green tea%' OR
		LOWER(drug) LIKE '%camellia%' OR
        LOWER(drug) LIKE '%lorlatinib%' OR
        LOWER(drug) LIKE '%phenytoin%' OR
        LOWER(drug) LIKE '%rifampin%' OR
		LOWER(drug) LIKE '%rifampicin%' OR
        LOWER(drug) LIKE '%john%' OR
        LOWER(drug) LIKE '%carbamazepine%' OR
        LOWER(drug) LIKE '%phenytoin%'
    );

