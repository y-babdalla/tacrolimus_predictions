-- Create drugs table with pgp & 3a4 inhibitors/inducers
-- If prescription start/stoptime is null then fill in with admission datetime range
CREATE TABLE drugs_pgp_3a4 AS
SELECT 
    p.subject_id,
    p.hadm_id,
	p.pharmacy_id,
    p.drug,
	p.formulary_drug_cd,
	p.gsn,
	p.ndc,
	p.prod_strength,
    p.route,
    p.dose_val_rx,
	p.dose_unit_rx,
	p.form_val_disp,
	p.form_unit_disp,
	p.doses_per_24_hrs,
    -- Add all other columns from p.* explicitly, except starttime and stoptime
    COALESCE(p.starttime, hadm.starttime) AS starttime,
    COALESCE(p.stoptime, hadm.stoptime) AS stoptime,
    CASE
        WHEN p.drug IN (SELECT drug FROM drugs_pgp_inhibitors) THEN 0
        WHEN p.drug IN (SELECT drug FROM drugs_pgp_inducers) THEN 1
        ELSE NULL
    END AS pgp,
    CASE
        WHEN p.drug IN (SELECT drug FROM drugs_3a4_inhibitors) THEN 0
        WHEN p.drug IN (SELECT drug FROM drugs_3a4_inducers) THEN 1
        ELSE NULL
    END AS cyp3a4
FROM mimiciv_hosp.prescriptions p
LEFT JOIN (
    SELECT hadm_id, admittime AS starttime, dischtime AS stoptime
    FROM mimiciv_hosp.admissions
) hadm ON p.hadm_id = hadm.hadm_id
WHERE p.subject_id IN (SELECT subject_id FROM tac_patients_presc)
    AND p.drug IN (
        SELECT drug 
        FROM drugs_pgp_inhibitors
        UNION
        SELECT drug 
        FROM drugs_pgp_inducers
        UNION
        SELECT drug 
        FROM drugs_3a4_inhibitors
        UNION
        SELECT drug 
        FROM drugs_3a4_inducers
    )
    AND LOWER(p.route) NOT LIKE '%eye%'
    AND LOWER(p.route) NOT LIKE '%ear%'
    AND p.route NOT IN ('TP','PB', 'OU', 'OS', 'OD')
    AND p.dose_val_rx != '0'
;



-- CREATE TABLE drugs_pgp_3a4 AS
-- SELECT 
--     p.*,
--     CASE
--         WHEN p.drug IN (SELECT drug FROM drugs_pgp_inhibitors) THEN 0
--         WHEN p.drug IN (SELECT drug FROM drugs_pgp_inducers) THEN 1
--         ELSE NULL
--     END AS pgp,
--     CASE
--         WHEN p.drug IN (SELECT drug FROM drugs_3a4_inhibitors) THEN 0
--         WHEN p.drug IN (SELECT drug FROM drugs_3a4_inducers) THEN 1
--         ELSE NULL
--     END AS cyp3a4
-- FROM mimiciv_hosp.prescriptions p
-- WHERE p.subject_id IN (SELECT subject_id FROM tac_patients_presc)
--     AND p.drug IN (
--         SELECT drug 
--         FROM drugs_pgp_inhibitors
--         UNION
--         SELECT drug 
--         FROM drugs_pgp_inducers
--         UNION
--         SELECT drug 
--         FROM drugs_3a4_inhibitors
--         UNION
--         SELECT drug 
--         FROM drugs_3a4_inducers
--     )
-- 	AND LOWER(p.route) NOT LIKE '%eye%'
-- 	AND LOWER(p.route) NOT LIKE '%ear%'
-- 	AND p.route NOT IN ('TP','PB', 'OU', 'OS', 'OD')
-- 	AND p.dose_val_rx != '0'
-- 	;
