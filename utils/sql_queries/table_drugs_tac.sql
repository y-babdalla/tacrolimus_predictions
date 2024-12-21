-- Create table with TAC dosage
DROP TABLE IF EXISTS drugs_tac;

CREATE TABLE drugs_tac AS
-- Extract TAC prescription data 
WITH tac_presc AS (
	SELECT
		p.subject_id,
	    p.hadm_id,
		p.pharmacy_id,
		p.starttime,
		p.stoptime,
		CASE
	        WHEN CAST(p.ndc AS BIGINT) BETWEEN 469060773 AND 469301601 THEN 'prograf'
	        WHEN CAST(p.ndc AS BIGINT) BETWEEN 781210201 AND 781210401 THEN 'adoport'
	        WHEN CAST(p.ndc AS BIGINT) BETWEEN 16729004101 AND 16729004301 THEN 'accord'
	        WHEN CAST(p.ndc AS BIGINT) = 51079081820 THEN 'mylan'
	        WHEN CAST(p.ndc AS BIGINT) BETWEEN 55111052501 AND 55111052701 THEN 'dr_reddy'
	        WHEN CAST(p.ndc AS BIGINT) BETWEEN 68084044901 AND 68084045001 THEN 'american_health'
	        WHEN CAST(p.ndc AS BIGINT) BETWEEN 68992301003 AND 68992307503 OR CAST(p.ndc AS INTEGER) = 0 THEN 'envarsus'
	        ELSE NULL
	    END AS formulation,
		CASE
	        WHEN p.route = 'SL' THEN 'SL'
	        ELSE 'ORAL'
	    END AS route,
	    p.dose_val_rx AS dose,
		p.doses_per_24_hrs
	FROM mimiciv_hosp.prescriptions p
	WHERE LOWER(DRUG) LIKE '%tacrolimus%'
		AND route NOT IN ('BU','TP','IV DRIP')
		AND subject_id IN (SELECT subject_id FROM tac_patients_presc)
		AND dose_unit_rx = 'mg'
		AND ndc IS NOT NULL
		AND CAST(dose_val_rx AS FLOAT) > 0
		AND dose_val_rx IS NOT NULL
		AND starttime<=stoptime
	),

-- Extract pharmacy data from previous TAC prescription
tac_pharma AS(
	SELECT ph.subject_id,
		ph.hadm_id,
		ph.pharmacy_id,
		ph.doses_per_24_hrs,
		ph.duration,
		ph.duration_interval,
		ph.starttime, 
		ph.stoptime,
		ph.medication,
		ph.proc_type,
		ph.status, 
		ph.route,
		ph.frequency,
		ph.disp_sched
	FROM mimiciv_hosp.pharmacy ph
	WHERE pharmacy_id IN (SELECT pharmacy_id FROM tac_presc)
	)

-- Merge
SELECT tp.subject_id,
       tp.hadm_id,
       tp.pharmacy_id,
       tp.starttime,
       tp.stoptime,
       tp.formulation,
       tp.route,
       tp.dose,
		tp.doses_per_24_hrs,
       tph.frequency,
       tph.disp_sched
FROM tac_presc tp
LEFT JOIN tac_pharma tph
ON tp.pharmacy_id = tph.pharmacy_id
WHERE tph.proc_type != 'Discharge Med' 
	AND tp.starttime IS NOT NULL
