-- Add TAC dosage to followup table

-- Step 1: Create table from preprocessed TAC dose table 
-- Data types following MIMIC guidelines: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/buildmimic/postgres/create.sql
-- CREATE TABLE drugs_tac_proc (
-- 	subject_id INTEGER NOT NULL,
-- 	pharmacy_id INTEGER NOT NULL,
-- 	charttime TIMESTAMP NOT NULL,
-- 	hadm_id INTEGER NOT NULL,
-- 	starttime TIMESTAMP NOT NULL,
-- 	stoptime TIMESTAMP NOT NULL,
-- 	formulation VARCHAR(15),
-- 	route VARCHAR(10),
-- 	dose DOUBLE PRECISION,
-- 	doses_per_24_hrs REAL
-- );

-- Step 2: Import CSV data into table manually with pgadmin

-- Step 3: Merge TAC dose to followup table
DROP TABLE IF EXISTS followup_tacdose;

CREATE TABLE followup_tacdose AS

WITH stg0 AS(
	SELECT 
	    fd.*,
	    dt.charttime AS dose_charttime,
		dt.pharmacy_id,
		dt.formulation,
		dt.route,
		dt.dose, 
		dt.doses_per_24_hrs,
		EXTRACT(EPOCH FROM (fd.charttime - dt.charttime)) / 86400 AS dose_diff_days,
		EXTRACT(EPOCH FROM (fd.charttime - fd.first_tac_dose)) / 86400 AS treatment_diff_days
	FROM 
	    followup_drugs fd
	LEFT JOIN LATERAL
		(SELECT
			charttime,
			pharmacy_id,
			formulation,
			route,
			dose, 
			doses_per_24_hrs
	     FROM 
	         drugs_tac_proc dt
	     WHERE 
	         dt.subject_id = fd.subject_id
	         AND dt.charttime <= fd.charttime
	     ORDER BY 
	         fd.charttime - dt.charttime ASC
	     LIMIT 1
	    ) dt ON true -- ensure all records of followup_drugs is included even if no correponding dose found
	)

SELECT *
FROM stg0
WHERE dose_diff_days<=365
	AND dose IS NOT NULL