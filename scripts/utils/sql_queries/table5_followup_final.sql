-- Final cleaning to generate the main followup table
DROP TABLE IF EXISTS followup_final;

CREATE TABLE followup_final AS
SELECT 
	subject_id,
	pharmacy_id,
	-- Demographic & patient specific features
	age,
	gender,
	race,
	weight,
	height,
	state,
	-- Lab features
	ast,
	alt,
	alp,
	bilirubin, 
	albumin,
	bun,
	creatinine,
	sodium,
	potassium,
	inr,
	hemoglobin,
	hematocrit,
	-- Co-medications
	pgp_inhibit,
	pgp_induce,
	cyp3a4_inhibit,
	cyp3a4_induce,
	-- TAC levels
	tac AS tac_level,
	charttime AS level_time,
	-- TAC doses 
	dose_charttime AS dose_time,
	formulation,
	route,
	dose,
	doses_per_24_hrs,
	-- Time between current level and current dose
	dose_diff_days AS level_dose_timediff,
	-- Time between current level and first treatment day 
	treatment_diff_days AS treatment_days, 
	-- Previous sample 
	LAG(dose) OVER (PARTITION BY subject_id ORDER BY charttime) AS previous_dose,
    LAG(tac) OVER (PARTITION BY subject_id ORDER BY charttime) AS previous_level,
    LAG(route) OVER (PARTITION BY subject_id ORDER BY charttime) AS previous_route,
	-- Time between current and previous level time
	EXTRACT(EPOCH FROM (charttime - LAG(charttime) OVER (PARTITION BY subject_id ORDER BY charttime)))/86400 AS previous_level_timediff,
	-- Time between current level and previous dose 
	EXTRACT(EPOCH FROM (charttime - LAG(dose_charttime) OVER (PARTITION BY subject_id ORDER BY charttime)))/3600 AS previous_dose_timediff

FROM followup_tacdose
	
	
	