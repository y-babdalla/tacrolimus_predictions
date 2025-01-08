-- Add PGP and CYP3A4/5 inhibitors/inducers information
DROP TABLE IF EXISTS followup_drugs;

CREATE TABLE followup_drugs AS
	
WITH RankedRows AS (
	SELECT 
		fl.*,
		d.starttime, 
		d.stoptime,
		CASE 
			WHEN d.pgp = 0 THEN 1 
			ELSE 0 
		END AS pgp_inhibit,
		CASE 
			WHEN d.pgp = 1 THEN 1 
			ELSE 0 
		END AS pgp_induce,
		CASE 
			WHEN d.cyp3a4 = 0 THEN 1 
			ELSE 0 
		END AS cyp3a4_inhibit,
		CASE 
			WHEN d.cyp3a4 = 1 THEN 1 
			ELSE 0 
		END AS cyp3a4_induce,
		ROW_NUMBER() OVER (PARTITION BY fl.subject_id, fl.charttime ORDER BY ABS(EXTRACT(EPOCH FROM (fl.charttime - d.starttime)))) AS rn
	FROM followup_labs fl
	LEFT JOIN drugs_pgp_3a4 d
	ON fl.subject_id = d.subject_id AND fl.charttime BETWEEN d.starttime AND d.stoptime
	)
SELECT 
	subject_id,
	hadm_id,
	specimen_id,
	charttime,
	tac,
	weight,
	height,
	-- weight_datediff,
	-- height_datediff,
	gender,
	age,
	race,
	first_tac_dose,
	first_tac_level,
	state,
	-- rn1_enzymes_daysdiff,
	-- rn2_enzymes_daysdiff,
	alt,
	alp,
	ast,
	bilirubin,
	ggt,
	-- rn1_chemistry_daysdiff,
	-- rn2_chemistry_daysdiff,
	albumin,
	bun,
	creatinine,
	sodium,
	potassium,
	-- rn1_inr_daysdiff,
	inr,
	-- rn1_hemogram_daysdiff,
	-- rn2_hemogram_daysdiff,
	hematocrit,
	hemoglobin,
	pgp_inhibit,
	pgp_induce,
	cyp3a4_inhibit,
	cyp3a4_induce
FROM RankedRows
WHERE rn = 1