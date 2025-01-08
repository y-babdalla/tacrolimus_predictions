-- Add lab results to followup table
DROP TABLE IF EXISTS followup_labs;

CREATE TABLE followup_labs AS 
-- Step 1: lab_enzymes
WITH followup_labs1 AS(
	WITH DateDiffs AS (
		SELECT
			fd.*,
			le.charttime AS lab_charttime,
			le.alt,
			le.alp,
			le.ast,
			le.bilirubin_total, 
			le.ggt,
			ABS(EXTRACT(EPOCH FROM (fd.charttime - le.charttime))) AS date_diff,
			ABS(EXTRACT(EPOCH FROM (fd.charttime - le.charttime))) / 86400 AS date_diff_days
		FROM
			followup_demographics fd
		LEFT JOIN
			labs_enzymes le ON fd.subject_id = le.subject_id
	),
		
	RankedDateDiffs AS (
		SELECT
			*,
			ROW_NUMBER() OVER (PARTITION BY subject_id, charttime ORDER BY date_diff ASC) AS rn
		FROM
			DateDiffs
	),
		
	RN1 AS (
	    SELECT *
	    FROM RankedDateDiffs
	    WHERE rn = 1
	),
		
	RN2 AS (
	    SELECT *
	    FROM RankedDateDiffs
	    WHERE rn = 2
	)
	
	SELECT
	    RN1.subject_id,
		RN1.hadm_id,
		RN1.specimen_id,
	    RN1.charttime,
		RN1.tac,
		RN1.weight,
		RN1.height,
		RN1.weight_datediff,
		RN1.height_datediff,
		RN1.gender,
		RN1.age,
		RN1.race,
		RN1.first_tac_dose,
		RN1.first_tac_level,
		RN1.state,
		CASE WHEN RN1.date_diff_days <= 7 THEN RN1.date_diff_days ELSE NULL END AS rn1_enzymes_daysdiff,
		CASE WHEN RN2.date_diff_days <= 7 THEN RN2.date_diff_days ELSE NULL END AS rn2_enzymes_daysdiff,
		COALESCE(
		    CASE WHEN RN1.date_diff_days <= 7 THEN RN1.alt ELSE NULL END, 
		    CASE WHEN RN2.date_diff_days <= 7 THEN RN2.alt ELSE NULL END
		) AS alt,
		
		COALESCE(
		    CASE WHEN RN1.date_diff_days <= 7 THEN RN1.alp ELSE NULL END, 
		    CASE WHEN RN2.date_diff_days <= 7 THEN RN2.alp ELSE NULL END
		) AS alp,
		
		COALESCE(
		    CASE WHEN RN1.date_diff_days <= 7 THEN RN1.ast ELSE NULL END, 
		    CASE WHEN RN2.date_diff_days <= 7 THEN RN2.ast ELSE NULL END
		) AS ast,
		
		COALESCE(
		    CASE WHEN RN1.date_diff_days <= 7 THEN RN1.bilirubin_total ELSE NULL END, 
		    CASE WHEN RN2.date_diff_days <= 7 THEN RN2.bilirubin_total ELSE NULL END
		) AS bilirubin,
		
		COALESCE(
		    CASE WHEN RN1.date_diff_days <= 7 THEN RN1.ggt ELSE NULL END, 
		    CASE WHEN RN2.date_diff_days <= 7 THEN RN2.ggt ELSE NULL END
		) AS ggt
		-- RN1.date_diff_days AS rn1_enzymes_daysdiff, 
		-- RN2.date_diff_days AS rn2_enzymes_daysdiff,
	 --    COALESCE(RN1.alt, RN2.alt) AS alt,
	 --    COALESCE(RN1.alp, RN2.alp) AS alp,
	 --    COALESCE(RN1.ast, RN2.ast) AS ast,
	 --    COALESCE(RN1.bilirubin_total, RN2.bilirubin_total) AS bilirubin,
	 --    COALESCE(RN1.ggt, RN2.ggt) AS ggt
		
	FROM
	    RN1
	LEFT JOIN
	    RN2 ON RN1.subject_id = RN2.subject_id AND RN1.charttime = RN2.charttime
),

-- Step 2: labs chemistry
followup_labs2 AS(
	WITH DateDiffs AS (
		SELECT
			fl1.*,
			lc.charttime AS lab_charttime,
			lc.albumin,
			lc.bun,
			lc.creatinine,
			lc.sodium, 
			lc.potassium,
			ABS(EXTRACT(EPOCH FROM (fl1.charttime - lc.charttime))) AS date_diff,
			ABS(EXTRACT(EPOCH FROM (fl1.charttime - lc.charttime))) / 86400 AS date_diff_days
		FROM
			followup_labs1 fl1
		LEFT JOIN
			labs_chemistry lc ON fl1.subject_id = lc.subject_id
	),
		
	RankedDateDiffs AS (
		SELECT
			*,
			ROW_NUMBER() OVER (PARTITION BY subject_id, charttime ORDER BY date_diff ASC) AS rn
		FROM
			DateDiffs
	),
		
	RN1 AS (
	    SELECT *
	    FROM RankedDateDiffs
	    WHERE rn = 1
	),
		
	RN2 AS (
	    SELECT *
	    FROM RankedDateDiffs
	    WHERE rn = 2
	)
	
	SELECT
	    RN1.subject_id,
		RN1.hadm_id,
		RN1.specimen_id,
	    RN1.charttime,
		RN1.tac,
		RN1.weight,
		RN1.height,
		RN1.weight_datediff,
		RN1.height_datediff,
		RN1.gender,
		RN1.age,
		RN1.race,
		RN1.first_tac_dose,
		RN1.first_tac_level,
		RN1.state,
		RN1.rn1_enzymes_daysdiff, 
		RN1.rn2_enzymes_daysdiff,
	    RN1.alt,
	    RN1.alp,
	    RN1.ast,
	    RN1.bilirubin,
	    RN1.ggt,
		CASE WHEN RN1.date_diff_days <= 7 THEN RN1.date_diff_days ELSE NULL END AS rn1_chemistry_daysdiff,
		CASE WHEN RN2.date_diff_days <= 7 THEN RN2.date_diff_days ELSE NULL END AS rn2_chemistry_daysdiff,
		COALESCE(
		    CASE WHEN RN1.date_diff_days <= 7 THEN RN1.albumin ELSE NULL END, 
		    CASE WHEN RN2.date_diff_days <= 7 THEN RN2.albumin ELSE NULL END
		) AS albumin,
		COALESCE(
		    CASE WHEN RN1.date_diff_days <= 7 THEN RN1.bun ELSE NULL END, 
		    CASE WHEN RN2.date_diff_days <= 7 THEN RN2.bun ELSE NULL END
		) AS bun,
		COALESCE(
		    CASE WHEN RN1.date_diff_days <= 7 THEN RN1.creatinine ELSE NULL END, 
		    CASE WHEN RN2.date_diff_days <= 7 THEN RN2.creatinine ELSE NULL END
		) AS creatinine,
		COALESCE(
		    CASE WHEN RN1.date_diff_days <= 7 THEN RN1.sodium ELSE NULL END, 
		    CASE WHEN RN2.date_diff_days <= 7 THEN RN2.sodium ELSE NULL END
		) AS sodium,
		COALESCE(
		    CASE WHEN RN1.date_diff_days <= 7 THEN RN1.potassium ELSE NULL END, 
		    CASE WHEN RN2.date_diff_days <= 7 THEN RN2.potassium ELSE NULL END
		) AS potassium
		-- RN1.date_diff_days AS rn1_chemistry_daysdiff, 
		-- RN2.date_diff_days AS rn2_chemistry_daysdiff,
		-- COALESCE(RN1.albumin, RN2.albumin) AS albumin,
	 --    COALESCE(RN1.bun, RN2.bun) AS bun,
	 --    COALESCE(RN1.creatinine, RN2.creatinine) AS creatinine,
	 --    COALESCE(RN1.sodium, RN2.sodium) AS sodium,
	 --    COALESCE(RN1.potassium, RN2.potassium) AS potassium
	FROM
	    RN1
	LEFT JOIN
	    RN2 ON RN1.subject_id = RN2.subject_id AND RN1.charttime = RN2.charttime
	
),

-- Step 3: labs INR
followup_labs3 AS(
	WITH DateDiffs AS (
		SELECT
			fl2.*,
			li.charttime AS lab_charttime,
			li.inr,
			ABS(EXTRACT(EPOCH FROM (fl2.charttime - li.charttime))) AS date_diff,
			ABS(EXTRACT(EPOCH FROM (fl2.charttime - li.charttime))) / 86400 AS date_diff_days
		FROM
			followup_labs2 fl2
		LEFT JOIN
			labs_inr li ON fl2.subject_id = li.subject_id
	),
		
	RankedDateDiffs AS (
		SELECT
			*,
			ROW_NUMBER() OVER (PARTITION BY subject_id, charttime ORDER BY date_diff ASC) AS rn
		FROM
			DateDiffs
	),
		
	RN1 AS (
	    SELECT *
	    FROM RankedDateDiffs
	    WHERE rn = 1
	)
	
	SELECT
	    RN1.subject_id,
		RN1.hadm_id,
		RN1.specimen_id,
	    RN1.charttime,
		RN1.tac,
		RN1.weight,
		RN1.height,
		RN1.weight_datediff,
		RN1.height_datediff,
		RN1.gender,
		RN1.age,
		RN1.race,
		RN1.first_tac_dose,
		RN1.first_tac_level,
		RN1.state,
		RN1.rn1_enzymes_daysdiff, 
		RN1.rn2_enzymes_daysdiff,
	    RN1.alt,
	    RN1.alp,
	    RN1.ast,
	    RN1.bilirubin,
	    RN1.ggt,
		RN1.rn1_chemistry_daysdiff, 
		RN1.rn2_chemistry_daysdiff,
		RN1.albumin,
	    RN1.bun,
	    RN1.creatinine,
	    RN1.sodium,
	    RN1.potassium,
		CASE WHEN RN1.date_diff_days<=7 THEN RN1.date_diff_days ELSE NULL END AS rn1_inr_daysdiff,
		CASE WHEN RN1.date_diff_days<=7 THEN RN1.inr ELSE NULL END AS inr
	FROM
	    RN1
),

-- Step 4: labs hemogram
followup_labs4 AS(
	WITH DateDiffs AS (
		SELECT
			fl3.*,
			lh.charttime AS lab_charttime,
			lh.hematocrit,
			lh.hemoglobin,
			ABS(EXTRACT(EPOCH FROM (fl3.charttime - lh.charttime))) AS date_diff,
			ABS(EXTRACT(EPOCH FROM (fl3.charttime - lh.charttime))) / 86400 AS date_diff_days
		FROM
			followup_labs3 fl3
		LEFT JOIN
			labs_hemogram lh ON fl3.subject_id = lh.subject_id
	),
		
	RankedDateDiffs AS (
		SELECT
			*,
			ROW_NUMBER() OVER (PARTITION BY subject_id, charttime ORDER BY date_diff ASC) AS rn
		FROM
			DateDiffs
	),
		
	RN1 AS (
	    SELECT *
	    FROM RankedDateDiffs
	    WHERE rn = 1
	),

	RN2 AS (
		SELECT *
		FROM RankedDateDiffs
		WHERE rn = 2
	)
	
	SELECT
	    RN1.subject_id,
		RN1.hadm_id,
		RN1.specimen_id,
	    RN1.charttime,
		RN1.tac,
		RN1.weight,
		RN1.height,
		-- RN1.weight_datediff,
		-- RN1.height_datediff,
		RN1.gender,
		RN1.age,
		RN1.race,
		RN1.first_tac_dose,
		RN1.first_tac_level,
		RN1.state,
		-- RN1.rn1_enzymes_daysdiff, 
		-- RN1.rn2_enzymes_daysdiff,
	    RN1.alt,
	    RN1.alp,
	    RN1.ast,
	    RN1.bilirubin,
	    RN1.ggt,
		-- RN1.rn1_chemistry_daysdiff, 
		-- RN1.rn2_chemistry_daysdiff,
		RN1.albumin,
	    RN1.bun,
	    RN1.creatinine,
	    RN1.sodium,
	    RN1.potassium,
		-- RN1.rn1_inr_daysdiff, 
		RN1.inr,
		-- CASE WHEN RN1.date_diff_days <= 7 THEN RN1.date_diff_days ELSE NULL END AS rn1_hemogram_daysdiff,
		-- CASE WHEN RN2.date_diff_days <= 7 THEN RN2.date_diff_days ELSE NULL END AS rn2_hemogram_daysdiff,
		COALESCE(
		    CASE WHEN RN1.date_diff_days <= 7 THEN RN1.hematocrit ELSE NULL END, 
		    CASE WHEN RN2.date_diff_days <= 7 THEN RN2.hematocrit ELSE NULL END
		) AS hematocrit,
		
		COALESCE(
		    CASE WHEN RN1.date_diff_days <= 7 THEN RN1.hemoglobin ELSE NULL END, 
		    CASE WHEN RN2.date_diff_days <= 7 THEN RN2.hemoglobin ELSE NULL END
		) AS hemoglobin	
		-- RN1.date_diff_days AS rn1_hemogram_daysdiff, 
		-- RN2.date_diff_days AS rn2_hemogram_daysdiff,
		-- COALESCE(RN1.hematocrit, RN2.hematocrit) AS hematocrit,
		-- COALESCE(RN1.hemoglobin, RN2.hemoglobin) AS hemoglobin

	FROM
	    RN1
	LEFT JOIN
	    RN2 ON RN1.subject_id = RN2.subject_id AND RN1.charttime = RN2.charttime
)

SELECT *
FROM followup_labs4