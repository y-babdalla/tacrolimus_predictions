-- Add weight and height information to followup table
DROP TABLE IF EXISTS followup_demographics;

CREATE TABLE followup_demographics AS 
-- Step 1: weight 
WITH merge_weight AS 
	(WITH DateDiffs AS (
		SELECT
			l.*,
			w.chartdate,
			w.weight,
			ABS(DATE(l.charttime)- w.chartdate) AS date_diff
		FROM
			labs_tac l
		LEFT JOIN
			weight w ON l.subject_id = w.subject_id
	),
	RankedDateDiffs AS (
		SELECT
			*,
			ROW_NUMBER() OVER (PARTITION BY subject_id, charttime ORDER BY date_diff ASC) AS rn
		FROM
			DateDiffs
	),
	ClosestWeights AS(
		SELECT
			subject_id,
			hadm_id,
			specimen_id,
			charttime,
			tac,
			weight,
			date_diff AS weight_datediff
		FROM
			RankedDateDiffs
		WHERE rn = 1
		)
	SELECT
        subject_id,
        hadm_id,
        specimen_id,
        charttime,
        tac,
        CASE WHEN weight_datediff <= 365 THEN weight ELSE NULL END AS weight,
		CASE WHEN weight_datediff <= 365 THEN weight_datediff ELSE NULL END AS weight_datediff
    FROM
        ClosestWeights
	), 
-- Step 2: add height 
merge_height AS	(
	WITH DateDiffs_h AS (
		SELECT
			mw.*,
			h.chartdate,
			h.height,
			ABS(DATE(mw.charttime)- h.chartdate) AS date_diff_h
		FROM
			merge_weight mw
		LEFT JOIN
			height h ON mw.subject_id = h.subject_id
	),
	RankedDateDiffs_h AS (
		SELECT
			*,
			ROW_NUMBER() OVER (PARTITION BY subject_id, charttime ORDER BY date_diff_h ASC) AS rn
		FROM
			DateDiffs_h
	)
	SELECT
		subject_id,
		hadm_id,
		specimen_id,
		charttime,
		tac,
		weight,
		height,
		weight_datediff,
		date_diff_h AS height_datediff
	FROM
		RankedDateDiffs_h
	WHERE
		rn = 1
	),

-- Step 3: add demographics 
merge_demographics AS (
    SELECT
        mh.*,
        pd.gender,
		(EXTRACT(YEAR FROM mh.charttime) - pd.year_of_birth) AS age,
		pd.race,
		pd.first_tac_dose,
		pd.first_tac_level
    FROM
        merge_height mh
    LEFT JOIN
        patient_demographics pd ON mh.subject_id = pd.subject_id
)

-- Step 4: add state
,merge_state AS(
	WITH RankedRows AS (
	SELECT 
		md.*,
		s.intime, 
		s.outtime,
		COALESCE(s.state, 'home') AS state,
		ROW_NUMBER() OVER (PARTITION BY md.subject_id, md.charttime ORDER BY ABS(EXTRACT(EPOCH FROM (md.charttime - s.intime)))) AS rn
	FROM merge_demographics md
	LEFT JOIN state s 
	ON md.subject_id = s.subject_id AND md.charttime BETWEEN s.intime AND s.outtime
	)
	SELECT subject_id, 
		hadm_id,
		specimen_id,
		charttime, 
		tac,
		weight,
		height,
		weight_datediff,
		height_datediff,
		gender,
		age,
		race,
		first_tac_dose,
		first_tac_level,
		state
	FROM RankedRows
	WHERE rn = 1
)

SELECT *
FROM merge_state