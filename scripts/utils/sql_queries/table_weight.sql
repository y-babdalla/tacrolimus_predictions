-- Select all recordings of weight per patient
CREATE TABLE weight AS

-- Step 1: Weight from hosp module
WITH weight_hosp AS(
	WITH RankedWeights AS (
		SELECT
		    omr.subject_id, omr.chartdate, omr.seq_num,
		    -- Ensure that all weights are in kg
		    ROUND(CAST(omr.result_value AS NUMERIC) * 0.45, 2) AS weight,
			ROW_NUMBER() OVER (PARTITION BY omr.subject_id, omr.chartdate ORDER BY omr.seq_num DESC) AS rn
		FROM mimiciv_hosp.omr omr
		WHERE omr.result_value IS NOT NULL
		    -- Weight (measured in kg)
		    AND LOWER(omr.result_name) LIKE 'weight%'
		    AND omr.subject_id IN (SELECT subject_id FROM tac_patients_presc)
			AND (CAST(omr.result_value AS NUMERIC)* 0.45) >17 
			AND (CAST(omr.result_value AS NUMERIC)* 0.45) <165
	)
	SELECT subject_id, chartdate, weight
	FROM RankedWeights
	-- Keep only the last measurement per subject_id and chartdate
	WHERE rn = 1
	),

-- Step 2: Weight from ICU module
weight_icu AS(
	-- This query extracts weights for adult ICU patients with start/stop times
	-- if an admission weight is given, then this is assigned from intime to outtime
	WITH wt_stg AS (
	    SELECT
	        c.stay_id,
			c.subject_id
	        , c.charttime
	        , CASE WHEN c.itemid = 226512 THEN 'admit'
	            ELSE 'daily' END AS weight_type
	        -- TODO: eliminate obvious outliers if there is a reasonable weight
	        , c.valuenum AS weight
	    FROM mimiciv_icu.chartevents c
	    WHERE c.valuenum IS NOT NULL
	        AND c.itemid IN
	        (
	            226512 -- Admit Wt
	            , 224639 -- Daily Weight
	        )
	        AND c.valuenum > 0
			AND subject_id IN (SELECT subject_id FROM tac_patients_presc)
	)
	
	-- assign ascending row number within each stay_id and weight_type
	, wt_stg1 AS (
	    SELECT
	        stay_id,
			subject_id
	        , charttime
	        , weight_type
	        , weight
	        , ROW_NUMBER() OVER (
	            PARTITION BY stay_id, weight_type ORDER BY charttime
	        ) AS rn
	    FROM wt_stg
	    WHERE weight IS NOT NULL
	)
	
	-- change charttime to intime for the first admission weight recorded
	, wt_stg2 AS (
	    SELECT
	        wt_stg1.stay_id,
			wt_stg1.subject_id
	        , ie.intime, ie.outtime
	        , wt_stg1.weight_type
	        , CASE WHEN wt_stg1.weight_type = 'admit' AND wt_stg1.rn = 1
	            THEN (ie.intime - INTERVAL '2 HOURS')
	            ELSE wt_stg1.charttime END AS starttime
	        , wt_stg1.weight
	    FROM wt_stg1
	    INNER JOIN mimiciv_icu.icustays ie
	        ON ie.stay_id = wt_stg1.stay_id
	)
	
	-- calculate endtime before next starttime in stay_id or if no next starttime then 2 hours after endtime
	, wt_stg3 AS (
	    SELECT
	        stay_id,
			subject_id
	        , intime, outtime
	        , starttime
	        , COALESCE(
	            LEAD(starttime) OVER (PARTITION BY stay_id ORDER BY starttime)
	            , (outtime + INTERVAL '2 HOURS')
	        ) AS endtime
	        , weight
	        , weight_type
	    FROM wt_stg2
	)
	
	-- this table is the start/stop times from admit/daily weight in charted data
	, wt1 AS (
	    SELECT
	        stay_id,
			subject_id 
	        , starttime
	        , COALESCE(
	            endtime
	            , LEAD(
	                starttime
	            ) OVER (PARTITION BY stay_id ORDER BY starttime)
	            -- impute ICU discharge as the end of the final weight measurement
	            -- plus a 2 hour "fuzziness" window
	            , (outtime + INTERVAL '2 HOURS')
	        ) AS endtime
	        , weight
	        , weight_type
	    FROM wt_stg3
	)
	
	-- if the intime for the patient is < the first charted daily weight
	-- then we will have a "gap" at the start of their stay
	-- to prevent this, we look for these gaps and backfill the first weight
	-- this adds (153255-149657)=3598 rows, meaning this fix helps for up
	-- to 3598 stay_id
	, wt_fix AS (
	    SELECT ie.stay_id,
			ie.subject_id
	        -- we add a 2 hour "fuzziness" window
	        , (ie.intime - INTERVAL '2 HOURS') AS starttime
	        , wt.starttime AS endtime
	        , wt.weight
	        , wt.weight_type
	    FROM mimiciv_icu.icustays ie
	    INNER JOIN
	        -- the below subquery returns one row for each unique stay_id
	        -- the row contains: the first starttime and the corresponding weight
	        (
	            SELECT wt1.stay_id, wt1.starttime, wt1.weight
	                , weight_type
	                , ROW_NUMBER() OVER (
	                    PARTITION BY wt1.stay_id ORDER BY wt1.starttime
	                ) AS rn
	            FROM wt1
	        ) wt
	        ON ie.stay_id = wt.stay_id
	            AND wt.rn = 1
	            AND ie.intime < wt.starttime
				AND ie.subject_id IN (SELECT subject_id FROM tac_patients_presc)
	)
	
	-- add the backfill rows to the main weight table
	, wt_merged0 AS (
		SELECT
	    wt1.stay_id
		, wt1.subject_id
	    , wt1.starttime 
	    -- , wt1.endtime
	    , wt1.weight
	    -- , wt1.weight_type
	FROM wt1
	UNION ALL
	SELECT
	    wt_fix.stay_id,
		wt_fix.subject_id
	    , wt_fix.starttime 
	    -- , wt_fix.endtime
	    , wt_fix.weight
	    -- , wt_fix.weight_type
	FROM wt_fix)
	
	-- Keep only the last measurement per day
	, wt_merged1 AS(
		SELECT
			subject_id, starttime, weight,
			ROW_NUMBER() OVER (PARTITION BY subject_id, DATE(starttime) ORDER BY starttime DESC) AS rn
		FROM wt_merged0
	)
		
	SELECT subject_id, DATE(starttime) AS chartdate, weight 
	FROM wt_merged1
	WHERE rn = 1
	AND weight IS NOT NULL
	AND weight > 17 AND weight < 165
	AND subject_id IN (SELECT subject_id FROM tac_patients_presc)
),

-- Step 3: Merge icu and hosp weights 
merged_weights AS (
    SELECT
        COALESCE(w1.subject_id, w2.subject_id) AS subject_id,
        COALESCE(w1.chartdate, w2.chartdate) AS chartdate,
        COALESCE(w1.weight, w2.weight) AS weight
    FROM weight_hosp w1
    FULL OUTER JOIN weight_icu w2
        ON w1.subject_id = w2.subject_id
        AND w1.chartdate = w2.chartdate
)

SELECT subject_id, chartdate, weight
FROM merged_weights