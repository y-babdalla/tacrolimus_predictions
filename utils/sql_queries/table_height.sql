-- Select all recordings of height per patient
CREATE TABLE height AS
	
-- Step 1: Height from hosp module
WITH height_hosp AS(
	WITH RankedHeights AS (
	    SELECT
	        omr.subject_id, omr.chartdate, omr.seq_num,
	        -- Ensure that all heights are in centimeters
	        ROUND(CAST(omr.result_value AS NUMERIC) * 2.54, 2) AS height,
	        ROW_NUMBER() OVER (PARTITION BY omr.subject_id, omr.chartdate ORDER BY omr.seq_num DESC) AS rn
	    FROM mimiciv_hosp.omr omr
	    WHERE omr.result_value IS NOT NULL
	        -- Height (measured in inches)
	        AND LOWER(omr.result_name) LIKE 'height%'
	        AND omr.subject_id IN (SELECT subject_id FROM tac_patients_presc)
	        -- filter out bad heights after conversion to centimeters
	        AND (CAST(omr.result_value AS NUMERIC) * 2.54) > 120 
	        AND (CAST(omr.result_value AS NUMERIC) * 2.54) < 230
	)
	SELECT subject_id, chartdate, height
	FROM RankedHeights
	WHERE rn = 1),


-- Step 2: Height from ICU module 
height_icu AS(
		WITH ht_in AS (
	    SELECT
	        c.subject_id, c.hadm_id, c.stay_id, c.charttime
	        -- Ensure that all heights are in centimeters
	        , ROUND(CAST(c.valuenum * 2.54 AS NUMERIC), 2) AS height
	        , c.valuenum AS height_orig
	    FROM mimiciv_icu.chartevents c
	    WHERE c.valuenum IS NOT NULL
	        -- Height (measured in inches)
	        AND c.itemid = 226707
	)
	
	, ht_cm AS (
	    SELECT
	        c.subject_id, c.hadm_id, c.stay_id, c.charttime
	        -- Ensure that all heights are in centimeters
	        , ROUND(CAST(c.valuenum AS NUMERIC), 2) AS height
	    FROM mimiciv_icu.chartevents c
	    WHERE c.valuenum IS NOT NULL
	        -- Height cm
	        AND c.itemid = 226730
	)
	
	-- merge cm/height, only take 1 value per charted row
	, ht_stg0 AS (
	    SELECT
	        COALESCE(h1.subject_id, h1.subject_id) AS subject_id
			, COALESCE(h1.hadm_id, h1.hadm_id) AS hadm_id
	        , COALESCE(h1.stay_id, h1.stay_id) AS stay_id
	        , COALESCE(h1.charttime, h1.charttime) AS charttime
	        , COALESCE(h1.height, h2.height) AS height
	    FROM ht_cm h1
	    FULL OUTER JOIN ht_in h2
	        ON h1.subject_id = h2.subject_id
	            AND h1.charttime = h2.charttime
	)
		
	, ht_stg1 AS (
	    SELECT
	        subject_id, charttime, height,
	        ROW_NUMBER() OVER (PARTITION BY subject_id, DATE(charttime) ORDER BY charttime DESC) AS rn
	    FROM ht_stg0
	)
	
	SELECT subject_id, DATE(charttime) AS chartdate, height
	FROM ht_stg1
	WHERE rn = 1
		AND height IS NOT NULL
	    -- filter out bad heights
	    AND height > 120 AND height < 230
		AND subject_id IN (SELECT subject_id FROM tac_patients_presc)
),

-- Step 3: Merge icu and hosp heights 
merged_heights AS (
    SELECT
        COALESCE(h1.subject_id, h2.subject_id) AS subject_id,
        COALESCE(h1.chartdate, h2.chartdate) AS chartdate,
        COALESCE(h1.height, h2.height) AS height
    FROM height_hosp h1
    FULL OUTER JOIN height_icu h2
        ON h1.subject_id = h2.subject_id
        AND h1.chartdate = h2.chartdate
)

SELECT subject_id, chartdate, height
FROM merged_heights