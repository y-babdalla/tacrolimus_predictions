
CREATE TABLE followup_tacdose2 AS
	
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
	     -- LIMIT 1
	    ) dt ON true -- ensure all records of followup_drugs is included even if no correponding dose found
	)

-- Filter doses that meet the time difference conditions
    , stg1 AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY charttime ORDER BY dose_diff_days ASC) AS dose_rank
        FROM stg0
        WHERE dose_diff_days <= 365
          AND dose_diff_days >= (1.0 / doses_per_24_hrs)
          AND dose IS NOT NULL
    )

    -- Select the first valid dose for each level
    SELECT *
    FROM stg1
    WHERE dose_rank = 1
