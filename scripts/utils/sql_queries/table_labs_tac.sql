-- TAC labevents extraction
CREATE TABLE labs_tac AS 
SELECT
    MAX(subject_id) AS subject_id
    , MAX(hadm_id) AS hadm_id
    , MAX(charttime) AS charttime
    , MAX(le.specimen_id) AS specimen_id 
    -- convert from itemid into a meaningful column
    , MAX(CASE WHEN itemid = 50986 THEN valuenum ELSE NULL END) AS tac
FROM mimiciv_hosp.labevents le
WHERE le.itemid IN
    (
        50986 -- TAC
    )
    AND valuenum IS NOT NULL
	AND valuenum > 0
	AND subject_id IN (SELECT subject_id FROM tac_patients_presc)
GROUP BY subject_id, charttime
