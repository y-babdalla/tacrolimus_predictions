-- Coagulation labevents extraction
CREATE TABLE labs_inr AS 
SELECT
    MAX(subject_id) AS subject_id
    , MAX(hadm_id) AS hadm_id
    , MAX(charttime) AS charttime
    , le.specimen_id
    -- convert from itemid into a meaningful column
    , MAX(CASE WHEN itemid = 51237 THEN valuenum ELSE NULL END) AS inr
FROM mimiciv_hosp.labevents le
WHERE le.itemid IN
    (
        51237 -- INR
    )
    AND valuenum IS NOT NULL
	AND subject_id IN (SELECT subject_id FROM tac_patients_presc)
GROUP BY le.specimen_id
;