-- begin query that extracts the data
CREATE TABLE labs_enzymes AS
SELECT
    MAX(subject_id) AS subject_id
    , MAX(hadm_id) AS hadm_id
    , MAX(charttime) AS charttime
    , le.specimen_id
    -- convert from itemid into a meaningful column
    , MAX(CASE WHEN itemid = 50861 THEN valuenum ELSE NULL END) AS alt
    , MAX(CASE WHEN itemid = 50863 THEN valuenum ELSE NULL END) AS alp
    , MAX(CASE WHEN itemid = 50878 THEN valuenum ELSE NULL END) AS ast
    , MAX(
        CASE WHEN itemid = 50885 THEN valuenum ELSE NULL END
    ) AS bilirubin_total
    , MAX(CASE WHEN itemid = 50927 THEN valuenum ELSE NULL END) AS ggt
FROM mimiciv_hosp.labevents le
WHERE le.itemid IN
    (
        50861 -- Alanine transaminase (ALT)
        , 50863 -- Alkaline phosphatase (ALP)
        , 50878 -- Aspartate transaminase (AST)
        , 50885 -- total bili
        , 50927 -- Gamma Glutamyltransferase (GGT)
    )
    AND valuenum IS NOT NULL
    -- lab values cannot be 0 and cannot be negative
    AND valuenum > 0
	AND subject_id IN (SELECT subject_id FROM tac_patients_presc)
GROUP BY le.specimen_id
;