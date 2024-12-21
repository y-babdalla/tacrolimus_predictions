-- extract chemistry labs
-- excludes point of care tests (very rare)
-- blood gas measurements are *not* included in this query
CREATE TABLE labs_chemistry AS 
SELECT
    MAX(subject_id) AS subject_id
    , MAX(hadm_id) AS hadm_id
    , MAX(charttime) AS charttime
    , le.specimen_id
    -- convert from itemid into a meaningful column
    , MAX(
        CASE WHEN itemid = 50862 AND valuenum <= 10 THEN valuenum ELSE NULL END
    ) AS albumin
	, MAX(
        CASE WHEN itemid = 51006 AND valuenum <= 300 THEN valuenum ELSE NULL END
    ) AS bun
    , MAX(
        CASE WHEN itemid = 50912 AND valuenum <= 150 THEN valuenum ELSE NULL END
    ) AS creatinine
    , MAX(
        CASE WHEN itemid = 50983 AND valuenum <= 200 THEN valuenum ELSE NULL END
    ) AS sodium
    , MAX(
        CASE WHEN itemid = 50971 AND valuenum <= 30 THEN valuenum ELSE NULL END
    ) AS potassium
FROM mimiciv_hosp.labevents le
WHERE le.itemid IN
    (
        -- comment is: LABEL | CATEGORY | FLUID | NUMBER OF ROWS IN LABEVENTS
        50862 -- ALBUMIN | CHEMISTRY | BLOOD | 146697
        , 50912 -- CREATININE | CHEMISTRY | BLOOD | 797476
        , 50971 -- POTASSIUM | CHEMISTRY | BLOOD | 845825
        , 50983 -- SODIUM | CHEMISTRY | BLOOD | 808489
        , 51006  -- UREA NITROGEN | CHEMISTRY | BLOOD | 791925
    )
    AND valuenum IS NOT NULL
    -- lab values cannot be 0 and cannot be negative
    AND valuenum > 0 
	AND subject_id IN (SELECT subject_id FROM tac_patients_presc)
GROUP BY le.specimen_id
;