-- Step 1: Create the patient_demographics_with_tacrolimus table
CREATE TABLE patient_demographics (
    subject_id INT PRIMARY KEY,
    gender VARCHAR(1),
    year_of_birth INT,
    race VARCHAR(50),
    first_tac_emar TIMESTAMP,
	first_tac_presc TIMESTAMP, 
	first_tac_level TIMESTAMP,
	first_tac_date TIMESTAMP,
	first_tac_dose TIMESTAMP
);

-- Step 2: Populate the patient_demographics_with_tacrolimus table

-- Create a CTE to fetch the subject_id and their standardized race,
-- defaulting to 'Other' if no match is found in the race_mapping table.
WITH race_details AS (
    SELECT
        adm.subject_id,
        COALESCE(rm.standardized_race, 'Other') AS standardized_race
    FROM (
        SELECT 
            subject_id, 
            race 
        FROM mimiciv_hosp.admissions
        WHERE subject_id IN (SELECT subject_id FROM tac_patients_presc)
    ) AS adm
    LEFT JOIN race_mapping rm ON adm.race = rm.original_race
),

-- Create a CTE to calculate the number of distinct races per subject_id
-- and the number of distinct races excluding 'Other'.
race_count AS (
    SELECT
        subject_id,
        COUNT(DISTINCT standardized_race) AS race_count,
        COUNT(DISTINCT CASE WHEN standardized_race = 'Other' THEN NULL ELSE standardized_race END) AS non_other_race_count
    FROM race_details
    GROUP BY subject_id
),

-- Create a CTE to determine the final standardized race for each subject_id
-- based on the specified conditions.
final_race AS (
    SELECT
        rd.subject_id,
        CASE
            WHEN rc.non_other_race_count = 1 THEN MAX(CASE WHEN rd.standardized_race != 'Other' THEN rd.standardized_race END)
            WHEN rc.race_count > 2 THEN 'Multiple'
            ELSE 'Other'
        END AS standardized_race
    FROM race_details rd
    JOIN race_count rc ON rd.subject_id = rc.subject_id
    GROUP BY rd.subject_id, rc.race_count, rc.non_other_race_count
),

-- Fetch patient details (subject_id, gender, year_of_birth)
patient_details AS (
    SELECT 
        subject_id,
        gender,
        anchor_year - anchor_age AS year_of_birth
    FROM mimiciv_hosp.patients
    WHERE subject_id IN (SELECT subject_id FROM tac_patients_presc)
),

-- Fetch the first date of tacrolimus administration from the emar table
tac_dates_emar AS (
    SELECT
        subject_id,
        MIN(charttime) AS first_tac_emar
    FROM mimiciv_hosp.emar
    WHERE LOWER(medication) LIKE '%tacrolimus%'
      AND event_txt = 'Administered'
		AND subject_id IN (SELECT subject_id FROM tac_patients_presc)
    GROUP BY subject_id
),

-- Fetch the first date of tacrolimus prescription
tac_dates_presc AS (
    SELECT
        subject_id,
        MIN(starttime) AS first_tac_presc
    FROM mimiciv_hosp.prescriptions
    WHERE LOWER(drug) LIKE '%tacrolimus%'
		AND subject_id IN (SELECT subject_id FROM tac_patients_presc)
    GROUP BY subject_id
),

-- Fetch the first date of tacrolimus level
tac_dates_levels AS (
    SELECT
        subject_id,
        MIN(charttime) AS first_tac_level
    FROM mimiciv_hosp.labevents
    WHERE itemid = 50986
		AND subject_id IN (SELECT subject_id FROM tac_patients_presc)
    GROUP BY subject_id
)

-- Combine the results from final_race, patient_details, and tacrolimus_dates
INSERT INTO patient_demographics (subject_id, gender, year_of_birth, race, first_tac_emar, first_tac_presc, first_tac_level,first_tac_date,first_tac_dose)
SELECT 
    pd.subject_id,
    pd.gender,
    pd.year_of_birth,
    fr.standardized_race AS race,
    de.first_tac_emar,
	dp.first_tac_presc,
	dl.first_tac_level,
	LEAST(de.first_tac_emar, dp.first_tac_presc, dl.first_tac_level) AS first_tac_date,
	LEAST(de.first_tac_emar, dp.first_tac_presc) AS first_tac_dose
FROM 
    patient_details pd
JOIN 
    final_race fr ON pd.subject_id = fr.subject_id
LEFT JOIN 
    tac_dates_emar de ON pd.subject_id = de.subject_id
LEFT JOIN
	tac_dates_presc dp ON pd.subject_id = dp.subject_id
LEFT JOIN
	tac_dates_levels dl ON pd.subject_id = dl.subject_id
;
