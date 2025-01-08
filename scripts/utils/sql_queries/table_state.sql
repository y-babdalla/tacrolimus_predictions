-- Create a table with the state of patients (ICU,hosp,home) 
CREATE TABLE state AS
SELECT *,
	CASE
        WHEN LOWER(careunit) LIKE '%icu%' THEN 'icu'
        WHEN LOWER(eventtype) = 'discharge' THEN 'home'
        ELSE 'hosp'
    END AS state
FROM mimiciv_hosp.transfers
WHERE subject_id IN (SELECT subject_id FROM tac_patients_presc)
