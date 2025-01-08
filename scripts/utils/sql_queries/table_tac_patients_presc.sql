-- Create table with subject_ids that have information on TAC levels and prescriptions (dose)
CREATE TABLE tac_patients_presc AS
SELECT subject_id
FROM mimiciv_hosp.labevents
WHERE itemid = 50986 
  AND subject_id IN (
      SELECT prescriptions.subject_id
      FROM mimiciv_hosp.prescriptions
      WHERE LOWER(prescriptions.drug) LIKE '%tacrolimus%'
		AND route NOT IN ('BU','TP')
  )
GROUP BY subject_id;
