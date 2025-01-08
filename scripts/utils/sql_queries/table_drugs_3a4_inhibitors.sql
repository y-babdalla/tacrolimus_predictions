-- Save 3a4_inhibitors
CREATE TABLE drugs_3a4_inhibitors AS
SELECT DISTINCT
	drug,
	formulary_drug_cd,
	gsn,
	ndc,
	-- prod_strength,
	form_unit_disp,
	route
FROM mimiciv_hosp.prescriptions
WHERE subject_id IN (SELECT subject_id FROM tac_patients_presc)
	AND (
       LOWER(drug) LIKE '%adagrasib%' OR
        LOWER(drug) LIKE '%atazanavir%' OR
        LOWER(drug) LIKE '%ceritinib%' OR
        LOWER(drug) LIKE '%clarithromycin%' OR
        LOWER(drug) LIKE '%cobicistat%' OR
        LOWER(drug) LIKE '%darunavir%' OR
        LOWER(drug) LIKE '%idelalisib%' OR
        LOWER(drug) LIKE '%indinavir%' OR
        LOWER(drug) LIKE '%itraconazole%' OR
        LOWER(drug) LIKE '%ketoconazole%' OR
        LOWER(drug) LIKE '%levoketoconazole%' OR
        LOWER(drug) LIKE '%lonafarnib%' OR
        LOWER(drug) LIKE '%lopinavir%' OR
        LOWER(drug) LIKE '%mifepristone%' OR
        LOWER(drug) LIKE '%nefazodone%' OR
        LOWER(drug) LIKE '%nelfinavir%' OR
		(LOWER(drug) LIKE '%nirmatrelvir%' AND LOWER(drug) LIKE '%ritonavir%') OR
        (LOWER(drug) LIKE '%ombitasvir%' AND LOWER(drug) LIKE '%paritaprevir%' AND LOWER(drug) LIKE '%ritonavir%') OR
        LOWER(drug) LIKE '%itraconazole%' OR
        LOWER(drug) LIKE '%posaconazole%' OR
        LOWER(drug) LIKE '%ritonavir%' OR
        LOWER(drug) LIKE '%saquinavir%' OR
        LOWER(drug) LIKE '%tucatinib%' OR
        LOWER(drug) LIKE '%voriconazole%' OR
        LOWER(drug) LIKE '%amiodarone%' OR
        LOWER(drug) LIKE '%aprepitant%' OR
        LOWER(drug) LIKE '%avacopan%' OR
        LOWER(drug) LIKE '%berotralstat%' OR
        LOWER(drug) LIKE '%cimetidine%' OR
        LOWER(drug) LIKE '%conivaptan%' OR
        LOWER(drug) LIKE '%crizotinib%' OR
        LOWER(drug) LIKE '%cyclosporine%' OR
        LOWER(drug) LIKE '%diltiazem%' OR
        LOWER(drug) LIKE '%duvelisib%' OR
        LOWER(drug) LIKE '%dronedarone%' OR
        LOWER(drug) LIKE '%erythromycin%' OR
        LOWER(drug) LIKE '%fedratinib%' OR
        LOWER(drug) LIKE '%fluconazole%' OR
        LOWER(drug) LIKE '%fosamprenavir%' OR
        LOWER(drug) LIKE '%fosaprepitant%' OR
		(LOWER(drug) LIKE '%fosnetupitant%' AND LOWER(drug) LIKE '%palonosetron%') OR
        LOWER(drug) LIKE '%imatinib%' OR
		LOWER(drug) LIKE '%isavuconazole%' OR
        (LOWER(drug) LIKE '%isavuconazonium%' AND LOWER(drug) LIKE '%sulfate%') OR
        LOWER(drug) LIKE '%lefamulin%' OR
        LOWER(drug) LIKE '%letermovir%' OR
        LOWER(drug) LIKE '%netupitant%' OR
        LOWER(drug) LIKE '%nilotinib%' OR
        LOWER(drug) LIKE '%nirogecestat%' OR
        LOWER(drug) LIKE '%ribociclib%' OR
        LOWER(drug) LIKE '%schisandra%' OR
        LOWER(drug) LIKE '%verapamil%' OR
        LOWER(drug) LIKE '%cyclosporine%' OR
        LOWER(drug) LIKE '%diltiazem%' OR
        LOWER(drug) LIKE '%fluconazole%' 
    );

SELECT *
FROM drugs_3a4_inhibitors