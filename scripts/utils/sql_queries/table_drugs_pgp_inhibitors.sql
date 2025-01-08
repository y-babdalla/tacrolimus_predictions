-- Save pgp_inhibitors
CREATE TABLE drugs_pgp_inhibitors AS
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
        LOWER(drug) LIKE '%abrocitinib%' OR
        LOWER(drug) LIKE '%adagrasib%' OR
        LOWER(drug) LIKE '%amiodarone%' OR
        LOWER(drug) LIKE '%azithromycin%' OR
        LOWER(drug) LIKE '%belumosudil%' OR
        LOWER(drug) LIKE '%cannabidiol%' OR
        LOWER(drug) LIKE '%capmatinib%' OR
        LOWER(drug) LIKE '%carvedilol%' OR
        LOWER(drug) LIKE '%clarithromycin%' OR
        LOWER(drug) LIKE '%cobicistat%' OR
        LOWER(drug) LIKE '%cyclosporine%' OR
        LOWER(drug) LIKE '%daclatasvir%' OR
        LOWER(drug) LIKE '%daridorexant%' OR
        LOWER(drug) LIKE '%diosmin%' OR
        LOWER(drug) LIKE '%dronedarone%' OR
        LOWER(drug) LIKE '%elagolix%' OR
		(LOWER(drug) LIKE '%elagolix%' AND LOWER(drug) LIKE '%estradiol%' AND LOWER(drug) LIKE '%norethindrone%') OR
        LOWER(drug) LIKE '%eliglustat%' OR
		(LOWER(drug) LIKE '%elexacaftor%' AND LOWER(drug) LIKE '%tezacaftor%' AND LOWER(drug) LIKE '%ivacaftor%') OR
        LOWER(drug) LIKE '%enzalutamide%' OR
		(LOWER(drug) LIKE '%erythromycin%' AND LOWER(drug) LIKE '%ethylsuccinate%') OR
        LOWER(drug) LIKE '%flibanserin%' OR
        LOWER(drug) LIKE '%fostamatinib%' OR
		(LOWER(drug) LIKE '%glecaprevir%' AND LOWER(drug) LIKE '%pibrentasvir%') OR
        LOWER(drug) LIKE '%isavuconazole%' OR
        LOWER(drug) LIKE '%itraconazole%' OR
        LOWER(drug) LIKE '%ivacaftor%' OR
        LOWER(drug) LIKE '%ketoconazole%' OR
        LOWER(drug) LIKE '%lapatinib%' OR
        LOWER(drug) LIKE '%ledipasvir%' OR
        LOWER(drug) LIKE '%levoketoconazole%' OR
        LOWER(drug) LIKE '%mavorixafor%' OR
        LOWER(drug) LIKE '%mifepristone%' OR
        LOWER(drug) LIKE '%neratinib%' OR
		(LOWER(drug) LIKE '%nirmatrelvir%' AND LOWER(drug) LIKE '%ritonavir%') OR
		(LOWER(drug) LIKE '%ombitasvir%' AND LOWER(drug) LIKE '%paritaprevir%' AND LOWER(drug) LIKE 'ritonavir') OR
        LOWER(drug) LIKE '%technivie%' OR
        LOWER(drug) LIKE '%osimertinib%' OR
        LOWER(drug) LIKE '%pirtobrutinib%' OR
        LOWER(drug) LIKE '%posaconazole%' OR
        LOWER(drug) LIKE '%propafenone%' OR
        LOWER(drug) LIKE '%quinidine%' OR
        LOWER(drug) LIKE '%quinine%' OR
        LOWER(drug) LIKE '%ranolazine%' OR
        LOWER(drug) LIKE '%ritonavir%' OR
        LOWER(drug) LIKE '%rolapitant%' OR
        LOWER(drug) LIKE '%selpercatinib%' OR
        LOWER(drug) LIKE '%simeprevir%' OR
        LOWER(drug) LIKE '%sotagliflozin%' OR
        LOWER(drug) LIKE '%sotorasib%' OR
        LOWER(drug) LIKE '%tamoxifen%' OR
        LOWER(drug) LIKE '%tepotinib%' OR
		(LOWER(drug) LIKE '%tezacaftor%' AND LOWER(drug) LIKE '%ivacaftor%') OR
        LOWER(drug) LIKE '%ticagrelor%' OR
        LOWER(drug) LIKE '%tucatinib%' OR
        LOWER(drug) LIKE '%velpatasvir%' OR
        LOWER(drug) LIKE '%vemurafenib%' OR
        LOWER(drug) LIKE '%verapamil%' OR
        LOWER(drug) LIKE '%voclosporin%' OR
        LOWER(drug) LIKE '%cyclosporine%' OR
		LOWER(drug) LIKE '%sandimmune%' 
    );

SELECT *
FROM drugs_pgp_inhibitors