-- Create a race mapping table to standarize races from hosp.admissions
CREATE TABLE race_mapping (
    original_race VARCHAR(255),
    standardized_race VARCHAR(255)
);

INSERT INTO race_mapping (original_race, standardized_race) VALUES
('WHITE', 'White'),
('WHITE - BRAZILIAN', 'Hispanic/Latino'),
('WHITE - OTHER EUROPEAN', 'White'),
('WHITE - RUSSIAN', 'White'),
('WHITE - EASTERN EUROPEAN', 'White'),
('PORTUGUESE', 'White'),
('BLACK/AFRICAN AMERICAN', 'Black/African American'),
('BLACK/CARIBBEAN ISLAND', 'Black/African American'),
('BLACK/CAPE VERDEAN', 'Black/African American'),
('BLACK/AFRICAN', 'Black/African American'),
('HISPANIC/LATINO - PUERTO RICAN', 'Hispanic/Latino'),
('HISPANIC OR LATINO', 'Hispanic/Latino'),
('HISPANIC/LATINO - DOMINICAN', 'Hispanic/Latino'),
('HISPANIC/LATINO - MEXICAN', 'Hispanic/Latino'),
('HISPANIC/LATINO - SALVADORAN', 'Hispanic/Latino'),
('HISPANIC/LATINO - CENTRAL AMERICAN', 'Hispanic/Latino'),
('HISPANIC/LATINO - HONDURAN', 'Hispanic/Latino'),
('HISPANIC/LATINO - COLUMBIAN', 'Hispanic/Latino'),
('HISPANIC/LATINO - GUATEMALAN', 'Hispanic/Latino'),
('HISPANIC/LATINO - CUBAN', 'Hispanic/Latino'),
('SOUTH AMERICAN', 'Hispanic/Latino'),
('ASIAN - CHINESE', 'Asian'),
('ASIAN', 'Asian'),
('ASIAN - ASIAN INDIAN', 'Asian'),
('ASIAN - SOUTH EAST ASIAN', 'Asian'),
('ASIAN - KOREAN', 'Asian'),
('AMERICAN INDIAN/ALASKA NATIVE', 'Native American'),
('NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'Pacific Islander'),
('MULTIPLE RACE/ETHNICITY', 'Multiple'),
('OTHER', 'Other'),
('UNKNOWN', 'Other'),
('UNABLE TO OBTAIN', 'Other'),
('PATIENT DECLINED TO ANSWER', 'Other');

