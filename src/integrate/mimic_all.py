"""Extract all MIMIC sources into state space."""
def extract_patient(subject_id, hosp_path, icu_path):
    return {"labs": {}, "vitals": {}, "demographics": {}, "medications": []}
