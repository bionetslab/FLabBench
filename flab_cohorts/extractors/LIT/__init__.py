"""
LIT (literature-based) cohort extractors.
REGISTRY maps cohort key → (module_path, class_name).
Keys must match the COHORT_NAME class attribute on each extractor.
"""

_LIT = "flab_cohorts.extractors.LIT"

REGISTRY = {
    "aplasia":              (f"{_LIT}.aplasia",                                   "AplasiaExtractor"),
    "neutropenic_fever":    (f"{_LIT}.neutropenic_fever",                         "NeutropenicFeverExtractor"),
    "aki":                  (f"{_LIT}.acute_kidney_injury",                       "AcuteKidneyInjuryExtractor"),
    "gi_bleed":             (f"{_LIT}.gastrointestinal_bleeding",                 "GastrointestinalBleedingExtractor"),
    "pressure_ulcer":       (f"{_LIT}.ulcer",                                     "PressureUlcerExtractor"),
    "alc_cirrhosis":        (f"{_LIT}.alcoholic_cirrhosis_mortality",             "AlcoholicCirrhosisExtractor"),
    "hf_af":                (f"{_LIT}.heart_failure_atrial_fibrillation_mortality","HFAndAFExtractor"),
    "bone_tumor":           (f"{_LIT}.bone_tumor_mortality",                      "BoneTumorExtractor"),
    "immunocompromised":    (f"{_LIT}.immunocompromised_mortality",               "ImmunocompromisedExtractor"),
    "liver_cirrhosis":      (f"{_LIT}.liver_cirrhosis_mortality",                 "LiverCirrhosisExtractor"),
    "mi":                   (f"{_LIT}.myocardial_infarction_mortality",           "MyocardialInfarctionExtractor"),
    "obesity_pneumonia":    (f"{_LIT}.obesity_pneumonia_mortality",               "ObesityPneumoniaExtractor"),
    "pneumonia":            (f"{_LIT}.pneumonia_mortality",                       "PneumoniaExtractor"),
    "prostate_cancer":      (f"{_LIT}.prostate_cancer",                           "ProstateCancerExtractor"),
    "uti":                  (f"{_LIT}.urinary_tract_infection_mortality",          "UrinaryTractInfectionExtractor"),
    "af":                   (f"{_LIT}.atrial_fibrillation_mortality",             "AtrialFibrillationExtractor"),
}
