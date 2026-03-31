"""
LIT (literature-based) cohort extractors.

REGISTRY maps short cohort keys → (module_path, class_name) for lazy
import by LITExtractor.  To add a new cohort, create the extractor
module under this package and add one entry here.
"""

REGISTRY = {
    "aplasia": (
        "flab_cohorts.extractors.LIT.aplasia",
        "AplasiaExtractor",
    ),
    "neutropenic_fever": (
        "flab_cohorts.extractors.LIT.neutropenic_fever",
        "NeutropenicFeverExtractor",
    ),
    "acute_kidney_injury": (
        "flab_cohorts.extractors.LIT.acute_kidney_injury",
        "AcuteKidneyInjuryExtractor",
    ),
    "gi_bleeding": (
        "flab_cohorts.extractors.LIT.gastrointestinal_bleeding",
        "GastrointestinalBleedingExtractor",
    ),
    "ulcer": (
        "flab_cohorts.extractors.LIT.ulcer",
        "PressureUlcerExtractor",
    ),
    "ac_mortality": (
        "flab_cohorts.extractors.LIT.alcoholic_cirrhosis_mortality",
        "AlcoholicCirrhosisExtractor",
    ),
    "hf_and_af_mortality": (
        "flab_cohorts.extractors.LIT.heart_failure_atrial_fibrillation_mortality",
        "HFAndAFExtractor",
    ),
    "bt_mortality": (
        "flab_cohorts.extractors.LIT.bone_tumor_mortality",
        "BoneTumorExtractor",
    ),
    "immune_mortality": (
        "flab_cohorts.extractors.LIT.immunocompromised_mortality",
        "ImmunocompromisedExtractor",
    ),
    "lc_mortality": (
        "flab_cohorts.extractors.LIT.liver_cirrhosis_mortality",
        "LiverCirrhosisExtractor",
    ),
    "mi_mortality": (
        "flab_cohorts.extractors.LIT.myocardial_infarction_mortality",
        "MyocardialInfarctionExtractor",
    ),
    "obesity_pneumonia": (
        "flab_cohorts.extractors.LIT.obesity_pneumonia_mortality",
        "ObesityPneumoniaExtractor",
    ),
    "pneumonia_mortality": (
        "flab_cohorts.extractors.LIT.pneumonia_mortality",
        "PneumoniaExtractor",
    ),
    "prostate_cancer": (
        "flab_cohorts.extractors.LIT.prostate_cancer",
        "ProstateCancerExtractor",
    ),
    "ut_infection_mortality": (
        "flab_cohorts.extractors.LIT.urinary_tract_infection_mortality",
        "UrinaryTractInfectionExtractor",
    ),
    "af_mortality": (
        "flab_cohorts.extractors.LIT.atrial_fibrillation_mortality",
        "AtrialFibrillationExtractor",
    ),
}
