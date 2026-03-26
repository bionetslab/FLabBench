from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.extractors.LIT.aplasia import AplasiaExtractor
from flab_cohorts.extractors.LIT.neutropenic_fever import NeutropenicFeverExtractor
from flab_cohorts.extractors.LIT.acute_kidney_injury import AcuteKidneyInjuryExtractor
from flab_cohorts.extractors.LIT.gastrointestinal_bleeding import GastrointestinalBleedingExtractor
from flab_cohorts.extractors.LIT.ulcer import PressureUlcerExtractor
from flab_cohorts.extractors.LIT.alcoholic_cirrhosis_mortality import AlcoholicCirrhosisExtractor
from flab_cohorts.extractors.LIT.atrial_fibrillation_mortality import AtrialFibrillationExtractor
from flab_cohorts.extractors.LIT.bone_tumor_mortality import BoneTumorExtractor
from flab_cohorts.extractors.LIT.immunocompromised_mortality import ImmunocompromisedExtractor
from flab_cohorts.extractors.LIT.liver_cirrhosis_mortality import LiverCirrhosisExtractor
from flab_cohorts.extractors.LIT.myocardial_infarction_mortality import MyocardialInfarctionExtractor
from flab_cohorts.extractors.LIT.obesity_pneumonia_mortality import ObesityPneumoniaExtractor
from flab_cohorts.extractors.LIT.pneumonia_mortality import PneumoniaExtractor
from flab_cohorts.extractors.LIT.prostate_cancer import ProstateCancerExtractor

#TODO: Make cohort names consistent

class LITExtractor(BaseExtractor):
    def __init__(self, args):
        super().__init__(args)
        
    def extract_full_cohort(self, cohort: str):
        
        if cohort == "aplasia":
            return AplasiaExtractor(self.args).extract_cohort()
        
        
        elif cohort == "neutropenic_fever":
            return NeutropenicFeverExtractor(self.args).extract_cohort()
        
        
        elif cohort == "acute_kidney_injury":
            return AcuteKidneyInjuryExtractor(self.args).extract_cohort()

        elif cohort == "gi_bleeding":
            return GastrointestinalBleedingExtractor(self.args).extract_cohort()

        elif cohort == "ulcer":
            return PressureUlcerExtractor(self.args).extract_cohort()
        
        elif cohort == "ac_mortality":
            return AlcoholicCirrhosisExtractor(self.args).extract_cohort()
        
        elif cohort == "af_mortality":
            return AtrialFibrillationExtractor(self.args).extract_cohort()
        
        elif cohort == "bt_mortality":
            return BoneTumorExtractor(self.args).extract_cohort()

        elif cohort == "immune_mortality":
            return ImmunocompromisedExtractor(self.args).extract_cohort()
        
        elif cohort == "lc_mortality":
            return LiverCirrhosisExtractor(self.args).extract_cohort()
        
        elif cohort == "mi_mortality":
            return MyocardialInfarctionExtractor(self.args).extract_cohort()
        
        elif cohort == "obesity_pneumonia":
            return ObesityPneumoniaExtractor(self.args).extract_cohort()

        elif cohort == "pneumonia_mortality":
            return PneumoniaExtractor(self.args).extract_cohort()
        
        elif cohort == "prostate_cancer":
            return ProstateCancerExtractor(self.args).extract_cohort()