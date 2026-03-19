from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.extractors.LIT.aplasia import AplasiaExtractor
from flab_cohorts.extractors.LIT.neutropenic_fever import NeutropenicFeverExtractor
from flab_cohorts.extractors.LIT.acute_kidney_injury import AcuteKidneyInjuryExtractor
from flab_cohorts.extractors.LIT.gastrointestinal_bleeding import GastrointestinalBleedingExtractor
from flab_cohorts.extractors.LIT.ulcer import PressureUlcerExtractor



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