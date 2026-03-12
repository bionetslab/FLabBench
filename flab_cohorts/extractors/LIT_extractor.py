from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.extractors.LIT.aplasia import AplasiaExtractor
from flab_cohorts.extractors.LIT.neutropenic_fever import NeutropenicFeverExtractor


class LITExtractor(BaseExtractor):
    def __init__(self, args):
        super().__init__(args)
        
    def extract_full_cohort(self, cohort: str):
        
        if cohort == "aplasia":
            return AplasiaExtractor(self.args).extract_cohort()
        
        
        elif cohort == "neutropenic_fever":
            return NeutropenicFeverExtractor(self.args).extract_cohort()



