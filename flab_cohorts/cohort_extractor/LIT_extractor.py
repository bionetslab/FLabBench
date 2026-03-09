from flab_cohorts.cohort_extractor.base import BaseExtractor
from flab_cohorts.cohort_extractor.lit import get_extractor


class LITExtractor(BaseExtractor):
    def __init__(self, args):
        super().__init__(args)

    def extract_full_cohort(self, cohort: str):
        return get_extractor(cohort, self.args).extract_full_cohort()
