import importlib
from flab_cohorts.extractors.base import BaseExtractor
from flab_cohorts.extractors.LIT import REGISTRY
from flab_cohorts.utils.logger import get_logger

logger = get_logger("LIT_EXTRACTOR")


class LITExtractor(BaseExtractor):
    def __init__(self, args):
        super().__init__(args)

    def extract_full_cohort(self, cohort: str):
        if cohort == "all":
            results = {}
            for key in REGISTRY:
                logger.info("Extracting cohort: %s", key)
                results[key] = self.extract_full_cohort(key)
                logger.info("Done: %s", key)
            return results

        if cohort not in REGISTRY:
            raise ValueError(
                f"Unknown LIT cohort: {cohort!r}. "
                f"Available: {sorted(REGISTRY)}"
            )

        module_path, class_name = REGISTRY[cohort]
        module = importlib.import_module(module_path)
        extractor_cls = getattr(module, class_name)
        return extractor_cls(self.args).extract_cohort()
