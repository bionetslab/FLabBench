from flab_cohorts.cohort_extractor.lit.neutropenic_fever import NeutropenicFeverExtractor

COHORT_REGISTRY = {
    "neutropenic_fever": NeutropenicFeverExtractor,
    "NF": NeutropenicFeverExtractor,
}


def get_extractor(cohort: str, args):
    cls = COHORT_REGISTRY.get(cohort)
    if cls is None:
        raise ValueError(f"Unknown LIT cohort: {cohort}. Available: {list(COHORT_REGISTRY.keys())}")
    return cls(args)
