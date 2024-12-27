from lm_saes.config import ActivationPipelineConfig


class ActivationPipeline:
    def __init__(self, cfg: ActivationPipelineConfig):
        self.cfg = cfg

    @staticmethod
    def build_processors(cfg: ActivationPipelineConfig):
        processors = []
        return processors

    def process(self):
        pass
