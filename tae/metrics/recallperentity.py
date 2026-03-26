import logging
from typing import Dict, List

from .metric_abc import MetricABC
from ..utils import Document, MaskedDocument

class RecallPerEntity(MetricABC):
    def _anonymization_eval(self, masked_docs:List[MaskedDocument], documents:Dict[str,Document], **kwargs):
        logging.error("Metric not implemented")
        return -1