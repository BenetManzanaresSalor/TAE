from abc import ABC, abstractmethod
from typing import Dict, List
from tqdm.autonotebook import tqdm

from ..utils import Document, MaskedDocument

class MetricABC(ABC):

    def __init__(self):
        super().__init__()    
    
    def evaluate(self, anonymizations:Dict[str, List[MaskedDocument]], documents:Dict[str,Document], **kwargs):
        results = {}

        with tqdm(anonymizations.items(), desc="Processing each anonymization") as pbar:
            for anon_name, masked_docs in pbar:
                pbar.set_description(f"Processing {anon_name} anonymization")
                output = self._anonymization_eval(masked_docs, documents, **kwargs)
                results[anon_name] = output[0] if isinstance(output, tuple) else output  # If tuple, the first is metric's value
        
        return results

    @abstractmethod
    def _anonymization_eval(self, masked_docs:List[MaskedDocument], documents:Dict[str,Document], **kwargs):
        """Computes the metric for a listed of masked documents, corresponding to a particular anonymization.
        When implementing the metric, override this method."""
        pass