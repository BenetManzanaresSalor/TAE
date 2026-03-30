from abc import ABC, abstractmethod
from typing import Dict, List
from tqdm.autonotebook import tqdm

from ..utils import DOC_ID_KEY, ORIGINAL_TEXT_KEY, Document, MaskedDocument

class MetricABC(ABC):
    
    #region Evaluation
    
    def evaluate(self, anonymizations:Dict[str, List[MaskedDocument]], documents:Dict[str,Document], **kwargs) -> Dict[str, float]:
        results = {}

        with tqdm(anonymizations.items(), desc="Processing each anonymization") as pbar:
            for anon_name, masked_docs in pbar:
                pbar.set_description(f"Processing {anon_name} anonymization")
                output = self._evaluate_anonymization(masked_docs, documents, **kwargs)
                results[anon_name] = output[0] if isinstance(output, tuple) else output  # If tuple, the first is metric's value
        
        return results

    @abstractmethod
    def _evaluate_anonymization(self, masked_docs:List[MaskedDocument], documents:Dict[str,Document], **kwargs) -> float:
        """Computes the metric for a listed of masked documents, corresponding to a particular anonymization.
        When implementing the metric, override this method."""
        pass

    #endregion

    #region Auxiliar
    
    def _get_anonymization_corpora(self, anonymizations:Dict[str, List[MaskedDocument]],
                                   documents:Dict[str,Document],
                                   include_original_text:bool=False) -> Dict[str, Dict[str,str]]:
        corpora = {}
        
        # Transform list of masked docs into dictionaries for faster processing
        anon_dicts = {}
        for anon_name, masked_docs in anonymizations.items():
            anon_dicts[anon_name] = {masked_doc.doc_id:masked_doc for masked_doc in masked_docs}

        # Create a dictionary per document
        for doc_id, doc in documents.items():
            doc_dict = {DOC_ID_KEY:doc_id}
            if include_original_text:
                doc_dict[ORIGINAL_TEXT_KEY] = doc.text
            for anon_name, masked_docs_dict in anon_dicts.items():
                masked_doc = masked_docs_dict[doc_id]
                doc_dict[anon_name] = masked_doc.get_masked_text(doc.text)
            corpora[doc_id] = doc_dict

        return corpora

    #endregion