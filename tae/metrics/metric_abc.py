from abc import ABC, abstractmethod
from typing import Dict, List
from tqdm.autonotebook import tqdm

from ..utils import DOC_ID_KEY, ORIGINAL_TEXT_KEY, Document, MaskedDocument

class MetricABC(ABC):
    """Abstract base class for implementing evaluation metrics for anonymization methods.

    Subclasses should implement the `_evaluate_anonymization` method to define
    how the metric is computed for a given set of masked documents.
    Optionally, if the metric deals with all the anonymizations simultaneously (as TRIR and NMI), the `evaluate` method should be overriden instead.
    """
    
    #region Evaluation
    
    def evaluate(self, anonymizations:Dict[str, List[MaskedDocument]], documents:Dict[str,Document], **kwargs) -> Dict[str, float]:
        """Evaluate the performance of each anonymization method on the provided documents.

        Iterates over each anonymization method, computes its metric using `_evaluate_anonymization`,
        and returns a dictionary of results. 

        If the inheriting metric deals with all the anonymizations simultaneously (as TRIR and NMI), override this method.

        Args:
            anonymizations (Dict[str, List[MaskedDocument]]): A dictionary mapping anonymization method names to lists of `MaskedDocument`.
            documents (Dict[str,Document]): A dictionary mapping document IDs to their original `Document` objects.
            **kwargs: Additional keyword arguments for the metric.

        Returns:
            A dictionary mapping anonymization method names to their respective evaluation result.
        """
        results = {}

        with tqdm(anonymizations.items(), desc="Processing each anonymization") as pbar:
            for anon_name, masked_docs in pbar:
                pbar.set_description(f"Processing {anon_name} anonymization")
                output = self._evaluate_anonymization(masked_docs, documents, **kwargs)
                results[anon_name] = output[0] if isinstance(output, tuple) else output  # If tuple, the first is metric's value
        
        return results

    @abstractmethod
    def _evaluate_anonymization(self, masked_docs:List[MaskedDocument], documents:Dict[str,Document], **kwargs) -> float:
        """Compute the evaluation metric for a list of masked documents corresponding to a specific anonymization method.

        This method must be overridden in subclasses to define the actual metric computation.

        Args:
            masked_docs (List[MaskedDocument]): A list of `MaskedDocument` for a specific anonymization method.
            documents (Dict[str,Document]): A dictionary mapping document IDs to their original `Document` objects.
            **kwargs: Additional keyword arguments for the metric.

        Returns:
            The computed evaluation result for the anonymization's masked documents.
        """
        pass

    #endregion

    #region Auxiliar
    
    def _get_anonymization_corpora(self, anonymizations:Dict[str, List[MaskedDocument]],
                                   documents:Dict[str,Document],
                                   include_original_text:bool=False) -> Dict[str, Dict[str,str]]:
        """Generate a structured corpus of anonymized documents for comparison and evaluation.
        Useful for metrics dealing with all the anonymizations simultaneously (as TRIR and NMI).

        Transforms the input anonymizations and documents into a dictionary format,
        where each document's anonymized versions (and optionally the original text) are easily accessible.

        Args:
            anonymizations: A dictionary mapping anonymization method names to lists of masked documents.
            documents: A dictionary mapping document IDs to their original document objects.
            include_original_text: If True, include the original text of each document in the output.

        Returns:
            A dictionary where each key is a document ID, and the value is another dictionary.
            The inner dictionary contains the document ID, optionally the original text, and the anonymized text
            for each anonymization method.
        """
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