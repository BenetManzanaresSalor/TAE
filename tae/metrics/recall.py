import logging
from typing import Dict, List, Tuple

from ..utils import Document, MaskedDocument
from .metric_abc import MetricABC

# Recall default settings
RECALL_INCLUDE_DIRECT=True
RECALL_INCLUDE_QUASI=True
RECALL_TOKEN_LEVEL=True

class Recall(MetricABC):
    def _anonymization_eval(self, masked_docs:List[MaskedDocument], documents:Dict[str,Document], include_direct:bool=RECALL_INCLUDE_DIRECT, 
                    include_quasi:bool=RECALL_INCLUDE_QUASI, token_level:bool=RECALL_TOKEN_LEVEL,
                    verbose:bool=True) -> float:
        """
        Standard privacy proxy for text anonymization.
        It measures the percentage of terms masked by the **manual annotations** that were also masked by the anonymizations.
        TAE's implementation follows the version proposed in [Pilán et al., The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization, Computational Linguistics, 2022](https://aclanthology.org/2022.cl-4.19/),
        which allows for multi-annotated documents (performing a micro-average over annotators), token-level and mention-level assessment and 
        independent consideration of direct and quasi identifiers.
        Args:
            masked_docs (List[MaskedDocument]): Documents together with spans masked by the anonymization method.
            include_direct (bool): Whether to consider direct identifiers in the metric computation.
            include_quasi (bool): Whether to include quasi identifiers in the metric computation.
            token_level (bool): If set to `True`, recall is computed at the level of tokens, otherwise it is at the mention-level.
                The latter implies that the whole human-annotated mention (rather than some tokens) needs to be masked for being considered a true positive.
            verbose (bool): Whether to print verbose output during execution.

        Returns:
            recall (float): The recall score.
        """

        nb_masked_by_type, nb_by_type = self._get_mask_counts(masked_docs, documents,
                                                              include_direct, include_quasi, token_level)
        
        nb_masked_elements = sum(nb_masked_by_type.values())
        nb_elements = sum(nb_by_type.values())
        
        if nb_elements != 0:
            recall = nb_masked_elements / nb_elements
        else:
            recall = 0
            if verbose: logging.warning("Zero annotated identifiers, resulting in a recall of zero")
        
        return recall

    def _get_mask_counts(self, masked_docs:List[MaskedDocument],
                        documents:Dict[str,Document],
                        include_direct:bool=RECALL_INCLUDE_DIRECT,                        
                        include_quasi:bool=RECALL_INCLUDE_QUASI,
                        token_level:bool=RECALL_TOKEN_LEVEL) -> Tuple[Dict[str,int],Dict[str,int]]:
        nb_masked_elements_by_type = {}
        nb_elements_by_type = {}
        
        for doc in masked_docs:            
            gold_doc = documents[doc.doc_id]           
            for entity in gold_doc.get_entities_to_mask(include_direct, include_quasi):
                
                if entity.entity_type not in nb_elements_by_type:
                    nb_elements_by_type[entity.entity_type] = 0
                    nb_masked_elements_by_type[entity.entity_type] = 0
                
                spans = list(entity.mentions)
                if token_level:
                    spans = [(start, end) for mention_start, mention_end in spans
                             for start, end in gold_doc.split_by_tokens(mention_start, mention_end)]
                
                for start, end in spans:
                    if gold_doc.is_mention_masked(doc, start, end):
                        nb_masked_elements_by_type[entity.entity_type] += 1
                    nb_elements_by_type[entity.entity_type] += 1
        
        return nb_masked_elements_by_type, nb_elements_by_type