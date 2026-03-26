import logging
from typing import Dict, List

from ..utils import Document, MaskedDocument
from .recall import Recall, RECALL_INCLUDE_DIRECT, RECALL_INCLUDE_QUASI, RECALL_TOKEN_LEVEL

class RecallPerEntityType(Recall):
    def _anonymization_eval(self, masked_docs:List[MaskedDocument], documents:Dict[str,Document], include_direct:bool=RECALL_INCLUDE_DIRECT, 
                                   include_quasi:bool=RECALL_INCLUDE_QUASI, token_level:bool=RECALL_TOKEN_LEVEL) -> Dict[str,float]:
        """
        It computes recall factored by the `entity_type` in the **manual annotations**, enabling a fine-grained analysis.
        TAE's implementation follows the version proposed in [Pilán et al., The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization, Computational Linguistics, 2022](https://aclanthology.org/2022.cl-4.19/),
        which allows for multi-annotated documents (performing a micro-average over annotators),
        token-level and mention-level assessment and independent consideration of direct and quasi identifiers.
        Args:
            masked_docs (List[MaskedDocument]): Documents together with spans masked by the system.
            include_direct (bool): Whether to include direct identifiers in the metric.
            include_quasi (bool): Whether to include quasi identifiers in the metric.
            token_level (bool): Whether to compute the recall at the level of tokens or mentions.

        Returns:
            dict: A dictionary where keys are entity types and values are their corresponding recall scores.
        """
                
        nb_masked_by_type, nb_by_type = self._get_mask_counts(masked_docs, documents,
                                                              include_direct, include_quasi, token_level)
        
        return {ent_type:nb_masked_by_type[ent_type]/nb_by_type[ent_type]
                for ent_type in nb_by_type}
                