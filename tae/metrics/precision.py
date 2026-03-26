import logging
from typing import Dict, List, Optional

from ..utils import Document, ICTokenWeighting, MaskedDocument, UniformTokenWeighting, DEVICE, IC_WEIGHTING_MAX_SEGMENT_LENGTH

from .metric_abc import MetricABC

PRECISION_TOKEN_LEVEL = True

class Precision(MetricABC):        
        def _anonymization_eval(self, masked_docs:List[MaskedDocument],
                                documents:Dict[str,Document],
                                weighting_model_name:Optional[str]=None,
                                weighting_max_segment_length:int=IC_WEIGHTING_MAX_SEGMENT_LENGTH,
                                token_level:bool=PRECISION_TOKEN_LEVEL,
                                verbose:bool=True) -> float:
            """
            Standard proxy of utility for text anonymization.
            It measures the percentage of terms masked by the anonymizations that were also masked by the **manual annotations**.
            TAE's implementation follows the version proposed in [Pilán et al., The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization, Computational Linguistics, 2022](https://aclanthology.org/2022.cl-4.19/),
            which allows for multi-annotated documents (performing a micro-average over annotators),
            token-level and mention-level assessment and weighting based on information content (IC).

            Args:
                masked_docs (List[MaskedDocument]): Documents together with spans masked by the anonymization method.
                weighting_model_name (Optional[str]): Name of the model to be used for IC weighting, implemented in the `ICTokenWeighting` class. 
                    If `None`, uniform weighting (same weights for all) is used. 
                    The name must be a valid [HuggingFace's model](https://huggingface.co/models) name, such as ["google-bert/bert-base-uncased"](https://huggingface.co/google-bert/bert-base-uncased).
                weighting_max_segment_length (int): Maximum segment length for `ICTokenWeighting`. Texts with more tokens than this will be splitted for IC computation.
                token_level (bool): If set to `True`, the precision is computed at the level of tokens, otherwise it is at the mention-level.
                    The latter implies that the whole human-annotated mention (rather than some tokens) needs to be masked for being considered a true positive.
                verbose (bool): Whether to print verbose output during execution.

            Returns:
                float: The precision score.
            """
            
            weighted_true_positives = 0.0
            weighted_system_masks = 0.0

            # Define token weighting
            if weighting_model_name is None:
                token_weighting = UniformTokenWeighting()        
            else:
                token_weighting = ICTokenWeighting(model_name=weighting_model_name, device=DEVICE,
                                                max_segment_length=weighting_max_segment_length)
            
            # For each masked document
            for doc in masked_docs:
                gold_doc:Document = documents[doc.doc_id]
                
                # We extract the list of spans (token- or mention-level)
                anonymization_masks = []
                for start, end in doc.masked_spans:
                    if token_level:
                        anonymization_masks += list(gold_doc.split_by_tokens(start, end))
                    else:
                        anonymization_masks += [(start,end)]
                
                # We compute the weights (information content) of each mask
                weights = token_weighting.get_weights(gold_doc.text, anonymization_masks)
                
                # We store the number of annotators in the gold standard document
                nb_annotators = len(set(entity.annotator for entity in gold_doc.gold_annotated_entities.values()))
                
                for (start, end), weight in zip(anonymization_masks, weights):
                    
                    # We extract the annotators that have also masked this token/span
                    annotators = gold_doc.get_annotators_for_span(start, end)
                    
                    # And update the (weighted) counts
                    weighted_true_positives += (len(annotators) * weight)
                    weighted_system_masks += (nb_annotators * weight)
            
            # Dispose token weighting
            del token_weighting

            # Return results
            if weighted_system_masks != 0:
                precision = weighted_true_positives / weighted_system_masks
            else:
                precision = 0
                if verbose: logging.warning("There are no masked spans, resulting in a precision of zero")
            
            return precision
