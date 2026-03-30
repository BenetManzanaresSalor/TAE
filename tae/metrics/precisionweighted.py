import logging
from typing import Dict, List, Optional

from ..utils import IC_WEIGHTING_MODEL_NAME, Document, MaskedDocument, IC_WEIGHTING_MAX_SEGMENT_LENGTH
from .precision import Precision, PRECISION_TOKEN_LEVEL

class PrecisionWeighted(Precision):
    def _evaluate_anonymization(self, masked_docs:List[MaskedDocument],
                            documents:Dict[str,Document],
                            weighting_model_name:Optional[str]=IC_WEIGHTING_MODEL_NAME,
                            weighting_max_segment_length:int=IC_WEIGHTING_MAX_SEGMENT_LENGTH,
                            token_level:bool=PRECISION_TOKEN_LEVEL,
                            verbose:bool=True) -> float:
        """
        Precision but employing IC weighting by default.
        It is implemented as a wrapper of `get_precision`, so the arguments are exactly the same.
        The only difference is that `weighting_model_name` defaults to ["google-bert/bert-base-uncased"](https://huggingface.co/google-bert/bert-base-uncased).
        This avoids the need to select the `weighting_model_name` for IC weighting.

        Args:
            masked_docs (List[MaskedDocument]): Documents together with spans masked by the anonymization method.
            weighting_model_name (Optional[str]): Name of the model to be used for IC weighting, implemented in the `ICTokenWeighting` class.
                Defaults to `IC_WEIGHTING_MODEL_NAME`.
                If `None`, uniform weighting (same weights for all) is used. 
                The name must be a valid [HuggingFace's model](https://huggingface.co/models), such as ["google-bert/bert-base-uncased"](https://huggingface.co/google-bert/bert-base-uncased).
            weighting_max_segment_length (int): Maximum segment length for `ICTokenWeighting`. Texts with more tokens than this will be splitted for IC computation.
            token_level (bool): If set to `True`, the precision is computed at the level of tokens, otherwise the precision is at the mention-level.
                The latter implies that the whole human-annotated mention (rather than some tokens) needs to be masked for being considered a true positive.
            verbose (bool): Whether to print verbose output during execution.
        """
        return super()._evaluate_anonymization(masked_docs, documents, weighting_model_name=weighting_model_name,
                      weighting_max_segment_length=weighting_max_segment_length,
                      token_level=token_level, verbose=verbose)
