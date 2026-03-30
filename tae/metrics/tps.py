from typing import Dict, List, Optional, Tuple
import gc

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..utils import DEVICE, IC_WEIGHTING_MAX_SEGMENT_LENGTH, IC_WEIGHTING_MODEL_NAME, Document, ICTokenWeighting, MaskedDocument, UniformTokenWeighting
from .tpi import TPI

# TPS default settings
TPS_TERM_ALTERNING = 6
TPS_USE_CHUNKING = True
TPS_SIMILARITY_MODEL_NAME = "paraphrase-albert-base-v2" # From the Sentence Transformers library (https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) or others such as "bert-base-cased"

class TPS(TPI):
    def _evaluate_anonymization(self, masked_docs:List[MaskedDocument], documents:Dict[str,Document],
                            weighting_model_name:Optional[str]=IC_WEIGHTING_MODEL_NAME,
                            weighting_max_segment_length:int=IC_WEIGHTING_MAX_SEGMENT_LENGTH, term_alterning=TPS_TERM_ALTERNING,
                            similarity_model_name:str=TPS_SIMILARITY_MODEL_NAME, use_chunking:bool=TPS_USE_CHUNKING,
                            verbose:bool=True) -> Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
        """
        **Text Preserved Similarity (TPS)** measures the percentage of information content (IC) still present in the masked documents,
        weighted by the similarity between replacement and original terms.
        This metric is used to assess utility preservation for replacement-based masking (i.e., text sanitization).
        It employs `ICTokenWeighting` for measuring IC and a specified similarity model for replacement similarity.
        This metric was proposed in [Pilán et al., Truthful Text Sanitization Guided by Inference Attacks, Submitted, 2024](https://arxiv.org/abs/2412.12928).
        TPS can be seen as a replacement-compatible version of [TPI](#tpi) (detailed above), pondering it with replacements' similarity.

        Args:
            masked_docs (List[MaskedDocument]): Documents together with spans masked by the anonymization method.
            weighting_model_name (Optional[str]): Name of the model to be used for IC weighting, implemented in the `ICTokenWeighting` class. 
                If `None`, uniform weighting (same weights for all) is used. 
                The name must be a valid [HuggingFace's model](https://huggingface.co/models) name, such as ["google-bert/bert-base-uncased"](https://huggingface.co/google-bert/bert-base-uncased).
            weighting_max_segment_length (int): Maximum segment length for `ICTokenWeighting`. Texts with more tokens than this will be splitted for IC computation.
            term_alterning (Union[int,str]): Parameter for term alternation in the multi-round IC calculation.
                It can be an integer (e.g., N = 6) or the string "sentence" 
                When using an integer N, one of each N terms will be masked each round.
                A larger N value implies a more accurate IC estimation (up to a certain point), but slower computation because more rounds are required.
                If "sentence" is used, the text will be split into sentences, and one of the sentence terms will be masked at each round.
                This approach is significantly slower but may provide the most accurate IC estimation.
            similarity_model_name (str): Name of the embedding model for calculating replacement similarity.
                It must be compatible with the [Sentence Transformers library](https://www.sbert.net/), such as ["paraphrase-albert-base-v2"](https://huggingface.co/sentence-transformers/paraphrase-albert-base-v2).
            use_chunking (bool): Whether to use chunking for term span extraction. It is recommended for a more precise IC calculation.

        Returns:
            Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
                - float: The average TPS for the corpus.
                - np.ndarray: An array of TPS values for each document.
                - Dict[str,np.ndarray]: A dictionary containing precomputed ICs (used for caching).
                - np.ndarray: An array of similarities for replacements.
        """
        
        # Initialize outputs
        tps_array = np.empty(len(masked_docs))
        if self.ics_dict is None:
            self.ics_dict = {} # Used to avoid recomputing, for each anonymization, the original document's ICs (which are always identical)
        similarity_array = []

        # Define token weighting
        if weighting_model_name is None:
            token_weighting = UniformTokenWeighting()
        
        else:
            token_weighting = ICTokenWeighting(model_name=weighting_model_name, device=DEVICE,
                                               max_segment_length=weighting_max_segment_length)
        
        # Load embedding model and function for similarity
        embedding_func, embedding_model = self._get_embedding_func(similarity_model_name)
        
        # Process each masked document
        for idx, masked_doc in enumerate(masked_docs):
            doc = documents[masked_doc.doc_id]

            # Get text spans
            spans = self._get_terms_spans(doc.spacy_doc, use_chunking=use_chunking)

            # Get IC for all spans
            if masked_doc.doc_id in self.ics_dict:
                spans_IC = self.ics_dict[masked_doc.doc_id] # Use precomputed ICs
            else:
                spans_IC = self._get_ics(spans, doc, term_alterning, token_weighting)
                self.ics_dict[masked_doc.doc_id] = spans_IC # Store ICs (useful as cache)

            # Get replacements, corresponding masked texts and corresponding spans indexes
            repl_out = self._get_replacements_info(masked_doc, doc, spans)
            (replacements, masked_texts, spans_idxs_per_replacement) = repl_out

            # Measure similarities of replacements
            masked_spans = self._filter_masked_spans(doc, masked_doc)
            spans_mask = self._get_spans_mask(spans, masked_spans) # Non-masked=True(1), Masked=False(0)
            spans_sims = np.array(spans_mask, dtype=float) # Similarities for terms: Non-masked=1, Supressed=0, Replaced=[0,1]
            if len(replacements) > 0:
                texts_to_embed = masked_texts + replacements
                embeddings = embedding_func(texts_to_embed)
      
                masked_embedds = embeddings[:len(masked_texts)]
                repl_embedds = embeddings[len(masked_texts):]
                for masked_embed, repl_embed, spans_idxs in zip(masked_embedds, repl_embedds, spans_idxs_per_replacement):
                    similarity = self._cos_sim(masked_embed, repl_embed)
                    spans_sims[spans_idxs] = similarity
                    similarity_array.append(similarity)
                
                # Limit similarities to range [0,1]
                spans_sims[spans_sims < 0] = 0
                spans_sims[spans_sims > 1] = 1

            # Get TPS
            masked_TIC_sim = (spans_IC * spans_sims).sum()
            original_TIC = spans_IC.sum()
            tps_array[idx] = masked_TIC_sim / original_TIC
        
        # Dispose token weighting
        del token_weighting

        # Dispose embedding model
        if not embedding_model is None:
            del embedding_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()             

        # Get mean TPS
        tps = tps_array.mean()

        # All similarities to NumPy array
        similarity_array = np.array(similarity_array)

        return tps, tps_array, self.ics_dict, similarity_array
    
    def _get_embedding_func(self, sim_model_name:str) -> Tuple:
        embedding_model = SentenceTransformer(sim_model_name, trust_remote_code=True)
        embedding_func = lambda x: embedding_model.encode(x, show_progress_bar=False)
        
        return embedding_func, embedding_model
    
    def _get_replacements_info(self, masked_doc:MaskedDocument, doc:Document,
                               spans:List[Tuple[int, int]]) -> Tuple[List[str], List[str], List[List[int]]]:
        replacements = []
        masked_texts = []
        spans_idxs_per_replacement = []
        
        for replacement, (masked_span_start, masked_span_end) in zip(masked_doc.replacements, masked_doc.masked_spans):
            if not replacement is None: # If there is a replacement
                replacements.append(replacement)
                masked_texts.append(doc.text[masked_span_start:masked_span_end])
                replacement_spans_idxs = []
                for span_idx, (span_start, span_end) in enumerate(spans):
                    if span_start <= masked_span_start < span_end or span_start < masked_span_end <= span_end:
                        replacement_spans_idxs.append(span_idx)
                    elif span_start > masked_span_end:  # Break if candidate span starts too late
                        break
                spans_idxs_per_replacement.append(replacement_spans_idxs)
        
        return replacements, masked_texts, spans_idxs_per_replacement
    
    def _cos_sim(self, a:np.ndarray, b:np.ndarray) -> float:
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        sim = dot_product / (magnitude_a * magnitude_b)
        if np.isnan(sim):
            sim = 0
        return sim
