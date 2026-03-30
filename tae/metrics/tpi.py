from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import spacy

from ..utils import DEVICE, IC_WEIGHTING_MAX_SEGMENT_LENGTH, IC_WEIGHTING_MODEL_NAME, Document, ICTokenWeighting, MaskedDocument, TokenWeighting, UniformTokenWeighting
from .metric_abc import MetricABC

# TPI default settings
TPI_TERM_ALTERNING = 6
TPI_USE_CHUNKING = True

class TPI(MetricABC):
    ics_dict:Optional[Dict[str,np.ndarray]]=None

    def _evaluate_anonymization(self, masked_docs:List[MaskedDocument], documents:Dict[str,Document],
                            weighting_model_name:Optional[str]=IC_WEIGHTING_MODEL_NAME,
                            weighting_max_segment_length:int=IC_WEIGHTING_MAX_SEGMENT_LENGTH, 
                            term_alterning:Union[int,str]=TPI_TERM_ALTERNING,
                            use_chunking:bool=TPI_USE_CHUNKING) -> Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
        """
        **Text Preserved Information (TPI)** measures the percentage of information content (IC) still present in the masked documents.
        This metric is used to assess utility preservation.
        It was proposed in **Manzanares-Salor et al., Unsupervised utility evaluation of text anonymization methods via neural language models, Neural Networks, In Press, 2026**.
        The `ICTokenWeighting` is employed for measuring IC.
        TPI can be seen as an simplified/ablated version of Text Preserved Similarity (TPS), not taking into account replacements and their similarities.

        Args:
            masked_docs (List[MaskedDocument]): A list of `MaskedDocument` for a specific anonymization method.
            documents (Dict[str,Document]): A dictionary mapping document IDs to their original `Document` objects.
            weighting_model_name (Optional[str]): Name of the model to be used for IC weighting, implemented in the `ICTokenWeighting` class. 
                If `None`, uniform weighting (same weights for all) is used. 
                The name must be a valid [HuggingFace's model](https://huggingface.co/models), such as ["google-bert/bert-base-uncased"](https://huggingface.co/google-bert/bert-base-uncased).
            weighting_max_segment_length (int): Maximum segment length for `ICTokenWeighting`. Texts with more tokens than this will be splitted for IC computation.
            term_alterning (Union[int,str]): Parameter for term alternation in the multi-round IC calculation.
                It can be an integer (e.g., N = 6) or the string "sentence" 
                When using an integer N, one of each N terms will be masked each round.
                A larger N value implies a more accurate IC estimation (up to a certain point), but slower computation because more rounds are required.
                If "sentence" is used, the text will be split into sentences, and one of the sentence terms will be masked at each round.
                This approach is significantly slower but may provide the most accurate IC estimation.
            use_chunking (bool): Whether to use chunking for term span extraction. It is recommended for a more precise IC calculation.

        Returns:
            Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
                - float: The average TPI for the corpus.
                - np.ndarray: An array of TPI values for each document.
                - Dict[str,np.ndarray]: A dictionary containing precomputed ICs (used for caching).
                - np.ndarray: An array of IC multipliers (i.e., IC of masked terms divided by IC of non-masked terms) for each document.
        """

        # Initialize outputs
        tpi_array = np.empty(len(masked_docs))
        if self.ics_dict is None:
            self.ics_dict = {} # Used to avoid recomputing, for each anonymization, the original document's ICs (which are always identical)
        ic_multiplier_array = np.empty(len(masked_docs))

        # Define token weighting
        if weighting_model_name is None:
            token_weighting = UniformTokenWeighting()        
        else:
            token_weighting = ICTokenWeighting(model_name=weighting_model_name, device=DEVICE,
                                               max_segment_length=weighting_max_segment_length)

        # For each masked document
        for i, masked_doc in enumerate(masked_docs):
            doc = documents[masked_doc.doc_id]

            # Get terms spans and mask
            spans = self._get_terms_spans(doc.spacy_doc, use_chunking=use_chunking)
            masked_spans = self._filter_masked_spans(doc, masked_doc)
            spans_mask = self._get_spans_mask(spans, masked_spans) # Non-masked=True(1), Masked=False(0)

            # Get IC for all spans
            if masked_doc.doc_id in self.ics_dict:
                spans_IC = self.ics_dict[masked_doc.doc_id] # Use precomputed ICs
            else:
                spans_IC = self._get_ics(spans, doc, term_alterning, token_weighting)
                self.ics_dict[masked_doc.doc_id] = spans_IC # Store ICs (useful as cache)
            
            # Get TIC of the original and masked documents
            original_TIC = spans_IC.sum()
            masked_TIC = spans_IC[spans_mask].sum()

            # Compute document TPI
            tpi_array[i] = masked_TIC / original_TIC 

            # Compute document IC multiplier
            n_terms = len(spans)
            n_masked_terms = np.count_nonzero(spans_mask==0)
            info_loss = original_TIC - masked_TIC
            masked_term_IC = info_loss / n_masked_terms if n_masked_terms != 0 else 0
            n_nonmasked_terms = n_terms - n_masked_terms
            nonmasked_term_IC = masked_TIC / n_nonmasked_terms if n_nonmasked_terms != 0 else 0
            ic_multiplier_array[i] = masked_term_IC / nonmasked_term_IC if nonmasked_term_IC != 0 else 0

        # Dispose token weighting
        del token_weighting

        # Get corpus TPI as the mean
        tpi = tpi_array.mean()

        return tpi, tpi_array, self.ics_dict, ic_multiplier_array

    def _get_terms_spans(self, spacy_doc:spacy.tokens.Doc, use_chunking:bool=True) -> List[Tuple[int, int]]:
        text_spans = []
        added_tokens = np.zeros(len(spacy_doc), dtype=bool)

        if use_chunking:
            for chunk in spacy_doc.ents:
                start = spacy_doc[chunk.start].idx
                last_token = spacy_doc[chunk.end - 1]
                end = last_token.idx + len(last_token)
                text_spans.append((start, end))
                added_tokens[chunk.start:chunk.end] = True

            for chunk in spacy_doc.noun_chunks:
                # If is it not already added
                if not added_tokens[chunk.start:chunk.end].any():
                    start = spacy_doc[chunk.start].idx
                    last_token = spacy_doc[chunk.end - 1]
                    end = last_token.idx + len(last_token)
                    text_spans.append((start, end))
                    added_tokens[chunk.start:chunk.end] = True                

        # Add text spans after last chunk (or all spans, if chunks are ignored)
        for token_idx in range(len(spacy_doc)):
            if not added_tokens[token_idx]:
                token = spacy_doc[token_idx]            
                if token.text.strip() not in ["", "\n"]:  # Avoiding empty spans
                    start = token.idx
                    end = start + len(token)
                    text_spans.append((start, end))

        # Sort text spans by starting position
        text_spans = sorted(text_spans, key=lambda span: span[0], reverse=False)

        return text_spans

    def _filter_masked_spans(self, doc:Document, masked_doc:MaskedDocument) -> List[Tuple[int, int]]:
        filtered_masked_spans = []

        masking_array = np.zeros(len(doc.spacy_doc.text), dtype=bool)
        for (s, e) in masked_doc.masked_spans:
            masking_array[s:e] = True
        
        ini_current_mask = -1
        for idx, elem in enumerate(masking_array):
            # Start of mask
            if ini_current_mask == -1 and elem:
                ini_current_mask = idx
            # End of mask
            elif ini_current_mask >= 0 and not elem:
                filtered_masked_spans.append((ini_current_mask, idx))
                ini_current_mask = -1
        
        return filtered_masked_spans

    def _get_spans_mask(self, spans:List[Tuple[int, int]], masked_spans:List[Tuple[int, int]]) -> np.ndarray:
        spans_mask = np.empty(len(spans), dtype=bool)
        sorted_masked_spans = sorted(masked_spans, key=lambda span: span[0], reverse=False)

        for i, (span_start, span_end) in enumerate(spans):
            # True(1)=Non-masked, False(0)=Masked
            spans_mask[i] = True
            for (masked_span_start, masked_span_end) in sorted_masked_spans:
                if span_start <= masked_span_start < span_end or span_start < masked_span_end <= span_end:
                    spans_mask[i] = False
                elif masked_span_start > span_end: # Break if masked span starts too late
                    break

        return spans_mask

    def _get_ics(self, spans:List[Tuple[int, int]], doc:Document, term_alterning:int, token_weighting:TokenWeighting) -> np.ndarray:
        spans_IC = np.empty(len(spans))

        # N-Term Alterning (N-TA)
        if isinstance(term_alterning, int) and term_alterning > 1: 
            # Get ICs by masking each N term at a time, with all the document as context
            spans_batch = [spans[i::term_alterning] for i in range(term_alterning)]
            batch_ICs = self._get_spans_ICs_batch(spans_batch, doc, token_weighting)
            for i in range(term_alterning): # Reconstruct the original order
                spans_IC[i::term_alterning] = batch_ICs[i]
        
        # Sentence-Term Alterning (S-TA)
        elif isinstance(term_alterning, str) and term_alterning.lower() == "sentence":
            # Get ICs by masking 1 term of each sentence at a time, with the sentence as context
            # Get sentences spans
            sentences_spans = [[sent.start_char, sent.end_char] for sent in doc.spacy_doc.sents]
            
            # Iterate sentences
            ini_span_idx = 0
            for sentence_span in sentences_spans:
                sentence_start, sentence_end = sentence_span

                # Get spans in the sentence
                span_idx = ini_span_idx
                first_sentence_span_idx = -1
                is_sentence_complete = False
                while span_idx < len(spans) and not is_sentence_complete:
                    # If span belongs to sentence (first spans may not belong to any sentence)
                    if spans[span_idx][0] >= sentence_start and spans[span_idx][1] < sentence_end:
                        if first_sentence_span_idx == -1:  # If first sentence span
                            first_sentence_span_idx = span_idx  # Store first index
                        span_idx += 1  # Go to next span
                    # If not belongs and sentence is started, sentence completed
                    elif first_sentence_span_idx != -1:
                        is_sentence_complete = True
                    # Otherwise, go to next span
                    else:
                        span_idx += 1

                # Update initial span index for sentence spans searching
                ini_span_idx = span_idx                         

                # Get IC for each span of the sentence
                spans_for_IC = spans[first_sentence_span_idx:span_idx]
                spans_batch = [[span] for span in spans_for_IC]
                batch_ICs = self._get_spans_ICs_batch(spans_batch, doc, token_weighting,
                                                      context_span=sentence_span)
                for i in range(len(spans_for_IC)):
                    spans_IC[first_sentence_span_idx+i] = batch_ICs[i][0]
        else:
            raise RuntimeError(f"Term alterning {term_alterning} is invalid. It must be an integer greater than 1 or \"sentence\".")

        return spans_IC

    def _get_spans_ICs_batch(self, spans_groups:List[List[Tuple[int,int]]], doc:Document, 
                        token_weighting:TokenWeighting, context_span:Optional[Tuple[int,int]] = None) -> List[np.ndarray]:
        """
        Obtains the ICs of a batch of spans using batched `token_weighting`.
        
        Args:
            spans_groups: List of span groups, where each group contains spans to process together
            doc: Document object
            token_weighting: TokenWeighting instance
            context_span: Optional context span, defaults to entire document
            
        Returns:
            List of numpy arrays, one for each spans group
        """
        # By default, context span is all the document
        if context_span is None:
            context_span = (0, len(doc.text))

        # Get context
        context_start, context_end = context_span
        context = doc.text[context_start:context_end]

        # Prepare batch inputs
        batch_contexts = []
        batch_spans = []
        
        for spans_group in spans_groups:
            # Adjust spans to the context for this group
            in_context_spans = []
            for (start, end) in spans_group:
                in_context_spans.append((start - context_start, end - context_start))
            
            batch_contexts.append(context)
            batch_spans.append(in_context_spans)

        # Process all groups in a single batch call
        batch_ICs = token_weighting.get_weights_batched_chunked(batch_contexts, batch_spans)
        
        return batch_ICs
    