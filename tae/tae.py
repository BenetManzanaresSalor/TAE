#region Imports

import json, re, logging, os, csv, gc
os.environ["OMP_NUM_THREADS"] = "1" # Done before loading MKL to avoid: \sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1
from typing import Dict, List, Tuple, Optional, Set, Iterator, Union
from datetime import datetime
from dataclasses import dataclass
from functools import partial

from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import spacy

import torch
from sentence_transformers import SentenceTransformer

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from .utils import *
from .tri import TRI

#endregion


#region Constants


#region Input

# Metric names
PRECISION_METRIC_NAME = "Precision"
WEIGHTED_PRECISION_METRIC_NAME = "PrecisionWeighted"
RECALL_METRIC_NAME = "Recall"
RECALL_PER_ENTITY_METRIC_NAME = "RecallPerEntityType"
TPI_METRIC_NAME = "TPI"
TPS_METRIC_NAME = "TPS"
NMI_METRIC_NAME = "NMI"
TRIR_METRIC_NAME = "TRIR"
METRIC_NAMES = [PRECISION_METRIC_NAME, WEIGHTED_PRECISION_METRIC_NAME, RECALL_METRIC_NAME, RECALL_PER_ENTITY_METRIC_NAME, TPI_METRIC_NAME, TPS_METRIC_NAME, NMI_METRIC_NAME, TRIR_METRIC_NAME]
METRICS_REQUIRING_GOLD_ANNOTATIONS = [PRECISION_METRIC_NAME, WEIGHTED_PRECISION_METRIC_NAME, RECALL_METRIC_NAME, RECALL_PER_ENTITY_METRIC_NAME]

#endregion


#region Metric-specific

# Precision default settings
PRECISION_TOKEN_LEVEL=True

# Recall default settings
RECALL_INCLUDE_DIRECT=True
RECALL_INCLUDE_QUASI=True
RECALL_TOKEN_LEVEL=True

# TPI default settings
TPI_TERM_ALTERNING = 6
TPI_USE_CHUNKING = True

# TPS default settings
TPS_TERM_ALTERNING = 6
TPS_USE_CHUNKING = True
TPS_SIMILARITY_MODEL_NAME = "paraphrase-albert-base-v2" # From the Sentence Transformers library (https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) or others such as "bert-base-cased"

# NMI default settings
NMI_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # From the Sentence Transformers library (https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) or others such as "bert-base-cased"
NMI_MIN_K = 2
NMI_MAX_K = 32
NMI_K_MULTIPLIER = 2
NMI_REMOVE_MASK_MARKS = True
NMI_N_CLUSTERINGS = 5
NMI_N_TRIES_PER_CLUSTERING = 50

# Most of TRIR default settings are defined in the TRI class (in the tri.py file)
BACKGROUND_KNOWLEDGE_KEY = "background_knowledge" # For TRIR background knowledge dataframe

#endregion


#endregion


#region TAE


class TAE:
    """Text Anonymization Evaluator (TAE) class, defined for the utility and privacy assessment of a text anonymization corpus.
    It is instanciated for a particular corpus, and provides functions for several evaluation metrics.
    Optionally, the corpus can include gold annotations, used for precision and recall metrics."""

    #region Attributes

    documents:Dict[str, Document]
    spacy_nlp=None
    gold_annotations_ratio:int
    metrics_funcs:dict

    #endregion


    #region Initialization
    
    def __init__(self, corpus:Union[str,List[Dict]], spacy_model_name:str=SPACY_MODEL_NAME):
        """
        Initializes the `TAE` with a given corpus and spaCy model.

        Args:
            corpus (Union[str,List[Document]]): Path to the corpus JSON file or 
                list of documents to be evaluated (result of loading the JSON).
            spacy_model_name (str): The name of the spaCy model to load.
        """

        # Load corpus from file if it's a path
        if type(corpus)==str:
            with open(corpus, encoding="utf-8") as f:
                corpus = json.load(f)
            if type(corpus)!=list:
                raise RuntimeError("Corpus JSON file must be a list of documents")

        # Documents indexed by identifier
        self.documents = {}

        # Loading the spaCy model
        self.spacy_nlp = spacy.load(spacy_model_name, disable=["lemmatizer"])        
        
        # Load corpus
        n_docs_with_annotations = 0
        for doc in tqdm(corpus, desc=f"Loading corpus of {len(corpus)} documents"):
            for key in MANDATORY_CORPUS_KEYS:
                if key not in doc:
                    raise RuntimeError(f"Document {doc.doc_id} missing mandatory key: {key}")
            
            # Parsing the document with spaCy
            spacy_doc = self.spacy_nlp(doc[ORIGINAL_TEXT_KEY])

            # Get gold annotations (if present)
            gold_annotations = doc.get(GOLD_ANNOTATIONS_KEY, None)
            
            # Creating the actual document (identifier, text and gold annotations)
            new_doc = Document(doc[DOC_ID_KEY], doc[ORIGINAL_TEXT_KEY], spacy_doc, gold_annotations)
            self.documents[doc[DOC_ID_KEY]] = new_doc
            if len(new_doc.gold_annotated_entities) > 0:
                n_docs_with_annotations += 1
        
        # Notify the number and percentage of annotated documents
        self.gold_annotations_ratio = n_docs_with_annotations / len(self.documents)
        logging.info(f"Number of gold annotated documents: {n_docs_with_annotations} ({self.gold_annotations_ratio:.3%})")

        # Create dictionary of metric functions (used in _get_partial_metric_func)
        self.metrics_funcs = {PRECISION_METRIC_NAME:self.get_precision,
                              WEIGHTED_PRECISION_METRIC_NAME:self.get_weighted_precision,
                              RECALL_METRIC_NAME:self.get_recall,
                              RECALL_PER_ENTITY_METRIC_NAME:self.get_recall_per_entity_type,
                              TPI_METRIC_NAME:self.get_TPI,
                              TPS_METRIC_NAME:self.get_TPS,
                              NMI_METRIC_NAME:self.get_NMI,
                              TRIR_METRIC_NAME:self.get_TRIR}

    #endregion


    #region Evaluation

    def evaluate(self, anonymizations:Union[Dict[str, List[MaskedDocument]],Dict[str, str]], metrics:Dict[str,dict], results_file_path:Optional[str]=None) -> dict:
        """
        Evaluates multiple anonymizations based on the specified metrics.

        Args:
            anonymizations (Union[Dict[str, List[MaskedDocument]],Dict[str, str]]): A dictionary where keys are anonymization names
                and values are lists of MaskedDocument or strings corresponding to paths to JSON files containing the anonymizations. 
                In the latter case, the lists of MaskedDocument contained in those JSON files are loaded.
            metrics (Dict[str, dict]): A dictionary where keys are metric names and values are their parameters.
                Metric names are splitted by underscores ("_"). The string before the first underscore must be one of those present in `METRIC_NAMES`.
            results_file_path (Optional[str]): The path to a file where results will be written.

        Returns:
            dict: A dictionary containing the evaluation results for each metric and anonymization.
        """
        
        results = {}

        # Load anonymizations from disk if they are paths
        if isinstance(next(iter(anonymizations.values())),str):
            for anon_name, anon_file_path in anonymizations.items():
                anonymizations[anon_name] = MaskedCorpus(anon_file_path)

        # Initial checks
        self._eval_checks(anonymizations, metrics)

        # Write results file header
        if results_file_path:
            self._write_into_results(results_file_path, ["Metric/Anonymization"]+list(anonymizations.keys()))

        # For each metric
        for metric_name, metric_parameters in metrics.items():
            logging.info(f"########################### Computing {metric_name} metric ###########################")
            try:
                metric_key = metric_name.split("_")[0] # Text before first underscore is name of the metric, the rest is freely used
                partial_eval_func = self._get_partial_metric_func(metric_key, metric_parameters)

                # If metric is invalid, results are None
                if partial_eval_func is None:
                    logging.warning("There are no results because the metric name is invalid.")
                
                # Otherwise, compute
                else:
                    # For NMI and TRIR, evaluate all anonymizations at once
                    if partial_eval_func.func==self.get_NMI or partial_eval_func.func==self.get_TRIR:
                        output = partial_eval_func(anonymizations)
                        metric_results = output[0] if isinstance(output, tuple) else output # If tuple, the first is metric_results
                    
                    # Otherwise, compute metric for each anonymization
                    else:
                        metric_results = {}
                        ICs_dict = None # ICs cache for TPI and TPS
                        with tqdm(anonymizations.items(), desc="Processing each anonymization") as pbar:
                            for anon_name, masked_docs in pbar:
                                pbar.set_description(f"Processing {anon_name} anonymization")

                                # For TPI and TPS, cache ICs
                                if partial_eval_func.func==self.get_TPI or partial_eval_func.func==self.get_TPS:
                                    output = partial_eval_func(masked_docs, ICs_dict=ICs_dict)
                                    ICs_dict = output[2]
                                # Otherwise, normal computation
                                else:
                                    output = partial_eval_func(masked_docs)
                                
                                metric_results[anon_name] = output[0] if isinstance(output, tuple) else output  # If tuple, the first is metric's value
                                
                    # Save results
                    results[metric_name] = metric_results
                    if results_file_path:
                        self._write_into_results(results_file_path, [metric_name]+list(metric_results.values()))
                    
                    # Show results all together for easy comparison
                    msg = f"Results for {metric_name}:"
                    for name, value in results[metric_name].items():
                        msg += f"\n\t\t\t\t\t{name}: {value}"
                    logging.info(msg)
            
            except Exception as e:
                logging.error(f"Exception in metric {metric_name}: {e}")
        
        return results

    def _eval_checks(self, anonymizations:Dict[str, List[MaskedDocument]], metrics:dict):
        # Check each anonymization has a masked version of all the documents in the corpus
        for anon_name, masked_docs in anonymizations.items():
            corpus_doc_ids = set(self.documents.keys())
            for masked_doc in masked_docs:
                if masked_doc.doc_id in corpus_doc_ids:
                    corpus_doc_ids.remove(masked_doc.doc_id)
                else:
                    logging.warning(f"Anonymization {anon_name} includes a masked document (ID={masked_doc.doc_id}) not present in the corpus")
            if len(corpus_doc_ids) > 0:
                raise RuntimeError(f"Anonymization {anon_name} misses masked documents for the following {len(corpus_doc_ids)} ID/s: {corpus_doc_ids}")
        
        # Check all metrics are valid and can be computed
        for name, parameters in metrics.items():
            metric_key = name.split("_")[0]
            if not metric_key in METRIC_NAMES:
                logging.warning(f"Metric {metric_key} (from {name}) is unknown, so there will be no results. | Options: {METRIC_NAMES}")
            elif name in METRICS_REQUIRING_GOLD_ANNOTATIONS and self.gold_annotations_ratio < 1:
                raise RuntimeError(f"Metric {name} depends on gold annotations, but these are not present for all documents (only for a {self.gold_annotations_ratio:.3%})")

    def _get_partial_metric_func(self, metric_name:str, parameters:dict) -> Optional[partial]:
        func = self.metrics_funcs.get(metric_name, None)
        partial_func = None if func is None else partial(func, **parameters)
        return partial_func # Result would be None if name is invalid

    def _write_into_results(self, results_file_path:str, values:list):
        # Create containing directory if it does not exist
        directory = os.path.dirname(results_file_path)
        if directory and not os.path.exists(directory): # If it does not exist
            os.makedirs(directory, exist_ok=True) # Create directory (including intermediate ones)

        # Store the row of results
        with open(results_file_path, "a+", newline="") as csvfile:
            writer = csv.writer(csvfile)
            datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([datetime_str]+values)
    
    #endregion


    #region Utility metrics


    #region Precision
    
    def get_precision(self, masked_docs:List[MaskedDocument], weighting_model_name:Optional[str]=None,
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
            gold_doc = self.documents[doc.doc_id]
            
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

    def get_weighted_precision(self, masked_docs:List[MaskedDocument], weighting_model_name:Optional[str]=IC_WEIGHTING_MODEL_NAME,
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
        return self.get_precision(masked_docs, weighting_model_name=weighting_model_name,
                      weighting_max_segment_length=weighting_max_segment_length,
                      token_level=token_level, verbose=verbose)

    #endregion
    

    #region TPI and TPS
    
    def get_TPI(self, masked_docs:List[MaskedDocument], weighting_model_name:Optional[str]=IC_WEIGHTING_MODEL_NAME,
            weighting_max_segment_length:int=IC_WEIGHTING_MAX_SEGMENT_LENGTH, 
            term_alterning:Union[int,str]=TPI_TERM_ALTERNING, use_chunking:bool=TPI_USE_CHUNKING,
            ICs_dict:Optional[Dict[str,np.ndarray]]=None) -> Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
        """
        **Text Preserved Information (TPI)** measures the percentage of information content (IC) still present in the masked documents.
        This metric is used to assess utility preservation.
        It was proposed in **Manzanares-Salor et al., A comparative analysis, enhancement and evaluation of text anonymization with pre-trained Large Language Models, Expert Systems With Applications, In Press, 2025**.
        The `ICTokenWeighting` is employed for measuring IC.
        TPI can be seen as an simplified/ablated version of Text Preserved Similarity (TPS), not taking into account replacements and their similarities.

        Args:
            masked_docs (List[MaskedDocument]): Documents together with spans masked by the anonymization method.
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
            ICs_dict (Optional[Dict[str,np.ndarray]]): Precomputed IC values for documents. 
                Used in `evaluate` to avoid recomputing, for each anonymization, the original document's ICs (which are always identical).

        Returns:
            Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
                - float: The average TPI for the corpus.
                - np.ndarray: An array of TPI values for each document.
                - Dict[str,np.ndarray]: A dictionary containing precomputed ICs (used for caching).
                - np.ndarray: An array of IC multipliers (i.e., IC of masked terms divided by IC of non-masked terms) for each document.
        """

        # Initialize outputs
        tpi_array = np.empty(len(masked_docs))
        if ICs_dict is None:
            ICs_dict = {}
        IC_multiplier_array = np.empty(len(masked_docs))

        # Define token weighting
        if weighting_model_name is None:
            token_weighting = UniformTokenWeighting()        
        else:
            token_weighting = ICTokenWeighting(model_name=weighting_model_name, device=DEVICE,
                                               max_segment_length=weighting_max_segment_length)

        # For each masked document
        for i, masked_doc in enumerate(masked_docs):
            doc = self.documents[masked_doc.doc_id]

            # Get terms spans and mask
            spans = self._get_terms_spans(doc.spacy_doc, use_chunking=use_chunking)
            masked_spans = self._filter_masked_spans(doc, masked_doc)
            spans_mask = self._get_spans_mask(spans, masked_spans) # Non-masked=True(1), Masked=False(0)

            # Get IC for all spans
            if masked_doc.doc_id in ICs_dict:
                spans_IC = ICs_dict[masked_doc.doc_id] # Use precomputed ICs
            else:
                spans_IC = self._get_ICs(spans, doc, term_alterning, token_weighting)
                ICs_dict[masked_doc.doc_id] = spans_IC # Store ICs (useful as cache)
            
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
            IC_multiplier_array[i] = masked_term_IC / nonmasked_term_IC if nonmasked_term_IC != 0 else 0

        # Dispose token weighting
        del token_weighting

        # Get corpus TPI as the mean
        tpi = tpi_array.mean()

        return tpi, tpi_array, ICs_dict, IC_multiplier_array

    def get_TPS(self, masked_docs:List[MaskedDocument], weighting_model_name:Optional[str]=IC_WEIGHTING_MODEL_NAME,
            weighting_max_segment_length:int=IC_WEIGHTING_MAX_SEGMENT_LENGTH, term_alterning=TPS_TERM_ALTERNING,
            similarity_model_name:str=TPS_SIMILARITY_MODEL_NAME, use_chunking:bool=TPS_USE_CHUNKING,
            ICs_dict:Optional[Dict[str,np.ndarray]]=None,
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
            ICs_dict (Optional[Dict[str,np.ndarray]]): Precomputed IC values for documents. 
                Used in `evaluate` to avoid recomputing, for each anonymization, the original document's ICs (which are always identical).

        Returns:
            Tuple[float, np.ndarray, Dict[str,np.ndarray], np.ndarray]:
                - float: The average TPS for the corpus.
                - np.ndarray: An array of TPS values for each document.
                - Dict[str,np.ndarray]: A dictionary containing precomputed ICs (used for caching).
                - np.ndarray: An array of similarities for replacements.
        """
        
        # Initialize outputs
        tps_array = np.empty(len(masked_docs))
        if ICs_dict is None:
            ICs_dict = {}
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
            doc = self.documents[masked_doc.doc_id]

            # Get text spans
            spans = self._get_terms_spans(doc.spacy_doc, use_chunking=use_chunking)

            # Get IC for all spans
            if masked_doc.doc_id in ICs_dict:
                spans_IC = ICs_dict[masked_doc.doc_id] # Use precomputed ICs
            else:
                spans_IC = self._get_ICs(spans, doc, term_alterning, token_weighting)
                ICs_dict[masked_doc.doc_id] = spans_IC # Store ICs (useful as cache)

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

        return tps, tps_array, ICs_dict, similarity_array

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

    def _get_ICs(self, spans:List[Tuple[int, int]], doc:Document, term_alterning:int, token_weighting:TokenWeighting) -> np.ndarray:
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
    
    def _get_embedding_func(self, sim_model_name:str) -> Tuple:
        embedding_model = None

        if sim_model_name is None: # Default spaCy model
            embedding_func = lambda x: np.array([self.spacy_nlp(text).vector for text in x])
        else:   # Sentence Transformer
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

    #endregion


    #region NMI

    def get_NMI(self, anonymizations:Dict[str, List[MaskedDocument]], min_k:int=NMI_MIN_K, max_k:int=NMI_MAX_K,
                k_multiplier:int=NMI_K_MULTIPLIER, embedding_model_name:str=NMI_EMBEDDING_MODEL_NAME,
                remove_mask_marks:bool=NMI_REMOVE_MASK_MARKS, mask_marks:List[str]=MASKING_MARKS,
                n_clusterings:int=NMI_N_CLUSTERINGS,
                n_tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING,
                verbose:bool=True) -> Tuple[Dict[str,float], List[List[np.ndarray]], np.ndarray, int]:
        """
        It compares the K-means++ clustering resulting from the original corpus to that resulting from the anonymized documents.
        **Normalized Mutual Information (NMI)** is employed for assessing clustering similarity.
        This approach allows to measure empirical utility preservation for the generic downstream task of clustering.
        This metric was proposed in [Pilán et al., Truthful Text Sanitization Guided by Inference Attacks, Submitted, 2024](https://arxiv.org/abs/2412.12928).
        Clustering is repeated multiple times for minimizing the impact of randomness.
        Furthermore, for this particular implementation, clustering is carried out with multiple Ks increased linearly.
        The returned results are those corresponding to the K which provided the best [silouhette score](https://www.sciencedirect.com/science/article/pii/0377042787901257) in original texts clustering.

        Args:
            anonymizations (Dict[str, List[MaskedDocument]]): A dictionary where keys are anonymization names and values are lists of masked documents.
            min_k (int): The minimum number of clusters `k` to consider.
            max_k (int): The maximum number of clusters `k` to consider.
            k_multiplier (int): The multiplier to increase `k` for each iteration.
                Iterations start with from `min_k` and end when `max_k` is surpassed.
            embedding_model_name (str): Name of the embedding model to use for document vectorial representation.
                It must be compatible with the [Sentence Transformers library](https://www.sbert.net/), such as ["all-MiniLM-L6-v2"](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
            remove_mask_marks (bool): Whether to remove mask marks (e.g., "SENSITIVE" or "PERSON") from the text before computing the embedding.
            mask_marks (List[str]): The list of mask marks to remove if `remove_mask_marks` is `True`.
            n_clusterings (int): The number of clusterings to perform for each `k`. The one with best silouhette will be selected.
            n_tries_per_clustering (int): Number of times the K-means algorithm is run with different centroid seeds, corresponding to `n_init` in [scikit-learn K-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). The one with the best inertia will be selected.
                This is done for each of the clusterings specified in `n_clusterings`.
                Subsequently, the total number of clusterings for each `k` will be `n_clusterings*n_tries_per_clustering`.
            verbose (bool): Whether to print verbose output during execution.

        Returns:
            Tuple[Dict[str,float], np.ndarray, np.ndarray, int]:
                - Dict[str,float]: A dictionary containing the NMI scores for each anonymization.
                - List[List[np.ndarray]]: A list of lists of clustering labels. 
                    For each of the `n_clusterings` for the best `k`, for each of the anonymizations.
                - np.ndarray: An array of silhouette scores for each evaluated `k`.
                - int: The best `k` value chosen based on silhouette score.
        """
        
        # Create the corpora
        orig_corpora = self._get_anonymization_corpora(anonymizations, include_original_text=True)
        nmi_corpora = [[doc_dict[ORIGINAL_TEXT_KEY] for doc_dict in orig_corpora.values()]] # Prepend original texts (ground truth)
        nmi_corpora += [[doc_dict[anon_name] for doc_dict in orig_corpora.values()] for anon_name in anonymizations.keys()]

        # Get the embeddings
        corpora_embeddings = self._get_corpora_embeddings(nmi_corpora, embedding_model_name,
                                                   remove_mask_marks=remove_mask_marks, mask_marks=mask_marks)
        
        # Clustering results based on the maximum silhouette
        values, all_corpora_labels, true_silhouettes, best_k = self._silhouette_based_NMI(corpora_embeddings, min_k=min_k, max_k=max_k, k_multiplier=k_multiplier,
                                                                      n_clusterings=n_clusterings, n_tries_per_clustering=n_tries_per_clustering,
                                                                      verbose=verbose)
        
        # Prepare results
        values = values[1:] # Remove result for the first corpus (ground truth defined by the original texts)
        results = {anon_name:value for anon_name, value in zip(anonymizations.keys(), values)}
        
        return results, all_corpora_labels, true_silhouettes, best_k

    def _get_corpora_embeddings(self, corpora:List[List[str]], embedding_model_name:str=NMI_EMBEDDING_MODEL_NAME,
                                 remove_mask_marks:bool=NMI_REMOVE_MASK_MARKS, mask_marks:List[str]=MASKING_MARKS,
                                 device:str=DEVICE) -> List[np.ndarray]:
        corpora_embeddings = []

        # Load model
        model = SentenceTransformer(embedding_model_name, device=device)
        model.eval()
        
        # Collect embeddings
        mask_marks_re_pattern = "|".join([m.upper() for m in mask_marks])
        for corpus in tqdm(corpora, desc="Computing embeddings"):
            # Remove mask marks if required
            if remove_mask_marks:
                corpus = [re.sub(mask_marks_re_pattern, "", text).strip() for text in corpus]            
            corpus_embeddings = model.encode(corpus,
                                             convert_to_numpy=True,
                                             show_progress_bar=False)
            corpora_embeddings.append(corpus_embeddings)
        
        # Remove model and tokenizer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return corpora_embeddings

    def _silhouette_based_NMI(self, corpora_embeddings:List[np.ndarray], min_k:int=NMI_MIN_K, max_k:int=NMI_MAX_K,
                k_multiplier:int=NMI_K_MULTIPLIER, n_clusterings:int=NMI_N_CLUSTERINGS, 
                n_tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING,
                verbose:bool=True) -> Tuple[np.ndarray, List[List[np.ndarray]], np.ndarray, int]:
        # For multiple ks, use results with maximum silhouette        
        outputs_by_k = {}
        max_silhouette = float("-inf")
        best_k = None
        k = min_k
        while k <= max_k:
            # Clustering for this k
            outputs_by_k[k] = self._get_corpora_multiclustering(corpora_embeddings, k, n_clusterings=n_clusterings,
                                                              n_tries_per_clustering=n_tries_per_clustering)            
            avg_silhouettee = outputs_by_k[k][2].mean() # Average of true_silhouettes
            if avg_silhouettee > max_silhouette:
                max_silhouette, best_k = avg_silhouettee, k
            k *= k_multiplier # By default, duplicate k

        if verbose: logging.info(f"Clustering results for k={best_k} were selected because they correspond to the maximum silhouette ({max_silhouette:.3f})")
        values, all_corpora_labels, true_silhouettes = outputs_by_k[best_k]

        return values, all_corpora_labels, true_silhouettes, best_k

    def _get_corpora_multiclustering(self, corpora_embeddings:List[np.ndarray], k:int, n_clusterings:int=NMI_N_CLUSTERINGS,
                                n_tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING
                                ) -> Tuple[np.ndarray, List[List[np.ndarray]], np.ndarray]:
        results = np.empty((n_clusterings, len(corpora_embeddings)))
        all_corpora_labels = []
        true_silhouettes = np.empty(n_clusterings)
        for clustering_idx in tqdm(range(n_clusterings), desc=f"Clustering k={k}"):
            true_labels, corpora_labels, true_silhouettes[clustering_idx] = self._get_corpora_clustering(corpora_embeddings, k,
                                                                                                        tries_per_clustering=n_tries_per_clustering)
            results[clustering_idx, :] = self._compare_clusterings(true_labels, corpora_labels)
            all_corpora_labels.append(corpora_labels)

        # Average for the n_clusterings
        results = results.mean(axis=0)

        return results, all_corpora_labels, true_silhouettes

    def _get_corpora_clustering(self, corpora_embeddings:List[np.ndarray], k:int,
                                 tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING) -> Tuple[np.ndarray, List[np.ndarray], float]:
        corpora_labels = []

        # First corpus corresponds to the ground truth
        true_labels = self._get_corpus_clustering(corpora_embeddings[0], k, tries=tries_per_clustering)
        true_silhouette = silhouette_score(corpora_embeddings[0], true_labels, metric="cosine")

        # Clusterize for each corpus
        for corpus_embeddings in corpora_embeddings: # Repeating for the first one (ground truth) allows to check consistency
            labels = self._get_corpus_clustering(corpus_embeddings, k, tries=tries_per_clustering)            
            corpora_labels.append(labels)

        return true_labels, corpora_labels, true_silhouette

    def _get_corpus_clustering(self, corpus_embeddings, k:int, tries:int=NMI_N_TRIES_PER_CLUSTERING) -> np.ndarray:
        kmeanspp = KMeans(n_clusters=k, init="k-means++", n_init=tries)
        labels = kmeanspp.fit_predict(corpus_embeddings)
        return labels

    def _compare_clusterings(self, true_labels:np.ndarray, corpora_labels:List[np.ndarray],
                             eval_metric=normalized_mutual_info_score) -> np.ndarray:
        metrics = np.empty(len(corpora_labels))
        
        for idx, corpus_labels in enumerate(corpora_labels):
            metric = eval_metric(corpus_labels, true_labels)
            metrics[idx] = metric
        
        return metrics

    #endregion


    #endregion


    #region Privacy metrics


    #region Recall

    def get_recall(self, masked_docs:List[MaskedDocument], include_direct:bool=RECALL_INCLUDE_DIRECT, 
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

        nb_masked_by_type, nb_by_type = self._get_mask_counts(masked_docs, include_direct, 
                                                                  include_quasi, token_level)
        
        nb_masked_elements = sum(nb_masked_by_type.values())
        nb_elements = sum(nb_by_type.values())
        
        if nb_elements != 0:
            recall = nb_masked_elements / nb_elements
        else:
            recall = 0
            if verbose: logging.warning("Zero annotated identifiers, resulting in a recall of zero")
        
        return recall
    
    def get_recall_per_entity_type(self, masked_docs:List[MaskedDocument], include_direct:bool=RECALL_INCLUDE_DIRECT, 
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
        
        nb_masked_by_type, nb_by_type = self._get_mask_counts(masked_docs, include_direct, 
                                                                  include_quasi, token_level)
        
        return {ent_type:nb_masked_by_type[ent_type]/nb_by_type[ent_type]
                for ent_type in nb_by_type}
                
    def _get_mask_counts(self, masked_docs:List[MaskedDocument], include_direct:bool=RECALL_INCLUDE_DIRECT, 
                                   include_quasi:bool=RECALL_INCLUDE_QUASI,
                                   token_level:bool=RECALL_TOKEN_LEVEL) -> Tuple[Dict[str,int],Dict[str,int]]:
        nb_masked_elements_by_type = {}
        nb_elements_by_type = {}
        
        for doc in masked_docs:            
            gold_doc = self.documents[doc.doc_id]           
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

    #endregion


    #region TRIR

    def get_TRIR(self, anonymizations:Dict[str, List[MaskedDocument]],
                 background_knowledge_file_path:str, output_folder_path:str,
                 verbose:bool=True, **kwargs) -> Dict[str, float]:
        """
        It simulates a **Text Re-Identification Attack (TRIA)** on the anonymized documents in order to measure their **Text Re-Identification Risk (TRIR)**.
        Introduced in [Manzanares-Salor et al., Evaluating the disclosure risk of anonymized documents via a machine learning-based re-identification attack, Data Mining and Knowledge Discovery, 2024](https://link.springer.com/article/10.1007/s10618-024-01066-3),
        this metric evaluates privacy protection focusing on the key factor of *empirical re-identification probability*.
        TRIA builds on the same principles as record linkage attacks, which are widely used for assessing disclosure risk in structured data.
        The approach assumes that an attacker possesses background knowledge (BK) consisting of public information about a *non-strict* superset of the protected individuals. 
        Using this knowledge, the attacker trains a classifier to associate documents with individuals, and then applies the model to anonymized documents in an attempt to link them to the correct individuals from the BK.
        TRIR is defined as the accuracy of this linkage process.
        Args:
            anonymizations (Dict[str, List[MaskedDocument]]): A dictionary where keys are anonymization names
                                                                and values are lists of masked documents.
            background_knowledge_file_path (str): Path to the background knowledge JSON file (*e.g.*, "data/tab/bk/TAB_test_BK=Public.json"). The file must contain a dictionary of background knowledge documents where
                *Key* is the `doc_id` of the document. Since the BK comprehends a *non-strict* superset of the protected individuals, some `doc_id`s may not appear in the corpus and not all corpus `doc_id`s will necessarily be present in the BK.
                *Value*, on the other hand, is the textual content of the document.
            output_folder_path (str): Path to the folder (*e.g.*, `"outputs/tab/TAB_test_BK=Public"`) where some **partial outputs** (e.g., curated data, trained model...) will be stored.
                If the folder or its containing folders are missing, they will be created.
                These outputs can be reused in later executions to compute different TRIR variants (*i.e.*, by adjusting optional parameters) without re-running the entire process.
            verbose (bool): Whether to print verbose output during execution.
            **kwargs: Additional optional parameters to be passed to the TRI class constructor.

        Returns:
            dict: A dictionary where keys are anonymization names and values are their TRIR scores.
        """
        
        # Load corpora
        corpora = self._get_anonymization_corpora(anonymizations)

        # Load background knowledge and add it to the corpora
        with open(background_knowledge_file_path, "r", encoding="utf-8") as f:
            bk_dict = json.load(f)
        for doc_id, bk in bk_dict.items():
            doc_dict = corpora.get(doc_id, {})
            doc_dict[DOC_ID_KEY] = doc_id
            doc_dict[BACKGROUND_KNOWLEDGE_KEY] = bk
            corpora[doc_id] = doc_dict

        # Create dataframe from corpora
        dataframe = pd.DataFrame.from_dict(list(corpora.values()))
        
        # Create and run TRI
        tri = TRI(
            dataframe=dataframe,
            background_knowledge_column=BACKGROUND_KNOWLEDGE_KEY,
            output_folder_path=output_folder_path,
            individual_name_column=DOC_ID_KEY,
            **kwargs)        
        results = tri.run(verbose=verbose)

        # Obtain TRIR
        results = {anon_name:values["eval_Accuracy"] for anon_name, values in results.items()}

        return results

    #endregion


    #endregion


    #region Auxiliar
    
    def _get_anonymization_corpora(self, anonymizations:Dict[str, List[MaskedDocument]],
                                   include_original_text:bool=False) -> Dict[str, Dict[str,str]]:
        corpora = {}
        
        # Transform list of masked docs into dictionaries for faster processing
        anon_dicts = {}
        for anon_name, masked_docs in anonymizations.items():
            anon_dicts[anon_name] = {masked_doc.doc_id:masked_doc for masked_doc in masked_docs}

        # Create a dictionary per document
        for doc_id, doc in self.documents.items():
            doc_dict = {DOC_ID_KEY:doc_id}
            if include_original_text:
                doc_dict[ORIGINAL_TEXT_KEY] = doc.text
            for anon_name, masked_docs_dict in anon_dicts.items():
                masked_doc = masked_docs_dict[doc_id]
                doc_dict[anon_name] = masked_doc.get_masked_text(doc.text)
            corpora[doc_id] = doc_dict

        return corpora

    #endregion


#endregion
