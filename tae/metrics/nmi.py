from typing import Dict, List, Tuple
import logging
import gc
import re

import numpy as np
from tqdm.autonotebook import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

from ..utils import DEVICE, MASKING_MARKS, ORIGINAL_TEXT_KEY, Document, MaskedDocument
from .metric_abc import MetricABC

# NMI default settings
NMI_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # From the Sentence Transformers library (https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) or others such as "bert-base-cased"
NMI_MIN_K = 2
NMI_MAX_K = 32
NMI_K_MULTIPLIER = 2
NMI_REMOVE_MASK_MARKS = True
NMI_N_CLUSTERINGS = 5
NMI_N_TRIES_PER_CLUSTERING = 50


class NMI(MetricABC):
    def evaluate(self, anonymizations:Dict[str, List[MaskedDocument]], documents:Dict[str,Document],
                 min_k:int=NMI_MIN_K, max_k:int=NMI_MAX_K,
                 k_multiplier:int=NMI_K_MULTIPLIER, embedding_model_name:str=NMI_EMBEDDING_MODEL_NAME,
                 remove_mask_marks:bool=NMI_REMOVE_MASK_MARKS, mask_marks:List[str]=MASKING_MARKS,
                 n_clusterings:int=NMI_N_CLUSTERINGS,
                 n_tries_per_clustering:int=NMI_N_TRIES_PER_CLUSTERING,
                 verbose:bool=True) -> Dict[str, float]:
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
        
        return results
    

    def _get_corpora_embeddings(self, corpora:List[List[str]], embedding_model_name:str=NMI_EMBEDDING_MODEL_NAME,
                                 remove_mask_marks:bool=NMI_REMOVE_MASK_MARKS, mask_marks:List[str]=MASKING_MARKS,
                                 device:str=DEVICE) -> List[np.ndarray]:
        corpora_embeddings = []

        # Load model
        model = SentenceTransformer(embedding_model_name, device=device)
        model.eval()
        
        # Collect embeddings
        mask_marks_re_pattern = "|".join([re.escape(m) for m in mask_marks])
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
