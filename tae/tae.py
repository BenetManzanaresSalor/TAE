import json, logging, os, csv
os.environ["OMP_NUM_THREADS"] = "1" # Done before loading MKL to avoid: \sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1
from typing import Dict, List, Optional, Union
from datetime import datetime
import inspect
import pkgutil
import importlib

from tqdm.autonotebook import tqdm
import spacy

from .utils import *
from .metrics import MetricABC
METRICS_PACKAGE_NAME = "tae.metrics"


class TAE:
    """Text Anonymization Evaluator (TAE) class, defined for the utility and privacy assessment of a text anonymization corpus.
    It is instanciated for a particular corpus, and provides functions for several evaluation metrics.
    Optionally, the corpus can include gold annotations, used for precision and recall metrics."""

    #region Attributes

    documents:Dict[str, Document]
    spacy_nlp=None
    gold_annotations_ratio:int
    metric_classes:dict
    metric_names:list

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
        
        # Preprocess corpus
        self._preprocess_corpus(corpus, spacy_model_name)

        self.metric_classes = self._get_metric_classes()
        self.metric_names = list(self.metric_classes.keys())

    def _preprocess_corpus(self, corpus, spacy_model_name):
        self.documents = {}  # Dictionary of documents indexed by identifier

        # Loading the spaCy model
        self.spacy_nlp = spacy.load(spacy_model_name, disable=["lemmatizer"])    
        
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

    def _get_metric_classes(self, package_name=METRICS_PACKAGE_NAME, abstract_class=MetricABC) -> dict:
        metric_classes = {}
        
        # 1. Load the parent package
        package = importlib.import_module(package_name)

        # 2. Iterate over modules in the package
        for loader, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            if is_pkg:
                continue

            full_module_name = f"{package_name}.{module_name}"
            module = importlib.import_module(full_module_name)

            # 3. Inspect members
            for class_name, class_type in inspect.getmembers(module, inspect.isclass):
                # Check 1: Is it a subclass of our abstract base?
                # Check 2: Is it NOT the abstract base itself?
                # Check 3: Was it DEFINED in this specific module?
                if (issubclass(class_type, abstract_class) and 
                    class_type is not abstract_class and 
                    class_type.__module__ == full_module_name):          
                    metric_classes[class_name.lower()] = class_type

        return metric_classes

    #endregion


    #region Evaluation

    def evaluate(self, anonymizations:Union[Dict[str, List[MaskedDocument]],Dict[str, str]],
                 metrics:Dict[str,dict],
                 results_file_path:Optional[str]=None) -> Dict[str,Dict[str,float]]:
        """
        Evaluates multiple anonymizations based on the specified metrics.

        Args:
            anonymizations (Union[Dict[str, List[MaskedDocument]],Dict[str, str]]): A dictionary where keys are anonymization names
                and values are lists of MaskedDocument or strings corresponding to paths to JSON files containing the anonymizations. 
                In the latter case, the lists of MaskedDocument contained in those JSON files are loaded.
            metrics (Dict[str, dict]): A dictionary where keys are metric names and values are their parameters.
                Metric names are splitted by underscores ("_").
                The string before the first underscore must correspond to the name of a class described in the `metrics` folder and inherits from `MetricABC`.
            results_file_path (Optional[str]): The path to a file where results will be written.

        Returns:
            dict: A dictionary mapping metric name to result, which is a (nested) dictionary mapping anonymization name to metric's numerical value.
        """
        
        results = {}

        # Load anonymizations from file if they are defined as paths
        if isinstance(next(iter(anonymizations.values())),str):
            for anon_name, anon_file_path in anonymizations.items():
                anonymizations[anon_name] = MaskedCorpus(anon_file_path)

        # Initial checks
        self._check_anonymizations(anonymizations)

        # Write results file header
        if results_file_path:
            self._write_into_results(results_file_path, ["Metric/Anonymization"]+list(anonymizations.keys()))

        # For each metric
        for metric_name, metric_parameters in metrics.items():
            logging.info(f"########################### Computing {metric_name} metric ###########################")
            try:
                metric_key = metric_name.split("_")[0].lower() # Text before first underscore is name of the metric, the rest is freely used
                metric_class = self.metric_classes.get(metric_key, None)

                # If metric is invalid
                if metric_class is None:
                    logging.warning(f"Metric key \"{metric_key}\" (extracted from metric name \"{metric_name}\") is unknown, so there will be no results. | Valid metric keys: {self.metric_names}")          
                # If metric is valid
                else:
                    # Compute
                    metric_instance = metric_class()
                    metric_results = metric_instance.evaluate(anonymizations, self.documents, **metric_parameters)
                    del metric_instance # Delete instance for saving memory

                    # Save results
                    results[metric_name] = metric_results
                    value_strings = []
                    for value in metric_results.values():
                        value_strings.append(f"{value:.3f}" if isinstance(value, float) else str(value))
                    if results_file_path:                        
                        self._write_into_results(results_file_path, [metric_name]+value_strings)
                    
                    # Show results all together for inmediate comparison
                    msg = f"Results for {metric_name}:"
                    for anon_name, value_str in zip(metric_results.keys(), value_strings):
                        msg += f"\n\t\t\t\t\t{anon_name}: {value_str}"
                    logging.info(msg)
            
            except Exception as e:
                logging.error(f"Exception in metric {metric_name}: {e}")
        
        return results

    def _check_anonymizations(self, anonymizations:Dict[str, List[MaskedDocument]]):
        # Check each anonymization has a masked version for all the documents in the corpus
        for anon_name, masked_docs in anonymizations.items():
            corpus_doc_ids = set(self.documents.keys())
            for masked_doc in masked_docs:
                if masked_doc.doc_id in corpus_doc_ids:
                    corpus_doc_ids.remove(masked_doc.doc_id)
                else:
                    logging.warning(f"Anonymization {anon_name} includes a masked document (ID={masked_doc.doc_id}) not present in the corpus")
            if len(corpus_doc_ids) > 0:
                raise RuntimeError(f"Anonymization {anon_name} misses masked documents for the following {len(corpus_doc_ids)} ID/s: {corpus_doc_ids}")

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
