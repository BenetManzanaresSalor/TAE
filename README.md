<h1 align="center">Text Anonymization Evaluator (TAE)</h1>
<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-orange" alt="License MIT"/>
</p>

This repository contains the code and experimental data for the **Text Anonymization Evaluator** (TAE), an evaluation tool for text anonymization including multiple state-of-the-art metrics for both utility preservation and privacy protection.

Experimental data was extracted from the [text-anonymization-benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark) repository, corresponding to the publication [Pilán, I., Lison, P., Øvrelid, L., Papadopoulou, A., Sánchez, D., & Batet, M., Pilán et al., The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization, Computational Linguistics, 2022, Computational Linguistics, 2022](https://aclanthology.org/2022.cl-4.19/). The exact files utilized are located in the [data](data) folder.




## Table of contents
* [Project structure](#project-structure)
* [Install](#install)
  * [From source](#from-source)
  * [From PyPi](#from-pypi)
* [Usage examples](#usage-examples)
  * [From CLI](#from-cli)
  * [From code](#from-code)
* [Configuration](#configuration)
  * [Corpus](#corpus)
  * [Anonymizations](#anonymizations)
  * [Metrics](#metrics)
    * [Utility preservation](#utility-preservation)
      * [Precision](#precision)
      * [PrecisionWeighted](#precisionweighted)
      * [TPI](#tpi)
      * [TPS](#tps)
      * [NMI](#nmi)
    * [Privacy protection](#privacy-protection)
      * [Recall](#recall)
      * [RecallPerEntityType](#recallperentitytype)
      * [TRIR](#trir)
  * [Results](#results)




# Project structure
```
Text Anonymization Evaluator (TAE)
│   README.md                               # This README
│   pyproject.toml                          # Package project defintion file, including dependencies for pip
│   environment.yml                         # Dependencies file for Conda
│   LICENSE.txt                             # License file
│   example_config.json                     # Example configuration file
└───taeval                                  # Package source code folder
│   |   __init__.py                         # Script for package initialization
│   |   __main__.py                         # Script to be executed as CLI
│   │   tae.py                              # Script including the TAE class, containing the main code of the package
│   |   tri.py                              # Script including the TRI class for re-identification risk assessment
│   |   utils.py                            # Script including the general common-usage classes
└───data                                    # Folder for data files
    └───tab                                 # Folder for TAB dataset
        └───corpora                         # Folder for dataset's corpus files
        |   |...
        └───anonymizations                  # Folder for anonymizations to evaluate
        |   |...
        └───bks                             # Folder for background knowledges for re-identification risk assessment
            |...
```




# Install
Our implementation uses [Python 3.9.19](https://www.python.org/downloads/release/python-3919/) as programming language. For dependencies management, we employed [Conda](https://docs.conda.io/en/latest/), with all used packages and resources listed in the [environment.yml]([environment.yml) file. However, we also considered **Pip**, including an equivalent [pyproject.toml](pyproject.toml) file and uploading the package to [PyPi](https://pypi.org/) under the name `taeval`. Below we detail how to install the package [from source](#from-source) and [from PyPi](#from-pypi).

## From source
If you want to use TAE from CLI (see [Usage section](#usage-examples) for details), we recommend to install it from source following the next steps:
1. Download or clone this repository:
    ```console
    git clone https://github.com/BenetManzanaresSalor/TAE
    cd TAE
    ```
2. Install dependencies:
    * Option A: Using Conda 
        * Install [Conda](https://docs.conda.io/en/latest/) if you haven't already.
        * Create a new Conda environment using the [environment.yml](environment.yml) file:
            ```console
            conda create --name ENVIRONMENT_NAME --file environment.yml
            ```
        * Activate the environment:
            ```console
            conda activate ENVIRONMENT_NAME
            ```
    * Option B: Using Pip (this uses the pyproject.toml file)
        ```console
        pip install -e .
        ```

## From PyPi
**IMPORTANT: This package has not yet been uploaded to PyPi.**

If you want to use TAE from code (see [Usage section](#usage-examples) for details), we recommend installing it from PyPi via Pip with the following command:
```console
pip install taeval
```




# Usage examples
TAE was designed to be run [from CLI](#from-cli), but it can also be executed [from code](#from-code). In the following, we instruct into how to execute it using both approaches, assuming that the steps from the [Install section](#install) have been already completed.

*NOTE: During execution with either approach, the current progress, errors, and results will be displayed. In addition, all evaluation results will be stored in a [CSV file at the specified results filepath](#results).*


## From CLI
Running from CLI requires to pass as argument the path to a JSON configuration file. This file contains a dictionary specifiying the corpus, anonymizations, metrics and results filepath to use (check the [Configuration section](#configuration) for details).

For instance, for using the [example_config.json](example_config.json) example configuration file, run the following command:
```console
python -m tae example_config.json
```
This assumes that the current working directory contains the [tae](tae) package folder. Finding this containing folder can be non-trivial if you installed the package [from PyPi](#from-pypi). That is why we recommend to install it [from source](#from-source) for CLI usage.


## From code
Running from code requires creating an instance of the `TAE` class (defined in [tae.py](tae/tae.py)), passing the desired configuration as arguments. This includes the corpus, anonymizations, metrics, and results filepath (see the [Configuration section](#configuration) for details). The setup is equivalent to using a JSON configuration file [from the CLI](#from-cli), but defined directly in code. Depending on the use case, running from code can help reduce the data load from disk.

The following script exemplifies how to use TAE from code (very similar to what is done in [\_\_main\_\_.py](tae/__main__.py)):
```python
from tae import TAE

# Create TAE instance from the corpus file
tae = TAE("data/tab/corpora/TAB_test_Corpus.json")

# Load anonymizations
anonymizations = {
    "Presidio":"data/tab/anonymizaitons/TAB_test_Presidio_Entity.json", 
    "spaCy":"data/tab/anonymizaitons/TAB_test_spaCy_Entity.json",
    "Manual": "data/tab/anonymizaitons/TAB_test_Manual_Entity.json"
}

# Define metrics dictionary
metrics = {
    "Precision":{},                     # Uses default configuration
    "TPS_Default": {},                  # Uses default configuration
    "TPS_TA=4": {"term_alterning":4},   # Uses default configuration except for term_alterning
}

# Define file path for the results CSV file (containing directory will be created automatically)
results_file_path = "outputs/results.csv"

# Run evaluation
results = tae.evaluate(anonymizations, metrics, results_file_path)

# NOTE: The TAE instance can be reused for evaluating the corpus using other anonymizations, metrics and/or results filepath
```
This assumes that you have TAE ready to import. That is trivial if you have install it [from PyPi](#from-pypi), but requires you to have the [tae](tae) package folder within your project workspace if you have install it [from source](#from-source). That is why we recommend to install it [from PyPi](#from-pypi) for usage from code.




# Configuration
The package allows to configure the corpus, anonymizations, metrics and results filepath to use. As specified in the [Usage section](#usage-examples), this can be done using a JSON configuration file (*e.g.*, [example_config.json](example_config.json)) or directly from code (as shown in the [from code section](#from-code)).
Subsections below detail all the parameters for each of the concepts, including input and output files format.


## Corpus
* `corpus | String`: Path to the JSON corpus file, defining the set of documents that need to be protected.
  This parameter can be provided either through the JSON configuration file if running [from CLI](#from-cli), or using the `TAE` constructor if running [from code](#from-code).
  
  The corpus file should follow a format such as that of [TAB_test_Corpus.json](data/tab/corpora/TAB_test_Corpus.json). That is, a list of dictionaries, with each dictionary corresponding to a document and containing at least these key-values:
    * `doc_id | String`: Unique identifier of the document. For the [TRIR metric](#trir), this requires also to be unique for the individual to protect, so each individual is assumed to appear in only one document.
    * `text | String`: Textual content of the document.
    * `annotations | Dictionary (Optional)`: Manual annotations used for the [Precision](#precision), [PrecisionWeighted](#precision), [Recall](#recall) and [RecallPerEntityType](#recallperentitytype) metrics. **Nevertheless, `annotations` can be ignored (*i.e.*, missing or assigned to `None`) if none of these metrics is used.** Annotations are defined by a dictionary where:
        * *key* is the `annotator_id | String` (*e.g*, "annotator1" and "annotator2"), being possible to have one or more annotators per document.
        * *value* is another dictionary containing `entity_mentions` as *key* and, as *value*, the list of all the entities mentioned in the `text`. Each mention in this list is defined by another dictionary, including at least the following key-values:
            * `start_offset | Integer`: Index of the first (included) character in the `text` corresponding to this entity mention.
            * `end_offset | Integer`: Index of the last (not included) character in the `text` corresponding to this entity mention.
            * `entity_type | String`: Conceptual/semantic type of annotated entity (*e.g.*, "CODE", "PERSON", "DATETIME" or "ORG"). Used in [RecallPerEntityType](#recallperentitytype). The set of possible types can be defined freely.
            * `identifier_type | String`: Indicates whether the entity is a "DIRECT" identifier (*i.e.*, allows to identify the individual to protect by itself), a "QUASI" identifier (*i.e.*, allows to identify the individual to protect in combination with other quasi-identifiers) or "NO_MASK" (*i.e.*, is not disclosive, so does not require any masking). 
            * `entity_id | String`: Unique identifier of the entity, not the particular mention. The same entity (*e.g.*, "Edinburgh") can appear multiple times in the text, each time being a different mention but with the same `entity_id`.

        The following JSON block exemplifies the `annotations` structure with a single entity mention:
        ```json
        {
            "annotations": {
                "annotator1": {
                    "entity_mentions": [
                        {
                            "entity_type": "CODE",
                            "entity_mention_id": "001-61807_a1_em1",
                            "start_offset": 54,
                            "end_offset": 62,
                            "span_text": "36110/97",
                            "edit_type": "check",
                            "identifier_type": "DIRECT",
                            "entity_id": "001-61807_a1_e1",
                            "confidential_status": "NOT_CONFIDENTIAL"
                        }
                    ]
                }
            }
        }
        ```

*NOTE: When using the `TAE` constructor [from code](#from-code), `corpus` can also be the list of dictionaries directly, rather than a path to a JSON file containing it. In this way, data load from disk can be reduced.*


## Anonymizations
* `anonymizations | Dictionary`: Defines all anonymizations to be evaluated.
  This parameter can be provided either through the JSON configuration file [from CLI](#from-cli), or using the `TAE.evaluate` function if running [from code](#from-code).
  
  Anonymizations are specified through a dictionary where:
  * *key* is the `anonymization_name | String` (*e.g.*, "Presidio").
  * *value* is the `anonymization_path | String` to the JSON anonymization file.
  
    This anonymization file must follow a structure such as that of [TAB_test_Manual_Entity.json](data/tab/anonymizations/TAB_test_Manual_Entity.json). Specifically, it should contain a dictionary where:
      * *key* is the `doc_id | String`, matching the one used in the `corpus`.
      * *value* is the `maskings_list | List` for that document. This list consists of tuples (represented as lists in JSON) containing two or three elements:
        1. `start_offset | Integer`: Index of the first (included) character in the `text` corresponding to this masking. Should be coherent with the `text` in the `corpus`.
        2. `end_offset | Integer`: Index of the last (not included) character in the `text` corresponding to this masking. Should be coherent with the `text` in the `corpus`.
        3. `replacement | String (Optional)`: Text replacement for this masking. It can be any length. **It can be neglected for some or all maskings, what would be equivalent to supression-based masking.**

      The following JSON block illustrates the structure of an anonymization file for a single document, with one replacement-based masking and one suppression-based masking:
      ```json
      {
        "001-61807": [
          [2956, 2974, "a legal authority"],
          [2940, 2951]
        ]
      }
      ```

*NOTE: When using the `TAE.evaluate` function [from code](#from-code), anonymizations files can also be a list of `MaskedDocument` or a `MaskedCorpus` (i.e., dataclasses defined in [utils.py](tae/utils.py)) directly. In this way, data load from disk can be reduced.*


## Metrics
* `metrics | Dictionary`: Specification of all the evaluation metrics to use. Defined by a dictionary where:
  * *key* corresponds to the `metric_name | String`. When processed, the value is split by the underscore ("_") character. The first part of the split, or the entire string if no underscore exists, is taken as the `metric_key`. This identifier must match one of the following: `["Precision", "PrecisionWeighted", "TPI", "TPS", "NMI", "Recall", "RecallPerEntityType", "TRIR"]`. If the `metric_key` is invalid (for instance, because `metric_name` starts with an underscore) this triggers a warning log and prevents the metric from being computed. Any text following the first underscore is independent of the `metric_key`, and can be used to indicate variations of the same metric (*e.g.*, "TPS", "TPS_TA=4", "TPS_TA=4_bert").

  * *value* corresponds to the `metric_parameters | Dictionary`, which specifies the configurable parameters of a metric that are either mandatory or different from the default ones. This dictionary uses the `parameter_name | String` as *key* and the corresponding `parameter_value`, of varying type, as *value*. For all metrics except [TRIR](#trir), no parameters are mandatory, so `metric_parameters` can be empty, in which case the default settings are applied. **The following subsections describe the configurable parameters for each metric.**


### Utility preservation
The following set of metrics measure or estimate how well the transformed data retains the characteristics, relationships, or patterns of the original corpus. All these metrics are "the higher, the better".

#### Precision
Standard proxy of utility for text anonymization.
It measures the percentage of terms masked by the anonymizations that were also masked by the **manual annotations**.
TAE's implementation follows the version proposed in [Pilán et al., The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization, Computational Linguistics, 2022](https://aclanthology.org/2022.cl-4.19/), which allows for multi-annotated documents (performing a micro-average over annotators), token-level and mention-level assessment and weighting based on information content (IC).

The configurable parameters are:
* `token_level | Boolean | Default=True`: If set to `True`, the precision is computed at the level of tokens, otherwise it is at the mention-level. The latter implies that the whole human-annotated mention (rather than some tokens) needs to be masked for being considered a true positive.
* `weighting_model_name | String | Default=None`: Name of the model to be used for IC weighting, implemented in the `ICTokenWeighting` class. If `None`, uniform weighting (same weights for all) is used. The name must be a valid [HuggingFace's model](https://huggingface.co/models) name, such as ["google-bert/bert-base-uncased"](https://huggingface.co/google-bert/bert-base-uncased).
* `weighting_max_segment_length | Integer | Default=100`: Maximum segment length for `ICTokenWeighting`. Texts with more tokens than this will be splitted for IC computation.

#### PrecisionWeighted
[Precision](#precision) but employing IC weighting by default. It is implemented as a wrapper of the aforementioned [Precision](#precision), so the configurable parameters are exactly the same. The only difference is that `weighting_model_name` defaults to ["google-bert/bert-base-uncased"](https://huggingface.co/google-bert/bert-base-uncased). This avoids the need to select the `weighting_model_name` for IC weighting.

#### TPI
**Text Preserved Information (TPI)** measures the percentage of information content (IC) still present in the masked documents.
It was proposed in **Manzanares-Salor et al., A comparative analysis, enhancement and evaluation of text anonymization with pre-trained Large Language Models, Expert Systems With Applications, In Press, 2025**.
TPI can be seen as an simplified/ablated version of [TPS](#tps) (presented below), not taking into account replacements and their similarities.

The configurable parameters are:
* `term_alterning | Integer or String | Default=6`: Parameter for term alternation in the multi-round IC calculation.
  It can be an integer (e.g., N = 6) or the string "sentence".
  When using an integer N, one of each N terms will be masked each round.
  A larger N value implies a more accurate IC estimation (up to a certain point), but slower computation because more rounds are required.
  If "sentence" is used, the text will be split into sentences, and one of the sentence terms will be masked at each round.
  This approach is significantly slower but may provide the most accurate IC estimation.
* `use_chunking | Boolean | Default=True`: Whether to use chunking for term span extraction. It is recommended for a more precise IC calculation.
* `weighting_model_name | String | Default="google-bert/bert-base-uncased"`: Name of the model to be used for IC weighting, implemented in the `ICTokenWeighting` class. If `None`, uniform weighting (same weights for all) is used. The name must be a valid [HuggingFace's model](https://huggingface.co/models) name, such as ["google-bert/bert-base-uncased"](https://huggingface.co/google-bert/bert-base-uncased).
* `weighting_max_segment_length | Integer | Default=100`: Maximum segment length for `ICTokenWeighting`. Texts with more tokens than this will be splitted for IC computation.

#### TPS
**Text Preserved Similarity (TPS)** measures the percentage of information content (IC) still present in the masked documents, weighted by the similarity between replacement and original terms.
It was proposed in [Pilán et al., Truthful Text Sanitization Guided by Inference Attacks, Submitted, 2024](https://arxiv.org/abs/2412.12928).
TPS can be seen as a replacement-compatible version of [TPI](#tpi) (presented above), pondering it with replacements' similarity.

The configurable parameters are:
* `similarity_model_name | String | Default="paraphrase-albert-base-v2"`: Name of the embedding model for calculating replacement similarity.
    It must be compatible with the [Sentence Transformers library](https://www.sbert.net/), such as ["paraphrase-albert-base-v2"](https://huggingface.co/sentence-transformers/paraphrase-albert-base-v2).
* `term_alterning | Integer or String | Default=6`: Parameter for term alternation in the multi-round IC calculation.
    It can be an integer (e.g., N = 6) or the string "sentence".
    When using an integer N, one of each N terms will be masked each round.
    A larger N value implies a more accurate IC estimation (up to a certain point), but slower computation because more rounds are required.
    If "sentence" is used, the text will be split into sentences, and one of the sentence terms will be masked at each round.
    This approach is significantly slower but may provide the most accurate IC estimation.
* `use_chunking | Boolean | Default=True`: Whether to use chunking for term span extraction. It is recommended for a more precise IC calculation.
* `weighting_model_name | String | Default="google-bert/bert-base-uncased"`: Name of the model to be used for IC weighting, implemented in the `ICTokenWeighting` class.
    If `None`, uniform weighting (same weights for all) is used.
    The name must be a valid [HuggingFace's model](https://huggingface.co/models) name, such as ["google-bert/bert-base-uncased"](https://huggingface.co/google-bert/bert-base-uncased).
* `weighting_max_segment_length | Integer | Default=100`: Maximum segment length for `ICTokenWeighting`. 
    Texts with more tokens than this will be splitted for IC computation.

#### NMI
It compares the K-means++ clustering resulting from the original corpus to that resulting from the anonymized documents.
**Normalized Mutual Information (NMI)** is employed for assessing clustering similarity.
This approach allows to measure empirical utility preservation for the generic downstream task of clustering.
This metric was proposed in [Pilán et al., Truthful Text Sanitization Guided by Inference Attacks, Submitted, 2024](https://arxiv.org/abs/2412.12928).
Clustering is repeated multiple times for minimizing the impact of randomness.
Furthermore, for this particular implementation, clustering is carried out with multiple Ks increased linearly.
The returned results are those corresponding to the K which provided the best [silouhette score](https://www.sciencedirect.com/science/article/pii/0377042787901257) in original texts clustering.

The configurable parameters are:
* `embedding_model_name | String | Default="all-MiniLM-L6-v2"`: Name of the embedding model to use for document vectorial representation.
    It must be compatible with the [Sentence Transformers library](https://www.sbert.net/), such as ["all-MiniLM-L6-v2"](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
* `remove_mask_marks | Boolean | Default=True`: Whether to remove mask marks (e.g., "SENSITIVE" or "PERSON") from the text before computing the embedding.
* `mask_marks | List | Default=MASKING_MARKS`=constant in [utils.py](tae/utils.py): The list of mask marks to remove if `remove_mask_marks` is `True`.
* `min_k | Integer | Default=2`: The minimum number of clusters `k` to consider.
* `max_k | Integer | Default=32`: The maximum number of clusters `k` to consider.
* `k_multiplier | Integer | Default=2`: The multiplier to increase `k` for each iteration.
    Iterations start with from `min_k` and end when `max_k` is surpassed.
    The one with best silouhette will be selected.
* `n_clusterings | Integer | Default=5`: The number of clusterings to perform for each `k`.
    The one with best silouhette will be selected.
* `n_tries_per_clustering | Integer | Default=50`: Number of times the K-means algorithm is run with different centroid seeds, corresponding to `n_init` in [scikit-learn K-means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). The one with the best inertia will be selected.
    This is done for each of the clusterings specified in `n_clusterings`.
    Subsequently, the total number of clusterings for each `k` will be `n_clusterings*n_tries_per_clustering`.



### Privacy protection

#### Recall
Standard privacy proxy for text anonymization.
It measures the percentage of terms masked by the **manual annotations** that were also masked by the anonymizations.
TAE's implementation follows the version proposed in [Pilán et al., The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization, Computational Linguistics, 2022](https://aclanthology.org/2022.cl-4.19/), which allows for multi-annotated documents (performing a micro-average over annotators), token-level and mention-level assessment and independent consideration of direct and quasi identifiers.
The configurable parameters are:
* `token_level | Boolean | Default=True`: If set to `True`, recall is computed at the level of tokens, otherwise it is at the mention-level.
  The latter implies that the whole human-annotated mention (rather than some tokens) needs to be masked for being considered a true positive.
* `include_direct | Boolean | Default=True`: Whether to consider direct identifiers in the metric computation.
* `include_quasi | Boolean | Default=True`: Whether to include quasi identifiers in the metric computation.


#### RecallPerEntityType
It computes [Recall](#recall) factored by the `entity_type` in the **manual annotations**, enabling a fine-grained analysis.
TAE's implementation follows the version proposed in [Pilán et al., The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework for Text Anonymization, Computational Linguistics, 2022](https://aclanthology.org/2022.cl-4.19/), which allows for multi-annotated documents (performing a micro-average over annotators), token-level and mention-level assessment and independent consideration of direct and quasi identifiers.
The configurable parameters are:
* `token_level | Boolean | Default=True`: If set to `True`, recall is computed at the level of tokens, otherwise it is at the mention-level.
  The latter implies that the whole human-annotated mention (rather than some tokens) needs to be masked for being considered a true positive.
* `include_direct | Boolean | Default=True`: Whether to consider direct identifiers in the metric computation.
* `include_quasi | Boolean | Default=True`: Whether to include quasi identifiers in the metric computation.

#### TRIR
It simulates a **Text Re-Identification Attack (TRIA)** on the anonymized documents in order to measure their **Text Re-Identification Risk (TRIR)**.
Introduced in [Manzanares-Salor et al., Evaluating the disclosure risk of anonymized documents via a machine learning-based re-identification attack, Data Mining and Knowledge Discovery, 2024](https://link.springer.com/article/10.1007/s10618-024-01066-3), this metric evaluates privacy protection focusing on the key factor of *empirical re-identification probability*.
TRIA builds on the same principles as record linkage attacks, which are widely used for assessing disclosure risk in structured data.
The approach assumes that an attacker possesses background knowledge (BK) consisting of public information about a *non-strict* superset of the protected individuals. 
Using this knowledge, the attacker trains a classifier to associate documents with individuals, and then applies the model to anonymized documents in an attempt to link them to the correct individuals from the BK.
TRIR is defined as the accuracy of this linkage process.

This metric requires **two mandatory parameters**:
* `background_knowledge_file_path | String`: Path to the background knowledge JSON file (*e.g.*, ["data/tab/bk/TAB_test_BK=Public.json"](data/tab/bks/TAB_test_BK=Public.json)). The file must contain a dictionary of background knowledge documents where:
  * *key* is the `doc_id` of the document. Since the BK comprehends a *non-strict* superset of the protected individuals, some `doc_id`s may not appear in the corpus and not all corpus `doc_id`s will necessarily be present in the BK.
  * *value* is the textual content of the document.
* `output_folder_path | String`: Path to the folder (*e.g.*, `"outputs/tab/TAB_test_BK=Public"`) where some **partial outputs** (e.g., curated data, trained model...) will be stored.
  If the folder or its containing folders are missing, they will be created.
  These outputs can be reused in later executions to compute different TRIR variants (*i.e.*, by adjusting optional parameters detailed below) without re-running the entire process.

**Additionally, TRIR has several optional parameters, detailed in the following.**

* **Data pretreatment**:
  * **Anonymized background knowledge**:
    * `anonymize_background_knowledge | Boolean | Default=True`: If during document pretreatment generate an anonymized version of the background knowledge documents using [spaCy NER](https://spacy.io/api/entityrecognizer) that would be used along with the non-anonymized version. Its usage is strongly recommended, since it can significantly improve re-identification risks. As a counterpoint, it roughly duplicates the training samples, incrementing the training time and RAM consumsumption.
    * `only_use_anonymized_background_knowledge | Boolean | Default=False`: If only using the anonymized version of the background knowledge, instead of concatenating it with the non-anonymized version. This usually results in higher re-identification risks than using only the non-anonymized version, but lower than using both (anonymized and non-anonymized). Created for an ablation study.
  * **Document curation**:
    * `use_document_curation | Boolean | Default=True`: Whether to perform the document curation, consisting of lemmatization and removing of stopwords and special characters. It is inexpensive compared to pretraining or finetuning.

* **Save pretreatment**:
  * `save_pretreatment | Boolean | Default=True`: Whether to save the data after pretreatment. A JSON file name `Pretreated_Data.json` will be generated and stored in the `output_folder_path` folder. As pretreatment it is also included the curation of new anonymizations caused by `updated_loaded_eval_pretreatment=True`.

* **Load pretreatment**:
  * `load_saved_pretreatment | Boolean | Default=True`: If the `Pretreated_Data.json` file exists in the `output_folder_path`, load that data instead of running the pretreatment. Disable it if you completely changed the `corpus` or `anonymizations`. It requires a previous execution with `save_pretreatment=True`.
  * `add_non_saved_anonymizations | Boolean | Default=True`: When loading pretreatment data from `Pretreated_Data.json`, this setting checks whether new `anonymizations` to be processed appeared. If new anonymizations are found, they are loaded and, if `use_document_curation` is true, only these new anonymizations will undergo curation. This option is particularly useful to avoid repeating the entire pretreatment.


* **Load already trained TRI model**:
  * `load_saved_finetuning | Boolean | Default=True`: If the `TRI_Pipeline` exists in the `output_folder_path` directory and contains the model file `model.safetensors`, load that already trained TRI model instead of running the additional pretraining and finetuning. It requires a previous execution with `save_finetuning=True`.
* **Create base language model**:
  * `base_model_name | String | Default="distilbert-base-uncased"`: Name of the base language model in the [HuggingFace's Transformers library](https://huggingface.co/docs/transformers/index) to be used for both additional pretraining and finetuning. Current code is designed for versions of BERT, DistilBERT and RoBERTa. Examples: "distilbert-base-uncased", "distilbert-base-cased", "bert-base-uncased", "bert-base-cased" and "roberta-base". The `ini_extended_model` method from the TRI class (in [tae/tri.py](tae/tri.py)) can be easily modified for other models.
  * `tokenization_block_size | Integer | Default=250`: Number of data samples tokenized at once with [Transformers' tokenizer](https://huggingface.co/docs/transformers/en/main_classes/tokenizer). This is done for limiting and optimizing RAM usage when processing large datasets. The value of 250 is roughly optimized for 32GB of RAM.
* **Additional pretraining**:
  * `use_additional_pretraining | Boolean | Default=True`: Whether additional pre-training (i.e. Masked Language Modeling, MLM) is to be performed to the base language model. Its usage is recommended, since it is inexpensive (compared to finetuning) and can improve re-identification risks.
  * `save_additional_pretraining | Boolean | Default=True`: Whether to save the additionally pretrained language model. The model will be saved as a PyTorch model file `Pretrained_Model.pt` in the `output_folder_path`.
  * `load_saved_pretraining | Boolean | Default=True`: If `use_additional_pretraining` is true and the `Pretrained_Model.pt` file exists, loads that additionally pretrained base model instead of running the process. It requires a previous execution with `save_additional_pretraining=True`.
  * `pretraining_epochs | Integer | Default=3`: Number of additional pretraining epochs.
  * `pretraining_batch_size | Integer | Default=8`: Size of the batches for additional pretraining.
  * `pretraining_learning_rate | Float | Default=5e-05`: Learning rate for the [AdamW optimizer](https://huggingface.co/docs/bitsandbytes/main/en/reference/optim/adamw) to use during additional pretraining.
  * `pretraining_mlm_probability | Float | Default=0.15`: Probability of masking tokens by the [Data Collator](https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorForLanguageModeling.mlm_probability) during the additional pretraining with MLM.
  * `pretraining_sliding_window | String | Default="512-128"`: Sliding window configuration for additional pretraining. Since input documents are assumed to be longer than the maximum number of tokens processable by the language model (maximum sequence length), the text is split into multiple samples. A sliding window mechasim has been implemented, defined by the size of the window and the overlap with the previous window. For instance, use "512-128" for samples/splits of 512 tokens and an overlap of 128 tokens with the previous split/sample. Alternatevely, if "No" is used, one sample/split per sentence will be created, leveraging that sentences are generally shorter than the model maximum sequence length. Reducing the window size and/or incrementing the overlap will result in more samples/splits, what increments the training time.
* **Finetuning**:
  * `finetuning_epochs | Integer | Default=15`: Number of epochs to perform during the finetuning.
  * `finetuning_batch_size | Integer | Default=16`: Size of the batches for finetuning.
  * `finetuning_learning_rate | Float | Default=5e-05`: Learning rate for the [AdamW optimizer](https://huggingface.co/docs/bitsandbytes/main/en/reference/optim/adamw) to use during finetuning.
  * `finetuning_sliding_window | String | Default="100-25"`: Sliding window configuration for finetuning. Since input documents are assumed to be longer than the maximum number of tokens processable by the language model (maximum sequence length), the text is split into multiple samples. A sliding window mechasim has been implemented, defined by the size of the window and the overlap with the previous window. For example, use "512-128" for samples/splits of 512 tokens and an overlap of 128 tokens with the previous split/sample. Alternatevely, if "No" is used, one sample/split per sentence will be created, leveraging that sentences are generally shorter than the model maximum sequence length. Reducing the window size and/or increasing the overlap will result in more samples/splits, what increments the training time.
  * `dev_set_column_name | String | Default=False`: Specifies the anonymization from [anonymizations](#anonymizations) to be used for model selection. If set to `False` (boolean, not string), the model with the highest average accuracy across all anonymization sets will be selected as the final model. If an actual name is provided, the accuracy corresponding to that specific anonymization will be used to choose the best model.
  * `save_finetuning | Boolean | Default=True`: Whether to save the TRI model after the finetuning. The model will be saved as a [Transformers' pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines), creating a folder `TRI_Pipeline` in the `output_folder_path` directory, containing the model file `model.safetensors`.


##### Results
After execution of TRIR, in the `output_folder_path` you can find the following files:
* `Pretreated_Data.json`: If `save_pretreatment` is true, this file is created for saving the pretreated background knowledge and protected documents, sometimes referred as training and evaluation data, respectively. Leveraged if `load_saved_pretreatment` is true.
* `Pretrained_Model.pt`: If `save_additional_pretraining` is true, this file is created for saving the additionally pretrained language model. Leveraged if `load_saved_pretraining` is true.
* `TRI_Pipeline`: If `save_finetuning` is true, this folder is created for saving the resulting classification pipeline. Leveraged if `load_saved_finetuning` is true.
* `Results.csv`: After each epoch of finetuning, the TRIR resulting from each anonymization method will be evaluated. These results are stored (always appending, not overwriting) in a CSV file named `Results.csv`. This file contains the epoch time, epoch number, the TRIR for each anonymization method and the average TRIR. The following table exemplifies TRIR results for three epochs and three methods:
  | Time                | Epoch | Method1 | Method2 | Average |
  | ------------------- | ----- | ------- | ------- | ------- |
  | 01/08/2024 08:50:04 | 1     | 74      | 36      | 55      |
  | 01/08/2024 08:50:37 | 2     | 92      | 44      | 68      |
  | 01/08/2024 08:50:10 | 3     | 94      | 48      | 71      |

  At the end of the program, TRIR is predicted for all the anonymization methods using the best TRI model, employing the criteria defined for the setting `dev_set_column_name`. This final evaluation is also stored in the `Results.csv` file as an "additional epoch".


## Results
* `results_file_path | String`: Path to the CSV file containing the results.  
  This value can be specified either in the JSON configuration file [from CLI](#from-cli) or directly via the `TAE.evaluate` function when running [from code](#from-code).

  While executing `TAE.evaluate`, the computed metric results are appended to this CSV file.  
  If the file or its parent directories do not exist, they will be created automatically.  
  The file is structured as follows:
    * *Header*: The first column stores the datetime, the second column is labeled "Metric/Anonymization" and then one column per `anonymization_name` defined in [anonymizations](#anonymizations).
    * *Metric result*: Each row corresponds to a valid metric defined in [metrics](#metrics). The first column holds the datetime, the second contains the `metric_name`, and the remaining columns hold the results for each anonymization.

  Example of a result table with three metrics and two anonymizations:
  | 2025-08-22 11:02:33 | Metric/Anonymization | Anonymization1 | Anonymization2 |
  |---------------------|----------------------|----------------|----------------|
  | 2025-08-22 11:04:21 | TPS                  | 0.86           | 0.73           |
  | 2025-08-22 11:05:42 | NMI                  | 0.62           | 0.55           |
  | 2025-08-22 11:10:27 | TRIR                 | 0.17           | 0.12           |

*NOTE: When executed [from code](#from-code), the `TAE.evaluate` function also returns the results as a dictionary, keys being the corresponding `metric_name` and values being another dictionary mapping `anonymization_name` to the obtained value.*
