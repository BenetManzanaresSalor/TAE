#region Imports

import logging, argparse, json

from .utils import *
from .tae import TAE

#endregion


#region Constants

# Configuration dictionary keys
CORPUS_CONFIG_KEY = "corpus"
ANONYMIZATIONS_CONFIG_KEY = "anonymizations"
RESULTS_CONFIG_KEY = "results_file_path"
METRICS_CONFIG_KEY = "metrics"
MANDATORY_CONFIG_KEYS = [CORPUS_CONFIG_KEY, ANONYMIZATIONS_CONFIG_KEY, RESULTS_CONFIG_KEY, METRICS_CONFIG_KEY]

#endregion


#region Main

if __name__ == "__main__":

    #region Additional configurations for running standalone

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO) # Configure logging
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)    # Suppress INFO logs from sentence_transformers
    logging.getLogger('transformers').setLevel(logging.WARNING)             # Suppress INFO logs from transformers
    logging.getLogger('torch').setLevel(logging.WARNING)                    # Suppress INFO logs from torch

    #endregion

    #region Arguments parsing

    parser = argparse.ArgumentParser(description='Computes evaluation metrics for text anonymization')
    parser.add_argument('config_file_path', type=str,
                        help='the path to the JSON file containing the evaluation configuration')
    args = parser.parse_args()

    # Load configuration dictionary
    with open(args.config_file_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    for key in MANDATORY_CONFIG_KEYS:
        if not key in config.keys():
            raise RuntimeError(f"Configuration JSON file misses a mandatory key: {key}")
    
    #endregion


    #region Initialization

    logging.info(f"Selected device: {DEVICE}")

    # Create TAE from corpus file path
    tae = TAE(config[CORPUS_CONFIG_KEY])

    # Get anonymizations file paths
    anonymizations = config[ANONYMIZATIONS_CONFIG_KEY]
    
    # Get metrics
    metrics = config[METRICS_CONFIG_KEY]

    # Get file path for the results CSV file
    results_file_path = config[RESULTS_CONFIG_KEY]

    #endregion


    #region Evaluate

    tae.evaluate(anonymizations, metrics, results_file_path)

    #endregion

#endregion