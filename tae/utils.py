#region Imports

import json, re, abc, logging, math, os, gc
os.environ["OMP_NUM_THREADS"] = "1" # Done before loading MKL to avoid: \sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1
from typing import Dict, List, Tuple, Optional, Set, Iterator, Union
from dataclasses import dataclass

import numpy as np
import spacy
import intervaltree

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

#endregion


#region Constants


#region Input

# Corpus dictionary keys
DOC_ID_KEY = "doc_id"
ORIGINAL_TEXT_KEY = "text"
MANDATORY_CORPUS_KEYS = [DOC_ID_KEY, ORIGINAL_TEXT_KEY]
GOLD_ANNOTATIONS_KEY = "annotations"
ENTITY_MENTIONS_KEY = "entity_mentions"
ENTITY_ID_KEY = "entity_id"
START_OFFSET_KEY = "start_offset"
END_OFFSET_KEY = "end_offset"
ENTITY_TYPE_KEY = "entity_type"
IDENTIFIER_TYPE_KEY = "identifier_type"
INDENTIFIER_TYPE_DIRECT = "DIRECT"
INDENTIFIER_TYPE_QUASI = "QUASI"
INDENTIFIER_TYPE_NO_MASK = "NO_MASK"
IDENTIFIER_TYPES = [INDENTIFIER_TYPE_DIRECT, 
                          INDENTIFIER_TYPE_QUASI, 
                          INDENTIFIER_TYPE_NO_MASK]

#endregion


#region General

SPACY_MODEL_NAME = "en_core_web_md"
IC_WEIGHTING_MODEL_NAME = "google-bert/bert-base-uncased"
IC_WEIGHTING_MAX_SEGMENT_LENGTH = 100
IC_WEIGHTING_BATCH_SIZE = 128

# POS tags, tokens or characters that can be ignored scores 
# (because they do not carry much semantic content, and there are discrepancies
# on whether to include them in the annotated spans or not)
POS_TO_IGNORE = {"ADP", "PART", "CCONJ", "DET"} 
TOKENS_TO_IGNORE = {"mr", "mrs", "ms", "no", "nr", "about"}
CHARACTERS_TO_IGNORE = " ,.-;:/&()[]–'\" ’“”"

MASKING_MARKS = ["SENSITIVE", "PERSON", "DEM", "LOC",
                 "ORG", "DATETIME", "QUANTITY", "MISC",
                 "NORP", "FAC", "GPE", "PRODUCT", "EVENT",
                 "WORK_OF_ART", "LAW", "LANGUAGE", "DATE",
                 "TIME", "ORDINAL", "CARDINAL", "DATE_TIME", "DATETIME",
                 "NRP", "LOCATION", "ORGANIZATION", "\*\*\*"]

# Check for GPU with CUDA
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    DEVICE = torch.device("cpu")

#endregion


#endregion


#region Utils

@dataclass
class MaskedDocument:
    """Represents a document in which some text spans are masked, each span
    being expressed by their (start, end) character boundaries.
    Optionally, spans can also have replacement strings.
    """

    doc_id: str
    masked_spans : List[Tuple[int, int]]
    replacements : List[str]

    def get_masked_offsets(self) -> set:
        """Returns the character offsets/indices that are masked"""

        if not hasattr(self, "masked_offsets"):
            self.masked_offsets = {i for start, end in self.masked_spans
                                   for i in range(start, end)}
        return self.masked_offsets
    
    def get_masked_text(self, original_text:str) -> str:
        """Applies masking to the original text based on masked spans and replacements.

        Args:
            original_text (str): The text to be masked.
        
        Returns:
            str: The masked text.
        """

        masked_text = ""+original_text
        
        for (start_idx, end_idx), replacement in zip(reversed(self.masked_spans), reversed(self.replacements)):
            if replacement is None: # If there is no replacement, use an empty string
                replacement = ""
            masked_text = masked_text[:start_idx] + replacement + masked_text[end_idx:]
        
        return masked_text

@dataclass
class MaskedCorpus(List[MaskedDocument]):
    """Auxiliar class that inherits from List[MaskedDocument] for the loading of a MaskedDocument list from file"""

    def __init__(self, masked_docs_file_path:str):
        """
        Initializes the `MaskedCorpus` object from a JSON file.

        Args:
            masked_docs_file_path (str): Path to a JSON file with masked spans (and replacements, optionally)
        """

        masked_docs_list = []
        
        with open(masked_docs_file_path, "r", encoding="utf-8") as fd:
            masked_docs_dict = json.load(fd)
        
        if type(masked_docs_dict)!= dict:
            raise RuntimeError(f"List of MaskedDocuments in {masked_docs_file_path} must contain a dictionary mapping between document identifiers"
                                + " and lists of masked spans in this document")
        
        for doc_id, masked_spans in masked_docs_dict.items():
            doc = MaskedDocument(doc_id, [], [])
            if type(masked_spans)!=list:
                raise RuntimeError("Masked spans must be defined as [start, end, replacement] tuples (replacement is optional)")
            
            for elems in masked_spans:
                # Store span
                start = elems[0]
                end = elems[1]
                doc.masked_spans.append((start, end))

                # Store replacement (None if non-existent or it's an empty string)
                replacement = None if len(elems) < 3 or elems[2].strip() == "" else elems[2]
                doc.replacements.append(replacement)
                
            masked_docs_list.append(doc)

        # Create the class from the list
        super().__init__(masked_docs_list)

@dataclass
class AnnotatedEntity:
    """Represents an entity annotated in a document, with a unique identifier,
    a list of mentions (character-level spans in the document), whether it
    needs to be masked, and whether it corresponds to a direct identifier"""

    entity_id: str
    mentions: List[Tuple[int, int]]
    need_masking: bool
    is_direct: bool
    entity_type: str
    mention_level_masking: List[bool]

    def __post_init__(self):
        """Checks that direct identifiers are masked"""

        if self.is_direct and not self.need_masking:
            raise RuntimeError(f"Annotated entity {self.entity_id} is a direct identifier but it is not always masked")

    @property
    def mentions_to_mask(self) -> list:
        """List of mentions to mask based on the mention level masking"""

        return [mention for i, mention in enumerate(self.mentions)
                if self.mention_level_masking[i]]

class Document:
    """Representation of a document, with an identifier and textual content. 
    Ooptionally, it can include its spaCy document object and/or gold annotations"""

    doc_id:str
    text:str
    spacy_doc:spacy.tokens.Doc
    gold_annotated_entities:Dict[str, AnnotatedEntity]

    #region Initialization
    
    def __init__(self, doc_id:str, text:str, spacy_doc:Optional[spacy.tokens.Doc]=None,
                 gold_annotations:Optional[Dict[str,List]]=None):
        """
        Initializes a new `Document`, optionally including gold annotations.

        Args:
            doc_id (str): The unique document identifier.
            text (str): The text content of the document.
            spacy_doc (Optional[spacy.tokens.Doc]): The spaCy document object.
            gold_annotations (Optional[Dict[str, List]]): Gold annotations, if available.
                Check the `README.md` for more information.
        """

        # The (unique) document identifier, its text and the spacy document
        self.doc_id = doc_id
        self.text = text
        self.spacy_doc = spacy_doc
        
        # Get gold annotated entities (indexed by id) if they exist
        self.gold_annotated_entities = {}
        if not gold_annotations is None: 
            for annotator, ann_by_person in gold_annotations.items():
                if ENTITY_MENTIONS_KEY in ann_by_person: # Optional key           
                    for entity in self._get_entities_from_mentions(ann_by_person[ENTITY_MENTIONS_KEY]):                
                        if entity.entity_id in self.gold_annotated_entities: # Each entity_id is specific for each annotator
                            raise RuntimeError(f"Gold annotations of document {self.doc_id} have an entity ID repeated by multiple annotators: {entity.entity_id}")                        
                        entity.annotator = annotator
                        entity.doc_id = doc_id
                        self.gold_annotated_entities[entity.entity_id] = entity
    
    def _get_entities_from_mentions(self, entity_mentions:List[dict]) -> List[AnnotatedEntity]:
        """
        Processes a list of entity mentions and returns a list of unique AnnotatedEntity objects.

        Args:
            entity_mentions (List[dict]): A list of dictionaries, where each dictionary represents an entity mention.
                Each mention dictionary must contain `entity_id`, `identifier_type`, `start_offset`, and `end_offset` keys.

        Returns:
            List[AnnotatedEntity]: A list of AnnotatedEntity objects, where each object represents a unique entity
            found in the input mentions, consolidating all its mentions.
        """

        entities = {}

        for mention in entity_mentions:                
            for key in [ENTITY_ID_KEY, IDENTIFIER_TYPE_KEY, START_OFFSET_KEY, END_OFFSET_KEY]:
                if key not in mention:
                    raise RuntimeError(f"Entity mention missing key {key}: {mention}")
            
            entity_id = mention[ENTITY_ID_KEY]
            start = mention[START_OFFSET_KEY]
            end = mention[END_OFFSET_KEY]
                
            if start < 0 or end > len(self.text) or start >= end:
                raise RuntimeError(f"Entity mention {entity_id} with invalid character offsets [{start}-{end}] for a text {len(self.text)} characters long")
            
            if mention[IDENTIFIER_TYPE_KEY] not in IDENTIFIER_TYPES:
                raise RuntimeError(f"Entity mention {entity_id} with unspecified or invalid identifier type: {mention['identifier_type']}")

            need_masking = mention[IDENTIFIER_TYPE_KEY] in [INDENTIFIER_TYPE_DIRECT, INDENTIFIER_TYPE_QUASI]
            is_direct = mention[IDENTIFIER_TYPE_KEY]==INDENTIFIER_TYPE_DIRECT
                
            # We check whether the entity is already defined
            if entity_id in entities:                    
                # If yes, we simply add a new mention
                current_entity = entities[entity_id]
                current_entity.mentions.append((start, end))
                current_entity.mention_level_masking.append(need_masking)
                    
            # Otherwise, we create a new entity with one single mention
            else:
                new_entity = AnnotatedEntity(entity_id, [(start, end)], need_masking, is_direct, 
                                             mention[ENTITY_TYPE_KEY], [need_masking])
                entities[entity_id] = new_entity
                
        for entity in entities.values():
            if set(entity.mention_level_masking) != {entity.need_masking}: # Solve inconsistent masking
                entity.need_masking = True
                #logging.warning(f"Entity {entity.entity_id} is inconsistently masked: {entity.mention_level_masking}")
                
        return list(entities.values())
    
    #endregion

    #region Functions

    def is_masked(self, masked_doc:MaskedDocument, entity:AnnotatedEntity) -> bool:
        """
        Given a document with a set of masked text spans, determines whether entity
        is fully masked (which means that all its mentions are masked).

        Args:
            masked_doc (MaskedDocument): The document with masked text spans.
            entity (AnnotatedEntity): The entity to check for masking.

        Returns:
            bool: True if the entity is fully masked, False otherwise.
        """

        for incr, (mention_start, mention_end) in enumerate(entity.mentions):
            
            if self.is_mention_masked(masked_doc, mention_start, mention_end):
                continue
            
            # The masking is sometimes inconsistent for the same entity, 
            # so we verify that the mention does need masking
            elif entity.mention_level_masking[incr]:
                return False
        
        return True
    
    def is_mention_masked(self, masked_doc:MaskedDocument, mention_start:int, mention_end:int) -> bool:
        """
        Given a document with a set of masked text spans and a particular mention span,
        determine whether the mention is fully masked (taking into account special
        characters or PoS/tokens to ignore).

        Args:
            masked_doc (MaskedDocument): The document with masked text spans.
            mention_start (int): The starting character offset of the mention.
            mention_end (int): The ending character offset of the mention.

        Returns:
            bool: True if the mention is fully masked, False otherwise.
        """

        # Computes the character offsets that must be masked
        offsets_to_mask = set(range(mention_start, mention_end))

        # We build the set of character offsets that are not covered
        non_covered_offsets = offsets_to_mask - masked_doc.get_masked_offsets()
            
        # If we have not covered everything, we also make sure punctuations
        # spaces, titles, etc. are ignored
        if len(non_covered_offsets) > 0:
            span = self.spacy_doc.char_span(mention_start, mention_end, alignment_mode="expand")
            for token in span:
                if token.pos_ in POS_TO_IGNORE or token.lower_ in TOKENS_TO_IGNORE:
                    non_covered_offsets -= set(range(token.idx, token.idx+len(token)))
        for i in list(non_covered_offsets):
            if self.text[i] in set(CHARACTERS_TO_IGNORE):
                non_covered_offsets.remove(i)

        # If that set is empty, we consider the mention as properly masked
        return len(non_covered_offsets) == 0

    def get_entities_to_mask(self, include_direct:bool=True, include_quasi:bool=True) -> List[AnnotatedEntity]:
        """Return entities that should be masked and satisfy the constraints specified as arguments.

        Args:
            include_direct (bool): Whether to include direct entities. Defaults to True.
            include_quasi (bool): Whether to include quasi entities. Defaults to True.

        Returns:
            List[AnnotatedEntity]: A list of entities that should be masked.
        """
        
        to_mask = []
        for entity in self.gold_annotated_entities.values():
            # We only consider entities that need masking and are the right type
            if not entity.need_masking:
                continue
            elif entity.is_direct and not include_direct:
                continue
            elif not entity.is_direct and not include_quasi:
                continue  
            to_mask.append(entity)
                
        return to_mask      
        
    def get_annotators_for_span(self, start_token:int, end_token:int) -> Set[str]:
        """Given a text span (typically for a token), determines which annotators
        have also decided to mask it.

        Args:
            start_token (int): The starting token index of the span.
            end_token (int): The ending token index of the span.

        Returns:
            Set[str]: A (possibly empty) set of annotator names that have masked that span.
        """
        
        # We compute an interval tree for fast retrieval
        if not hasattr(self, "masked_spans"):
            self.masked_spans = intervaltree.IntervalTree()
            for entity in self.gold_annotated_entities.values():
                if entity.need_masking:
                    for i, (start, end) in enumerate(entity.mentions):
                        if entity.mention_level_masking[i]:
                            self.masked_spans[start:end] = entity.annotator
        
        annotators = set()      
        for mention_start, mention_end, annotator in self.masked_spans[start_token:end_token]:            
            # We require that the span is fully covered by the annotator
            if mention_start <=start_token and mention_end >= end_token:
                annotators.add(annotator)
                    
        return annotators

    def split_by_tokens(self, start:int, end:int) -> Iterator[Tuple[int, int]]:
        """
        Generates the (start, end) boundaries of each token included in this span.

        Args:
            start (int): The starting index of the span.
            end (int): The ending index of the span.

        Returns:
            Iterator[Tuple[int, int]]: An iterator of (start, end) tuples for each token.
        """    

        for match in re.finditer(r"\w+", self.text[start:end]):
            start_token = start + match.start(0)
            end_token = start + match.end(0)
            yield start_token, end_token

    #endregion

class TokenWeighting:
    """Abstract class for token weighting schemes (i.e., `ICTokenWeighting` and `UniformTokenWeighting`)"""

    @abc.abstractmethod
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]) -> np.ndarray:
        """Given a text and a list of text spans, returns a NumPy array of numeric weights
        (of same length as the list of spans) corresponding to each span.

        Args:
            text (str): The input text.
            text_spans (List[Tuple[int,int]]): A list of text spans, where each span
                is represented as a tuple of (start_index, end_index).

        Returns:
            np.ndarray: A NumPy array of numeric weights, with the same length as
            `text_spans`.
        """

        return

class ICTokenWeighting(TokenWeighting):
    """Token weighting based on a BERT language model. 
    The weighting mechanism runs the model on a text in which the provided spans are masked. 
    The weight of each token is then defined as its information content:
    -log(probability of the actual token value)
    
    In other words, a token that is difficult to predict will have a high
    information content, and therefore a high weight, whereas a token which can
    be predicted from its content will received a low weight (closer to zero)"""

    max_segment_length:int
    model_name:str
    device:str

    model=None
    tokenizer=None
    
    def __init__(self, model_name:str, device:str, max_segment_length:int):
        """Initializes the `ICTokenWeighting`

        Args:
            model_name (str): The name of the BERT model to use (e.g., "bert-base-uncased").
            device (str): The device to run the model on (e.g., "cpu" or "cuda").
            max_segment_length (int): The maximum sequence length for the model.
        """

        self.max_segment_length = max_segment_length
        self.model_name = model_name
        self.device = device
    
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]) -> np.ndarray:
        """Returns an array of numeric information content weights, where each value
        corresponds to -log(probability of predicting the value of the text span
        according to the BERT model).

        If the span corresponds to several BERT tokens, the probability is the
        minimum of the probabilities for each token.

        Args:
            text (str): The input text.
            text_spans (List[Tuple[int,int]]): A list of text spans, where each span
                is represented as a tuple of (start_index, end_index).

        Returns:
            np.ndarray: A NumPy array of numeric weights, with the same length as
            `text_spans`. A weight close to 0 represents a span with low information
            content (i.e. which can be easily predicted from the remaining context),
            while a higher weight represents a high information content (which is
            difficult to predict from the context).
        """

        # Create model if it is not already created
        if self.model is None:
            self._create_model()
        
        # Prepare inputs for predictions
        input_ids, attention_mask, tokens_by_span, input_ids_seq = self._prepare_input(text, text_spans)
          
        # Run the masked language model     
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Obtain the probabilities for the actual tokens
        probs = self._logits_to_probs(logits, input_ids_seq)
        
        # Transform the probabilities into weights with -log(probability)
        weights = self._probs_to_weights(probs, text_spans, tokens_by_span)
        
        return weights
    
    def get_weights_batched(self, texts:List[str], texts_spans:List[List[Tuple[int, int]]]) -> List[np.ndarray]:
        """Returns a list of arrays of numeric information content weights for multiple texts.
        
        Each array corresponds to -log(probability of predicting the value of the text spans
        according to the BERT model) for the corresponding text.

        If a span corresponds to several BERT tokens, the probability is the
        minimum of the probabilities for each token.

        Args:
            texts (List[str]): A list of input texts.
            texts_spans (List[List[Tuple[int,int]]]): A list of text span lists, where 
                each inner list contains spans for the corresponding text. Each span is 
                represented as a tuple of (start_index, end_index).

        Returns:
            List[np.ndarray]: A list of NumPy arrays of numeric weights, with the same length as
                `texts`. Each array has the same length as the corresponding `text_spans` list.
                A weight close to 0 represents a span with low information content, while a 
                higher weight represents high information content.
        """
        
        # Create model if it is not already created
        if self.model is None:
            self._create_model()
        
        # Prepare inputs for all texts in the batch
        batch_input_ids = []
        batch_attention_masks = []
        batch_tokens_by_span = []
        batch_input_ids_seq = []        
        for text, text_spans in zip(texts, texts_spans):
            input_ids, attention_mask, tokens_by_span, input_ids_seq = self._prepare_input(text, text_spans)
            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_tokens_by_span.append(tokens_by_span)
            batch_input_ids_seq.append(input_ids_seq)
        
        # Stack inputs for batch processing (assuming they can be batched)
        # Note: This assumes all sequences have the same length after padding
        if len(batch_input_ids) == 1:
            # For single text, preserve original dimensions
            stacked_input_ids = batch_input_ids[0]
            stacked_attention_masks = batch_attention_masks[0]
        else:
            # Concatenate all inputs along the first dimension
            # This turns [(2,100), (2,100), (2,100)] into (6,100) for 3 texts with 2 masks each
            stacked_input_ids = torch.cat(batch_input_ids, dim=0)
            stacked_attention_masks = torch.cat(batch_attention_masks, dim=0)
        
        # Run the masked language model on the entire batch
        batch_logits = self.model(input_ids=stacked_input_ids, attention_mask=stacked_attention_masks).logits
        
        # Process each item in the batch
        batch_weights = []
        segment_idx = 0
        for input_ids, input_ids_seq, text_spans, tokens_by_span in zip(batch_input_ids, batch_input_ids_seq, texts_spans, batch_tokens_by_span):
            # Extract segments logits for this specific text
            text_n_segments = len(input_ids)
            logits = batch_logits[segment_idx:segment_idx+text_n_segments]
            
            # Increment segment_idx
            segment_idx += text_n_segments

            # Obtain the probabilities for the actual tokens
            probs = self._logits_to_probs(logits, input_ids_seq)
            
            # Transform the probabilities into weights with -log(probability)
            weights = self._probs_to_weights(probs, text_spans, tokens_by_span)
            batch_weights.append(weights)
        
        return batch_weights

    def get_weights_batched_chunked(self, texts:List[str], texts_spans:List[List[Tuple[int, int]]], 
                                batch_size:int=IC_WEIGHTING_BATCH_SIZE) -> List[np.ndarray]:
        """Optimized batched version that processes texts in smaller chunks if needed.
        
        This version is useful when you have a large number of texts and want to control
        memory usage by processing them in smaller batches.
        
        Args:
            texts (List[str]): A list of input texts.
            texts_spans (List[List[Tuple[int,int]]]): A list of text span lists.
            batch_size (int): Number of texts/text_spans pair to process simultaneously.
            
        Returns:
            List[np.ndarray]: A list of NumPy arrays of numeric weights.
        """
        
        all_weights = []
        
        # Process in chunks (automatically handles remainder when len(texts) % batch_size != 0)
        for i in range(0, len(texts), batch_size):
            chunk_texts = texts[i:i+batch_size]  # Last chunk may be smaller than batch_size
            chunk_spans = texts_spans[i:i+batch_size]
            
            chunk_weights = self.get_weights_batched(chunk_texts, chunk_spans)
            all_weights += chunk_weights
        
        return all_weights
    
    def _create_model(self):
        """
        Initializes the BERT model and tokenizer from the pre-trained model name.
        Only executed the first time the get_weights method is invoked.
        """

        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def _prepare_input(self, text:str, text_spans:List[Tuple[int,int]]):
        # Tokenise the text
        bert_tokens = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = bert_tokens["input_ids"]
        n_tokens = len(input_ids) # Total number of tokens
        input_ids_seq = np.array(input_ids) # Consecutive tokens, without segments
        
        # Record the mapping between spans and BERT tokens
        bert_token_spans = bert_tokens["offset_mapping"]
        tokens_by_span = self._get_tokens_by_span(bert_token_spans, text_spans, text)

        # Mask the tokens that we wish to predict
        attention_mask = bert_tokens["attention_mask"]
        for token_indices in tokens_by_span.values():
            for token_idx in token_indices:
                attention_mask[token_idx] = 0
                input_ids[token_idx] = self.tokenizer.mask_token_id
        
        # Upload to device
        input_ids = torch.tensor(input_ids)[None,:].to(self.device)
        attention_mask = torch.tensor(attention_mask)[None,:].to(self.device)
        
        # Split into segments of size max_segment_length
        n_segments = math.ceil(n_tokens/self.max_segment_length)
        
        # Split the input_ids (and add padding if necessary)
        split_pos = [self.max_segment_length * (i + 1) for i in range(n_segments - 1)]
        input_ids_splits = torch.tensor_split(input_ids[0], split_pos)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_splits, batch_first=True)
        
        # Split the attention masks
        attention_mask_splits = torch.tensor_split(attention_mask[0], split_pos)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_splits, batch_first=True)        
        
        return input_ids, attention_mask, tokens_by_span, input_ids_seq

    def _get_tokens_by_span(self, bert_token_spans:List[Tuple[int,int]],
                            text_spans:List[Tuple[int,int]], text:str) -> Dict[Tuple[int,int],List[int]]:
        """Given two lists of spans (one with the spans of the BERT tokens, and one with
        the text spans to weight), returns a dictionary where each text span is associated
        with the indices of the BERT tokens it corresponds to.

        Args:
            bert_token_spans (List[Tuple[int,int]]): A list of tuples, where each tuple
                represents the (start_index, end_index) of a BERT token within the text.
            text_spans (List[Tuple[int,int]]): A list of tuples, where each tuple
                represents the (start_index, end_index) of a text span to be weighted.
            text (str): The original text.

        Returns:
            Dict[Tuple[int,int],List[int]]: A dictionary where keys are text spans
            (tuples of start and end indices) and values are lists of BERT token indices
            that fall within the respective text span.
        """
        
        # We create an interval tree to facilitate the mapping
        text_spans_tree = intervaltree.IntervalTree()
        for start, end in text_spans:
            text_spans_tree[start:end] = True
        
        # We create the actual mapping between spans and tokens
        tokens_by_span = {span:[] for span in text_spans}
        for token_idx, (start, end) in enumerate(bert_token_spans):
            for span_start, span_end, _ in text_spans_tree[start:end]:
                tokens_by_span[(span_start, span_end)].append(token_idx) 
        
        # And control that everything is correct
        for span_start, span_end in text_spans:
            if len(tokens_by_span[(span_start, span_end)]) == 0:
                logging.warning(f"Span ({span_start},{span_end}) without any token [{repr(text[span_start:span_end])}]")
        
        return tokens_by_span
    
    def _logits_to_probs(self, logits:torch.Tensor,
                         input_ids_seq:torch.Tensor) -> np.ndarray:
        # If the batch contains several segments, concatenate the result
        if len(logits) > 1:
            logits = torch.vstack([logits[i] for i in range(len(logits))])
            logits = logits[:len(input_ids_seq)]
        else:
            logits = logits[0]
        
        # Transform logits into probabilities
        unnorm_probs = torch.exp(logits)
        probs = unnorm_probs / torch.sum(unnorm_probs, axis=1)[:,None]

        # We are only interested in the probs for the actual token values
        probs_actual = probs[torch.arange(len(input_ids_seq)), input_ids_seq]
        probs_actual = probs_actual.detach().cpu().numpy()

        return probs_actual

    def _probs_to_weights(self, probs:np.ndarray, text_spans:List[Tuple[int,int]],
                           tokens_by_span:Dict[Tuple[int,int],List[int]]) -> np.ndarray:
        # Compute the weights from those predictions
        weights = []
        for (span_start, span_end) in text_spans:
            
            # If the span does not include any actual token, skip
            if not tokens_by_span[(span_start, span_end)]:
                weights.append(0)
                continue
            
            # If the span has several tokens, we take the minimum prob
            prob = np.min([probs[token_idx] for token_idx in 
                           tokens_by_span[(span_start, span_end)]])
            
            # We finally define the weight as -log(p)
            weights.append(-np.log(prob))
        
        weights = np.array(weights) # Transform to NumPy array

        return weights

    def __del__(self):
        """Method invoked when deleting the instance to dispose the model and the tokenizer
        (if these are already defined)"""

        if not self.model is None:
            del self.model
        if not self.tokenizer is None:
            del self.tokenizer
        if not gc is None:
            gc.collect()
        if not torch is None and torch.cuda.is_available():
            torch.cuda.empty_cache()

class UniformTokenWeighting(TokenWeighting):
    """Uniform weighting (all tokens assigned to a weight of 1.0)"""
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]) -> np.ndarray:
        """Given a text and a list of text spans, returns a NumPy array of uniform weights.

        Args:
            text (str): The input text.
            text_spans (List[Tuple[int,int]]): A list of text spans, where each span
                is represented as a tuple of (start_index, end_index).

        Returns:
            np.ndarray: A NumPy array with all weights set to 1.0, with the same length
                as `text_spans`.
        """

        return np.ones(len(text_spans))

#endregion
