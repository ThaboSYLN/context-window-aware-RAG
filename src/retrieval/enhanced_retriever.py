import os
import logging
from typing import List, Optional, Dict
from dotenv import load_dotenv

from src.retrieval.vector_store import get_vector_store
from src.retrieval.web_scrapper import get_web_scraper
from src.retrieval.embeddings import get_embedding_generator
from src.core.prioritizer import RetrievalChunk
from src.utils.token_counter import get_token_counter

class EnhancedRetriever:
    """
    Docstring for EnhancedRetriever

    
    """

    def __init__(
            self,
            max_corpus_results:Optional[int] = None,
            max_web_results:Optional[int] = None,
            Similarity_threshhold:Optional[float] = None,
            web_trigger_threshold:Optional[float] = None
    ):
        load_dotenv()
        self.logger = logging.getLogger(__name__)
        self.max_corpus_results = max_corpus_results or int(
            os.getenv('MAX_RETRIEVAL_RESULTS','10')
        )
        self.max_web_results = max_web_results or 3 # fewer web result--Expensive
        self.Similarity_threshhold = Similarity_threshhold or float(
            os.getenv('SIMILARITY_THRESHOLD','0.3')
        )
        self.web_trigger_threshold = web_trigger_threshold or float(
            os.getenv('WEB_TRIGGER_THRESHOLD','0.5')
        )
     
        # Initioalizing the components

        self.vector_store = get_vector_store()
        self.web_scrapper = get_web_scraper()
        self.embedding_generator = get_embedding_generator
        self.token_counter = get_token_counter()

        self.logger.infor(
            f"Initialized EnhancedRetriever"
            f"(Corpus = {self.max_corpus_results}, web = {self.max_corpus_results},"
            f"threshold = {self.Similarity_threshhold}, web_trigger = {self.web_trigger_threshold})"
        )

    def _should_use_web(self,corpus_results:List[Dict],query:str)-> bool:
        """Will check later"""   

