import os
import logging
from typing import List, Optional, Dict
from dotenv import load_dotenv

from src.retrieval.vector_store import get_vector_store
from src.retrieval.web_scrapper import get_web_scraper
from src.retrieval.embeddings import get_embedding_generator
from src.core.prioritizer import RetrievalChunk
from src.utils.token_counter import get_token_counter
#Will continue later football is starting