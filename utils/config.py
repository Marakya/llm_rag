import os
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIR = os.getenv("PERSIST_DIR")
DOCSTORE_FILE = './docstore.pkl'
IMAGE_DIR = './figures'