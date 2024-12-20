from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
import boto3
import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

FILE_STORAGE_MODE = 'file'
S3_STORAGE_MODE = 's3'

class FAISSVectorStoreIngestorDisc:
    def __init__(self, 
                 embedding_model_id="amazon.titan-embed-text-v1", 
                 chunk_size=2000, 
                 chunk_overlap=400, 
                 separator=",",
                 boto3_bedrock_client=None,
                 faiss_index_path=None,
                 s3_bucket_name=None, 
                 storage_mode="s3"):
        """
        Initializes the FAISSVectorStoreIngestor.

        Parameters:
        embedding_model_id (str): The ID of the Bedrock embedding model.
        chunk_size (int): The size of text chunks for splitting documents.
        chunk_overlap (int): The overlap between text chunks.
        separator (str): The separator used for splitting.
        boto3_bedrock_client: Pre-configured AWS Bedrock client.
        faiss_index_path (str): Local path for storing/loading FAISS index.
        s3_bucket_name (str): S3 bucket name for storing/loading FAISS index.
        storage_mode (str): S3 for s3 file store and load; file for local file storage 
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

        self.faiss_index_path = faiss_index_path
        self.s3_bucket_name = s3_bucket_name

        self.embeddings_model_id = embedding_model_id
        self.boto3_bedrock_client = boto3_bedrock_client

        self.embeddings = BedrockEmbeddings(
            model_id=self.embeddings_model_id,
            client=self.boto3_bedrock_client
        )
        self.s3_client = boto3.client('s3')
        self.storage_mode = storage_mode

        # Init vectorstore variable
        self.vectorstore = None 

    def ingest_csv_to_vectorstore(self, csv_path):
        """
        Processes a CSV document and ingests it into an FAISS vector store.

        Parameters:
        csv_path (str): Path to the CSV file (local or S3).

        Returns:
        FAISS: The FAISS vector store containing ingested document embeddings.
        """
        loader = CSVLoader(csv_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap, 
            separator=self.separator
        )
        split_docs = text_splitter.split_documents(documents)

        self.vectorstore = self._load_or_create_faiss_index(split_docs)
        return self.vectorstore

    def ingest_pdf_to_vectorstore(self, file_paths):
        """
        Ingest an array of PDF files into the FAISS vector store.

        Parameters:
        file_paths (list of str): Paths to the PDF files to ingest.

        Returns:
        FAISS: The FAISS vector store containing ingested document embeddings.
        """
        all_splits = []
        for file_path in file_paths:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)

        self.vectorstore = self._load_or_create_faiss_index(all_splits)

        return self.vectorstore

    def save_faiss_index(self, vectorstore):
        """
        Saves the FAISS index locally and optionally to an S3 bucket.

        Parameters:
        vectorstore (FAISS): The FAISS vector store to save.
        """
        if self.storage_mode == FILE_STORAGE_MODE and self.faiss_index_path:
            vectorstore.save_local(self.faiss_index_path)
            print(f"Index saved locally at {self.faiss_index_path}")

        elif self.storage_mode == S3_STORAGE_MODE and self.s3_bucket_name:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, "faiss_index")
                vectorstore.save_local(temp_path)

                for root, _, files in os.walk(temp_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        s3_key = os.path.relpath(file_path, temp_path)
                        self.s3_client.upload_file(file_path, self.s3_bucket_name, s3_key)
                        print(f"Uploaded {s3_key} to S3 bucket {self.s3_bucket_name}")

    def load_faiss_index(self):
        """
        Loads a FAISS index from a local path or an S3 bucket.

        Returns:
        FAISS: The loaded FAISS vector store.
        """
        if self.storage_mode == FILE_STORAGE_MODE and self.faiss_index_path and os.path.exists(self.faiss_index_path):
            print("Loading index from local path")
            return FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)

        elif self.storage_mode == S3_STORAGE_MODE and self.s3_bucket_name:
            print("Load index from s3 bucket ")
            with tempfile.TemporaryDirectory() as temp_dir:
                objects = self.s3_client.list_objects_v2(Bucket=self.s3_bucket_name).get('Contents', [])
                if not objects:
                    raise FileNotFoundError("No index files found in S3 bucket")

                for obj in objects:
                    s3_key = obj['Key']
                    local_file_path = os.path.join(temp_dir, s3_key)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    self.s3_client.download_file(self.s3_bucket_name, s3_key, local_file_path)
                return FAISS.load_local(temp_dir, self.embeddings, allow_dangerous_deserialization=True)

        raise FileNotFoundError("No local or S3 FAISS index found")

    def _load_or_create_faiss_index(self, documents):
        """
        Helper method to load or create a FAISS index.

        Parameters:
        documents (list): The documents to add to the index if creating a new one.

        Returns:
        FAISS: The FAISS vector store.
        """
        try:
            if self.vectorstore is None: 
                self.vectorstore = self.load_faiss_index()
            self.vectorstore.add_documents(documents)

        except FileNotFoundError:
            print("Creating index for the first time")
            self.vectorstore = FAISS.from_documents(documents=documents, embedding=self.embeddings)

        return self.vectorstore

    def search(self, text, k=2):
        """
        Perform a similarity search on the vector store.

        Parameters:
        text (str): The input text to search for.
        k (int): The number of top results to return.

        Returns:
        None: Prints the search results.
        """
        print(" ----------------- Similarity Search Function -----------------")
        try:
            if self.vectorstore is None:
                self.vectorstore = self.load_faiss_index()
            docs = self.vectorstore.similarity_search(text, k=k)
            for doc in docs:
                print(f'Page {doc.metadata.get("page", "unknown")}: {doc.page_content[:300]}\n')
        except AttributeError:
            print("Vector store is not initialized. Please load or create it first.")
