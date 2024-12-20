from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
import boto3

# %pip install -qU pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
    Currently, I am using the 
"""
class FAISSVectorStoreIngestor:
    def __init__(self, 
                 embedding_model_id="amazon.titan-embed-text-v1", 
                 chunk_size=2000, 
                 chunk_overlap=400, 
                 separator=",",
                 boto3_bedrock_client=None,
                 faiss_index_path=None):
        """
        Initializes the FAISSVectorStoreIngestor.

        Parameters:
        embedding_model_id (str): The ID of the Bedrock embedding model.
        chunk_size (int): The size of text chunks for splitting documents.
        chunk_overlap (int): The overlap between text chunks.
        separator (str): The separator used for splitting.
        boto3_bedrock_client: Pre-configured AWS Bedrock client.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

        self.faiss_index_path = faiss_index_path
        
        self.embeddings_model_id = embedding_model_id
        self.boto3_bedrock_client = boto3_bedrock_client
        
        self.embeddings = BedrockEmbeddings(
            model_id=self.embeddings_model_id,
            client=self.boto3_bedrock_client
        )

    def ingest_csv_to_vectorstore(self, csv_path):
        """
        Processes a CSV document and ingests it into an FAISS vector store.

        Parameters:
        csv_path (str): Path to the CSV file (local or S3).
        existing_faiss_path (str): Path to an existing FAISS index file, if any.

        Returns:
        FAISS: The FAISS vector store containing ingested document embeddings.
        """
        # Load documents
        loader = CSVLoader(csv_path)
        documents = loader.load()

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap, 
            separator=self.separator
        )
        split_docs = text_splitter.split_documents(documents)

        # Load or create FAISS vector store
        if self.faiss_index_path and os.path.exists(self.faiss_index_path):
            vectorstore = self.load_faiss_index() 
            vectorstore.add_documents(split_docs)
        else:
            print("Creating index for the first time ")
            vectorstore = FAISS.from_documents(
                documents=split_docs, 
                embedding=self.embeddings
            )
        return vectorstore


    # https://python.langchain.com/v0.2/docs/tutorials/pdf_qa/
    def ingest_pdf_to_vectorstore(self, file_paths):
        """
        Ingest an array of PDF files into the FAISS vector store.

        Parameters:
        file_paths (list of str): Paths to the PDF files to ingest.

        Returns:
        FAISS: The FAISS vector store containing ingested document embeddings.
        """

        all_splits = []  # List to hold all the document splits

        for file_path in file_paths:
            # Load each PDF file
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)  # Add the splits to the master list

        # Load or create FAISS vector store
        if self.faiss_index_path and os.path.exists(self.faiss_index_path):
            vectorstore = self.load_faiss_index() 
            vectorstore.add_documents(split_docs)
        else:
            print("Creating index for the first time ")
            # Create FAISS vector store from all splits
            vectorstore = FAISS.from_documents(documents=all_splits, embedding=self.embeddings)

        self.vectorstore = vectorstore
        return vectorstore
    
   

    def save_faiss_index(self, vectorstore):
        """
        Saves the FAISS index to a local path.

        Parameters:
        vectorstore (FAISS): The FAISS vector store to save.
        save_path (str): Path to save the FAISS index.
        """
        vectorstore.save_local(self.faiss_index_path)

    def load_faiss_index(self):
        """
        Loads a FAISS index from a local path.

        Parameters:
        faiss_path (str): Path to the FAISS index file.

        Returns:
        FAISS: The loaded FAISS vector store.
        """
        print("Creating index for the first time ")
        return FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)



if __name__ == "__main__":
    import os

    # Paths
    s3_path = "s3://jumpstart-cache-prod-us-east-2/training-datasets/Amazon_SageMaker_FAQs/Amazon_SageMaker_FAQs.csv"
    local_csv_path = "./rag_data/Amazon_SageMaker_FAQs.csv"
    faiss_index_path = "./faiss_index"

    # Download the file from S3
    os.system(f"aws s3 cp {s3_path} {local_csv_path}")

    # Instantiate the ingestor
    ingestor = FAISSVectorStoreIngestor(boto3_bedrock_client=boto3_bedrock, faiss_index_path=faiss_index_path)

    # Check if there's an existing FAISS index to load
    vectorstore = ingestor.ingest_csv_to_vectorstore(local_csv_path)

    # Save the updated FAISS index
    ingestor.save_faiss_index(vectorstore)
    
    print(f"FAISS Index contains {vectorstore.index.ntotal} elements.")

    # Load existing index and add documents to it 


    #from faiss_ingestor import FAISSVectorStoreIngestor  # Assuming the class is in `faiss_ingestor.py`
    # Paths
    # faiss_index_path = "./faiss_index"  # Path where the FAISS index is saved

    # # Instantiate the ingestor
    # ingestor = FAISSVectorStoreIngestor()

    # # Load the FAISS index
    # vectorstore = ingestor.load_faiss_index()

    # # Print the total number of elements in the FAISS index
    # print(f"Loaded FAISS Index contains {vectorstore.index.ntotal} elements.")
