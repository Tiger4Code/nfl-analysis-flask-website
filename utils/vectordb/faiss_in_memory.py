from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
import boto3

class FAISSVectorStoreIngestor:
    def __init__(self, 
                 embedding_model_id="amazon.titan-embed-text-v1", 
                 chunk_size=2000, 
                 chunk_overlap=400, 
                 separator=",",
                 boto3_bedrock_client=None):
        """
        Initializes the FAISSVectorStoreIngestor.

        Parameters:
        embedding_model_id (str): The ID of the Bedrock embedding model.
        chunk_size (int): The size of text chunks for splitting documents.
        chunk_overlap (int): The overlap between text chunks.
        separator (str): The separator used for splitting.
        boto3_bedrock_client: Pre-configured AWS Bedrock client.
        """
        # embeddings_model_id="amazon.titan-embed-text-v1"
        # br_embeddings = BedrockEmbeddings(model_id=embeddings_model_id, client=boto3_bedrock)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        
        self.embeddings_model_id = embedding_model_id
        self.boto3_bedrock_client = boto3_bedrock_client
        
        self.embeddings = BedrockEmbeddings(
            model_id=self.embeddings_model_id,
            client=self.boto3_bedrock_client
        )

        self.vectorstore = None 

    # https://python.langchain.com/docs/how_to/document_loader_pdf/
    # async def async_ingest_pdf_to_vectorstore(self, file_path):
    #     # %pip install -qU pypdf
    #     from langchain_community.document_loaders import PyPDFLoader

    #     loader = PyPDFLoader(file_path)
    #     pages = []
    #     async for page in loader.alazy_load():
    #         pages.append(page)

    #     print(f"{pages[0].metadata}\n")
    #     print(pages[0].page_content)

    #     # from langchain_core.vectorstores import InMemoryVectorStore
    #     # vectorstore = InMemoryVectorStore.from_documents(pages, embedding=self.embeddings)
    #     vectorstore = FAISS.from_documents(pages, embedding=self.embeddings)
    #     docs = vectorstore.similarity_search("What is Alpha Health?", k=2)
    #     for doc in docs:
    #         print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')

    #     self.vectorstore = vectorstore
    #     return vectorstore
    
    # https://python.langchain.com/v0.2/docs/tutorials/pdf_qa/
    def ingest_pdf_to_vectorstore(self, file_paths):
        """
        Ingest an array of PDF files into the FAISS vector store.

        Parameters:
        file_paths (list of str): Paths to the PDF files to ingest.

        Returns:
        FAISS: The FAISS vector store containing ingested document embeddings.
        """
        # %pip install -qU pypdf
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        all_splits = []  # List to hold all the document splits

        for file_path in file_paths:
            # Load each PDF file
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)  # Add the splits to the master list

        # Create FAISS vector store from all splits
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=self.embeddings)

        self.vectorstore = vectorstore
        return vectorstore
    
    
    def search(self, text):
        print(" ----------------- Similarity Search Function -----------------")
        docs = self.vectorstore.similarity_search(text, k=2)
        for doc in docs:
            print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')
            

    def ingest_csv_to_vectorstore(self, csv_path):
        """
        Processes a CSV document and ingests it into an in-memory FAISS vector store.

        Parameters:
        csv_path (str): Path to the CSV file (local or S3).

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

        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(
            documents=split_docs, 
            embedding=self.embeddings
        )
        return self.vectorstore

# Usage Example
# if __name__ == "__main__":
#     s3_path = "s3://jumpstart-cache-prod-us-east-2/training-datasets/Amazon_SageMaker_FAQs/Amazon_SageMaker_FAQs.csv"
#     local_path = "./rag_data/Amazon_SageMaker_FAQs.csv"

#     # Download the file from S3
#     !aws s3 cp $s3_path $local_path

#     # Instantiate the ingestor
#     ingestor = FAISSVectorStoreIngestor(boto3_bedrock_client=boto3_bedrock)

#     # Ingest the CSV into FAISS
#     vectorstore_faiss_aws = ingestor.ingest_csv_to_vectorstore(local_path)


#     # Ingest the PDF into FAISS
#     vectorstore_faiss_aws = ingestor.ingest_pdf_to_vectorstore(local_path)

#     print(f"vectorstore_faiss_aws: number of elements in the index={vectorstore_faiss_aws.index.ntotal}")
