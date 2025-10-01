"""

Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)

Design Pattern called Builder (https://refactoring.guru/design-patterns/builder/python/example#lang-features)

Design Pattern called Builder

"""


import faiss
import chromadb

import pandas as pd 

from tqdm import tqdm
from uuid import uuid4

from abc import ABC, abstractmethod

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

class BaseVectorStoreMixin(ABC):
    
    def set_embedding_model(self, embedding_model_name):
        supported_models = {
            "Hugging Face": "sentence-transformers/all-mpnet-base-v2"
        }
        if embedding_model_name in supported_models:
            if embedding_model_name == "Hugging Face":
                model_path = supported_models[embedding_model_name]
                self.embedding_model = HuggingFaceEmbeddings(model_name=model_path)
            elif embedding_model_name == "Ollama":
                model_path = supported_models[embedding_model_name]
                self.embedding_model = OllamaEmbeddings(model_name=model_path)
            return self.embedding_model
        else:
            raise ValueError(f"""
                Embedding Model Name: '{embedding_model_name}' is not yet supported.
                Choose from: {supported_models}
            """)

class BaseVectorStoreBuilder(BaseVectorStoreMixin):
    def __init__(self):
        self.vector_store = None
        self.documents = []
        self.uuids = None
        self.embedding_model = None
    
    def reset(self):
        self.__init__() # Resets all properties to their initial state
        print(f"\tVector Store: {self.vector_store}\n\tDocments: {self.documents}\n\tUUIDS: {self.uuids}\n\tEmbedding Model: {self.embedding_model}")
            
    def build_documents(self, df: pd.DataFrame):
        """(Abstract) Processes raw data into LangChain Documents."""
        metadata_cols = [col for col in df.columns if col != 'sentence']
        print(f"\tMetadata Columns: {metadata_cols}")
        
        for _, row in tqdm(df.iterrows()):
            # Correctly create metadata from just the current row
            doc_metadata = {col: row[col] for col in metadata_cols}
            
            document = Document(
                page_content=row['sentence'],
                metadata=doc_metadata
            )
            self.documents.append(document)
        
        # Generate unique IDs for the documents we just built
        self.uuids = [str(uuid4()) for _ in self.documents]
        print(f"\tUUIDS (N = D): {len(self.uuids)}")

    def add_documents_to_vector_store(self):
        """Correctly adds documents to the vector store product, not itself."""
        if not self.documents:
            print("Warning: No documents to add.")
            return
        # This is the fix for the recursion error.
        self.vector_store.add_documents(documents=self.documents, ids=self.uuids)

    def get_vector_store(self):
        """Shared implementation for returning the final product."""
        if not self.vector_store:
            raise ValueError("Product has not been built yet.")
        return self.vector_store
        
    @abstractmethod
    def initialize_vector_store(self):
        """(Abstract) Creates the initial vector store object."""
        pass

    @abstractmethod
    def load_vector_store(self):
        pass
    

class BaseVectorStoreLoader(BaseVectorStoreMixin):
    def __init__(self):
        self.client = None
    
    def reset(self):
        self.__init__() # Resets all properties to their initial state
    
    def load_vector_store(self):
        self.client = client
    
class ChromaVectorStore(BaseVectorStoreBuilder, BaseVectorStoreLoader):
    
    def __init__(self,
                 collection_name: str = None,
                 persist_directory: str = None
                ):
        self.client = None
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vector_store = None
        self.documents = []
        self.uuids = None
        self.embedding_model = None
        print(f"\tCollection Name: {self.collection_name}\n\tPersist Directory: {self.persist_directory}")
        print(f"\tVector Store: {self.vector_store}\n\tDocments: {self.documents}\n\tUUIDS: {self.uuids}\n\tEmbedding Model: {self.embedding_model}")
        
    def initialize_vector_store(self):
        if not self.embedding_model:
            raise ValueError("Embedding model must be set before initializing the vector store.")
        
        print(f"\tCollection Name: {self.collection_name}\n\tEmbedding Model: {self.embedding_model}\n\tPersist Directory: {self.persist_directory}")
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory
        )
        print(f"\tVector Store (Original): {self.vector_store}")
        
    def load_vector_store(self, client_type: str = None):
        if client_type == 'Persistence':
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            if not self.embedding_model:
                raise ValueError("Embedding model must be set before initializing the vector store.")
        
            print(f"\tCollection Name: {self.collection_name}\n\tEmbedding Model: {self.embedding_model}\n\tPersist Directory: {self.persist_directory}")
            
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
            )
            print(f"\tVector Store (Original): {self.vector_store}")
    
    def query_vector_store(self, query_string, k):
        print("\t1. Similarity")
        results = self.vector_store.similarity_search(query_string, k=k)
        for res in results:
            print(f"\t\t* {res.page_content} [{res.metadata}]\n")

        print("\t2. Similarity with score")
        results = self.vector_store.similarity_search_with_score(query_string, k=k)
        for res, score in results:
            print(f"\t\t* [SIM={score:3f}] {res.page_content} [{res.metadata}]\n")
        
        print("\t3. Similarity by vector")
        results = self.vector_store.similarity_search_by_vector(embedding=self.embedding_model.embed_query(query_string), k=k)
        for doc in results:
            print(f"\t\t* {doc.page_content} [{doc.metadata}]\n")
    
        print("\t4. Retriever")
        retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k})
        retriever.invoke(query_string)
        print(f"\t\t* {retriever}\n")
        

class VectorStoreDirector:
    def __init__(self, builder: BaseVectorStoreBuilder = None, loader: BaseVectorStoreLoader = None):
        """Initializes the director with a specific builder instance."""
        if builder:
            print(f"### BUILDER ###")
            self._builder = builder
            print(f"\t{self._builder}")
        
        if loader: 
            print(f"### LOADER ###")
            self._loader = loader
            print(f"\t{self._loader}")
    
    def construct(self, embedding_model_name, data):

        # print("### RESET ###")
        # self._builder.reset()
        
        print(f"### EMBEDDING MODEL ###")
        self._builder.set_embedding_model(embedding_model_name)
        print(f"\t{embedding_model_name}")
        
        print(f"### INITIALIZE VECTOR STORE ###")
        self._builder.initialize_vector_store()
        print(f"\tVector Store (Prediction's Wrapper): {self._builder}")
        
        print("### BUILD DOCUMENT ###""")
        self._builder.build_documents(data)
        print(f"\tDocuments (D) {len(self._builder.documents)}")
        
        print("### ADD DOCUMENTS TO VECTOR STORE ###")
        self._builder.add_documents_to_vector_store()
        print(f"\tDocuments added: {self._builder.vector_store}")
    
    def query(self, embedding_model_name, query_string, k):
        """Retrieves the final vector store from the builder."""
        
        print(f"### INITIALIZE CLIENT VECTOR STORE ###")
        self._loader.load_vector_store()
        print(f"\tVector Store (Prediction's Wrapper): {self._loader.client}")
        
        print(f"### LOAD EMBEDDING MODEL ###")
        self._loader.set_embedding_model(embedding_model_name)
        print(f"\t{embedding_model_name}")
        
        print(f"### LOAD VECTOR STORE ###")
        self._loader.initialize_vector_store()
        print(f"\tVector Store (Prediction's Wrapper): {self._loader}")
        
        print(f"### TOP K ###")
        results = self._loader.query_vector_store(query_string, k)       
        print(f"\tQuery Results): {results}")

        

    