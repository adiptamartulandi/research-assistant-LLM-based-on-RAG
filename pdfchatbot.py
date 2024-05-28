import yaml
import fitz
import torch
import gradio as gr
import weaviate
import os
from PIL import Image
from config import MODEL_CONFIG
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class PDFChatBot:
    def __init__(self):
        """
        Initialize the PDFChatBot instance.
        """
        self.processed = False
        self.page = 0
        self.chat_history = []
        # Initialize other attributes to None
        self.prompt = None
        self.documents = None
        self.embeddings = None
        self.vectordb = None
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.chain = None

    def add_text(self, history, text):
        """
        Add user-entered text to the chat history.

        Parameters:
            history (list): List of chat history tuples.
            text (str): User-entered text.

        Returns:
            list: Updated chat history.
        """
        if not text:
            raise gr.Error('Enter text')
        history.append((text, ''))
        return history

    def create_prompt_template(self):
        """
        Create a prompt template for the chatbot.
        """
        template = """
        You are an AI Assistant that help user answer question from user.
        Combine the chat history and follow up question into a standalone question.
        
        Chat History: {chat_history}
        Question: {question}
        Answer: """
        self.prompt = PromptTemplate.from_template(template)

    def load_embeddings(self):
        """
        Load embeddings from Hugging Face and set in the config file.
        """
        self.embeddings = OpenAIEmbeddings(model=MODEL_CONFIG.MODEL_EMBEDDINGS)

    def load_vectordb(self):
        """
        Load the vector database from the documents and embeddings.
        """
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(self.documents)

        weaviate_client = weaviate.connect_to_wcs(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
        )
        
        self.vectordb = WeaviateVectorStore.from_documents(docs, self.embeddings, client=weaviate_client)


    def create_chain(self):
        """
        Create a Conversational Retrieval Chain
        """
        llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(search_kwargs={"k": 1}),
            condense_question_prompt=self.prompt,
            return_source_documents=True
        )

    def process_file(self, file):
        """
        Process the uploaded PDF file and initialize necessary components: Tokenizer, VectorDB and LLM.

        Parameters:
            file (FileStorage): The uploaded PDF file.
        """
        self.create_prompt_template()
        self.documents = PyPDFLoader(file.name).load()
        self.load_embeddings()
        self.load_vectordb()
        self.create_chain()

    def generate_response(self, history, query, file):
        """
        Generate a response based on user query and chat history.

        Parameters:
            history (list): List of chat history tuples.
            query (str): User's query.
            file (FileStorage): The uploaded PDF file.

        Returns:
            tuple: Updated chat history and a space.
        """
        if not query:
            raise gr.Error(message='Submit a question')
        if not file:
            raise gr.Error(message='Upload a PDF')
        if not self.processed:
            self.process_file(file)
            self.processed = True

        result = self.chain({"question": query, 'chat_history': self.chat_history}, return_only_outputs=True)
        self.chat_history.append((query, result["answer"]))
        self.page = 0

        for char in result['answer']:
            history[-1][-1] += char
        return history, " "

    def render_file(self, file):
        """
        Renders a specific page of a PDF file as an image.

        Parameters:
            file (FileStorage): The PDF file.

        Returns:
            PIL.Image.Image: The rendered page as an image.
        """
        doc = fitz.open(file.name)
        page = doc[self.page]
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image