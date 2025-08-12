

from langchain_community.document_loaders import DirectoryLoader, MergedDataLoader
import glob


from enum import Enum
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader, WebBaseLoader, unstructured, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain import hub
from langgraph.graph import START, StateGraph
from typing import Literal
from typing_extensions import Annotated
from langchain_core.documents import Document

from PIL import Image

from pydantic import BaseModel, Field
from typing import Literal, List, TypedDict

from pymilvus import FieldSchema, CollectionSchema, DataType

schema_fields = [

    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=769,  # <-- change to your embedding dimension
    ),
    
]

class Search(BaseModel):
    query: str = Field(..., description="Search query to run.")
    section: Literal["beginning", "middle", "end"] = Field(..., description="Section to query.")


class Excelsheet():
    def __init__(self):
        pass

    def __call__(self, path):
        loader = UnstructuredExcelLoader(path, mode='elements')
        return loader

class PDF():
    def __init__(self):
        pass

    def __call__(self, path):
        loader = UnstructuredPDFLoader(path, mode='elements')
        return loader

class Docx():
    def __init__(self):
        pass

    def __call__(self, path):
        loader = UnstructuredWordDocumentLoader(path, mode='elements')
        return loader

class Web():
    def __init__(self):
        pass

    async def __call__(self, paths: list):
        loader = WebBaseLoader(paths)

        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)

        return docs

class FileType(Enum): 
    excel = 1
    pdf  = 2
    docx = 3
    web = 4 
    
class FilesHandler():

    def __init__(self, file_type, file_path):
        try:
            self.loader = self.files_seletor(file_type)()(file_path)
        except KeyError as e: 
            raise ValueError(f"file type is not supprt: {e}")

    def files_seletor(self,file_type: FileType): 
        files_type_selctor = {
            FileType.excel: Excelsheet,
            FileType.pdf: PDF, 
            FileType.docx: Docx, 
            FileType.web: Web
        }
        return files_type_selctor[file_type]


class Search(BaseModel):
    query: str = Field(..., description="Search query to run.")
    section: Literal["beginning", "middle", "end"] = Field(..., description="Section to query.")


class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

class LangChainRAG():
    def __init__(self, vector_store:Milvus = None):
        self.extensions = ['.xlsx', '.docx', '.pdf', '.html']

        self.llm_embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        if vector_store: 
            self.vector_store = vector_store
        else: 
            self.vector_store = Milvus(
                embedding_function=self.llm_embedding,
                connection_args={"uri": './milvus_demo2.db'},
                index_params={"index_type": "FLAT", "metric_type": "L2"},
                # vector_schema= schema_fields,
                # vector_field='embedding',
                # primary_field="pk",
                collection_name="rag_collection",
                # auto_id=True,
                # text_field="text",
                # metadata_field="metadata",
        )
        
        # downlaod promot tmeplate realte to rag
        self.prompt = hub.pull("rlm/rag-prompt")

    def _add_extension(self, extn):
        self.extensions.append(f'.{extn}')

    def load_documents(self, dirc='./data'):
        extens = [f"*{x}" for x in self.extensions]

        docs_loader = DirectoryLoader(dirc, glob=extens, show_progress=True)

        return docs_loader
    
    def load_single_document(self, file_type):
        pass

    def doc_splitter(self, docs, chunck_size=2024, chunk_ovrlap=200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunck_size, chunk_overlap=chunk_ovrlap)
        if isinstance(docs, list):
            texts = text_splitter.split_documents(docs)
        else:
            texts = text_splitter.split_text(docs)

        return texts
    
    def add_embedding(self,texts_splits):
        return self.vector_store.add_documents(texts_splits)
    
    def add_metadata(self, texts_splits):
        total_splits = len(texts_splits)
        third = total_splits//3

        for i, document in enumerate(texts_splits):
            if i < third:
                document.metadata["section"] = "beginning"
            elif i < 2 * third:
                document.metadata["section"] = "middle"
            else:
                document.metadata["section"] = "end"
        
        return texts_splits

    def analyze_query(self, state: State):
        structured_llm = llm.with_structured_output(Search)
        query = structured_llm.invoke(state["question"])
        return {"query": query}
    
    def retrieve(self,state: State):
        query = state["query"]
        retrieved_docs = self.vector_store.similarity_search(
            query.query,
            expr=f"section == '{query.section}'",
        )
        return {"context": retrieved_docs}
    
    def generate(self,state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    def build_graph(self):
        graph_builder = StateGraph(State).add_sequence([self.analyze_query, self.retrieve, self.generate])
        graph_builder.add_edge(START, "analyze_query")
        graph = graph_builder.compile()

        return graph
    
    def save_graph_structure(self, graph):
        png_bytes = graph.get_graph().draw_mermaid_png()
        img = Image.open(io.BytesIO(png_bytes))  # now Image is the PIL class
        img.save("graph.png")

    

if __name__ == "__main__":

    rag = LangChainRAG()
    # print(glob.glob('./data/*.docx'))
    loader = FilesHandler(FileType.pdf, './data/1btcpp.pdf')
    doc = loader.loader.load()

    # loader = rag.load_documents('./data')
    # doucs = loader.load()
    doucs = rag.add_metadata(doc)
    # for d in doucs:
    #     print(d.metadata)
    doucs = rag.doc_splitter(doucs)
    # for doc in doucs: 
    #     print(doc)
    for doc in doucs: 
        doc.metadata['languages'] = str(doc.metadata)
    rag.add_embedding(doucs)
    graph = rag.build_graph()
    # # results = rag.vector_store.similarity_search_with_score('what is btcpp', k=1)
    # # all_keys = set()
    # # for doc in results:
    # #     all_keys.update(doc.metadata.keys())

    # # print("Metadata keys in the collection:", all_keys)

    rag.save_graph_structure(graph)
    for step in graph.stream(
        {"question": "expalin this line we expand TL to element-wise lookup table (ELUT) for low-bit LLMs in the appendix,?"},
        stream_mode="updates",
    ):
        print(f"{step}\n\n----------------\n")
