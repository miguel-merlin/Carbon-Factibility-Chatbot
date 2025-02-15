from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GroqEmbeddings
from langchain.vectorstores import FAISS


pdf_loader = PyPDFLoader("path/to/node_document.pdf")
pdf_docs = pdf_loader.load()
md_loader = UnstructuredMarkdownLoader("path/to/node_document.md")
md_docs = md_loader.load()

all_docs = pdf_docs + md_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(all_docs)

embeddings = GroqEmbeddings("mixtral-8x7b-32768")
vectorstore = FAISS.from_documents(docs, embeddings)


def retrieve_docs(query, k=3):
    """Retrieve top-k relevant document chunks for a given query."""
    results = vectorstore.similarity_search(query, k=k)
    # Concatenate the content from the retrieved documents
    return "\n".join([doc.page_content for doc in results])


class Route(BaseModel):
    step: Literal["model_info", "model_params_node"] = Field(
        ..., description="Routing decision for the carbon mass capture model agent."
    )


llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

router_llm = llm.with_structured_output(Route)


class State(TypedDict):
    input: str  # The user's question
    decision: str  # Decision from the router node
    response: str  # Response from the model node


def router_node(state: State) -> dict:
    decision = router_llm.invoke(state["input"])
    return {"decision": decision.step}  # type: ignore


def route_decision(state: State) -> str:
    if state["decision"] == "model_info":
        return "model_info"
    elif state["decision"] == "model_inputs":
        return "model_params_node"
    return END


def model_info_node(state: dict) -> dict:
    """
    This node provides detailed information about the Carbon Mass Capture forecasting model.
    It augments its answer with relevant documentation retrieved from Markdown and PDF files.
    """
    retrieved_context = retrieve_docs(state["input"])
    prompt = (
        f"Using the following documentation:\n{retrieved_context}\n\n"
        f"Answer the following question:\n{state['input']}"
    )

    # Invoke the LLM with the augmented prompt
    response = llm.invoke(prompt)
    return {"response": response.content}


def model_params_node(state: dict) -> dict:
    """This node collects the model parameters required for the Carbon Mass Capture forecasting model."""
    retrieved_context = retrieve_docs(state["input"])
    prompt = (
        f"Using the following documentation:\n{retrieved_context}\n\n"
        f"Answer the following question:\n{state['input']}"
    )
    response = llm.invoke(prompt)
    return {"response": response.content}


graph_builder = StateGraph(State)

graph_builder.add_node("router", router_node)
graph_builder.add_node("model_info", model_info_node)
graph_builder.add_node("model_inputs", model_params_node)

agent = graph_builder.compile()
