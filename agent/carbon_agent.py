import json
from typing import TypedDict, Sequence, List, Dict, Optional
from datetime import datetime
from uuid import uuid4
import chromadb
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import Graph, StateGraph
from enum import Enum
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing_extensions import Literal

load_dotenv()


class ConversationState(TypedDict):
    messages: Sequence[BaseMessage]
    current_input: str
    conversation_id: str
    similar_conversations: List[Dict]
    summaries: List[str]
    final_response: Optional[str]
    metadata: Optional[Dict]
    should_use_context: bool
    routing_decision: Optional[str]


class Action(str, Enum):
    FIND_SIMILAR = "find_similar"
    SUMMARIZE = "summarize"
    GENERATE = "generate"
    STORE = "store"
    END = "end"


class ModelRoute(BaseModel):
    step: Literal["default", "insight", "update"] = Field(
        None,
        description="Routing decision: default conversation, model insight, or model update",
    )


class ConversationMemoryGraph:
    def __init__(
        self,
        persist_directory: str = "./chroma_conversations",
        collection_name: str = "conversation_history",
        similarity_threshold: float = 0.6,
    ):
        # Initialize ChromaDB
        self.client = chromadb.HttpClient(
            ssl=True,
            host="api.trychroma.com",
            tenant="059b2ec8-9b7b-4223-b5e7-3628afcc27c2",
            database="Carbon-Factibility",
            headers={
                "x-chroma-token": "ck-CPNwRiwNLReoAu9tPrGSZZQALVvMtK3jXzygqF8AYsJc"
            },
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        # Initialize LLM
        self.llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        self.similarity_threshold = similarity_threshold
        self.workflow = self._create_workflow()

    def _route_input(self, state: ConversationState) -> ConversationState:
        """Route the input to decide whether it is a model improvement query or a standard conversation.
        Uses structured output from the LLM.
        """
        print("\n=== Routing Input ===")
        # Augment the LLM with structured output for routing
        router = self.llm.with_structured_output(ModelRoute)
        decision = router.invoke(
            [
                SystemMessage(
                    content=(
                        "Route the input as follows: if the user is asking for insights on how to improve the model, "
                        "output 'insight'. If the user wants to update the model parameters (e.g., change variables to "
                        "affect predictions), output 'update'. Otherwise, output 'default'."
                    )
                ),
                HumanMessage(content=state["current_input"]),
            ]
        )
        state["routing_decision"] = decision.step
        print(f"Routing decision: {decision.step}")
        return state

    def _find_similar_conversations(
        self, state: ConversationState
    ) -> ConversationState:
        """Node: Find similar previous conversations"""
        print("\n=== Starting Similarity Search ===")
        print(f"Searching for messages similar to: {state['current_input']}")

        results = self.collection.query(
            query_texts=[state["current_input"]],
            n_results=3,
            where={"role": "human"},
        )

        print(f"\nFound {len(results['documents'][0])} potential matches")  # type: ignore

        similar_conversations = []
        for doc, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]  # type: ignore
        ):
            similarity_score = 1 - distance
            print(f"\nPotential match:")
            print(f"- Message: {doc[:100]}...")
            print(f"- Similarity score: {similarity_score:.3f}")
            print(f"- Threshold: {self.similarity_threshold}")

            if similarity_score >= self.similarity_threshold:
                print("✓ Match accepted (above threshold)")
                conv_id = metadata["conversation_id"]
                print(f"- Conversation ID: {conv_id}")

                conv_messages = self.collection.get(
                    where={"conversation_id": conv_id},
                )

                messages = []
                for msg_content, msg_metadata in zip(
                    conv_messages["documents"], conv_messages["metadatas"]  # type: ignore
                ):
                    messages.append(
                        {
                            "role": msg_metadata["role"],
                            "content": msg_content,
                            "timestamp": msg_metadata["timestamp"],
                        }
                    )
                messages.sort(key=lambda x: x["timestamp"])

                similar_conversations.append(
                    {
                        "conversation": messages,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                    }
                )
            else:
                print("✗ Match rejected (below threshold)")

        print(f"\nTotal similar conversations found: {len(similar_conversations)}")
        print("=== Similarity Search Complete ===\n")

        return {**state, "similar_conversations": similar_conversations}

    def _summarize_conversations(self, state: ConversationState) -> ConversationState:
        """Node: Generate summaries for similar conversations"""
        if not state["similar_conversations"]:
            return {**state, "summaries": []}

        template = """Provide a concise summary of this conversation, 
focusing on key points and conclusions:

{conversation}

Summary:"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        summaries = []
        for conv in state["similar_conversations"]:
            conversation_text = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in conv["conversation"]
            )

            summary = chain.invoke({"conversation": conversation_text})
            summaries.append(
                {
                    "summary": summary,
                    "timestamp": conv["metadata"]["timestamp"],
                    "similarity": conv["similarity_score"],
                }
            )

        return {**state, "summaries": summaries}

    def _generate_response(self, state: ConversationState) -> ConversationState:
        """Node: Generate response using context if available"""
        messages = state.get("messages", [])

        if messages:
            conversation_history = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}" for msg in messages  # type: ignore
            )
            conversation_context = f"\nConversation history:\n{conversation_history}\n"
        else:
            conversation_context = ""

        if state["summaries"] and state["should_use_context"]:
            template = """You have access to summaries of previous related conversations and the current conversation history.
You should explicitly mention when you're using information from previous conversations.
If there is no previous conversation, don't say anything about "no prior context", just answer the question.

{conversation_context}

Previous relevant conversations:
{summaries}

Current query: {query}

Please start your response by acknowledging the similar conversations you found, 
and then provide your answer incorporating insights from both the previous conversations 
and the current context.

Response:"""
            summaries_text = "\n".join(
                f"\n---\nPrevious conversation ({s['timestamp']}, similarity: {s['similarity']:.2f}):\n{s['summary']}"
                for s in state["summaries"]
            )
        else:
            template = """Please provide a response to the following query, taking into account 
any existing conversation context. If there is no conversation history, just answer the question.

{conversation_context}
Query: {query}

If there is conversation history, maintain continuity and reference 
previous parts of the conversation when relevant.

Response:"""
            summaries_text = ""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke(
            {
                "summaries": summaries_text,
                "query": state["current_input"],
                "conversation_context": conversation_context,
            }
        )

        return {**state, "final_response": response}

    def _model_insight(self, state: ConversationState) -> ConversationState:
        """Node: Provide expert insights on how to improve the model.
        Focus on which variables to adjust and how they affect the predicted value and its shape.
        """
        print("\n=== Generating Model Improvement Insights ===")
        prompt = ChatPromptTemplate.from_template(
            """You are an expert in machine learning model improvement.
The user asked: "{query}"

Please provide detailed insights on how to improve the model, including which variables (e.g., temperature, max_tokens) to adjust,
how these variables affect the predicted value, and what impact they have on the shape of the output.

Insights:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": state["current_input"]})
        state["final_response"] = response
        return state

    def _update_model_parameters(self, state: ConversationState) -> ConversationState:
        """Node: Parse and process model parameter updates from the input.
        The LLM should return a JSON with the new parameters and an explanation of their impact.
        """
        print("\n=== Processing Model Parameter Update ===")
        prompt = ChatPromptTemplate.from_template(
            """You are an expert in configuring machine learning models.
The user requested an update with the following input: "{query}"

Parse the input and return a JSON object that specifies:
 - Which model parameters should be updated,
 - Their new values,
 - And a brief explanation of how these changes may affect the predicted value and its shape.

Response:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": state["current_input"]})
        state["final_response"] = response
        # Optionally, you can parse and store the JSON in state (e.g., state["model_parameters"])
        try:
            state["metadata"]["model_parameters"] = json.loads(response)
        except Exception:
            state["metadata"]["model_parameters"] = {"raw_response": response}
        return state

    def _store_conversation(self, state: ConversationState) -> ConversationState:
        """Node: Store the conversation in ChromaDB along with additional metadata.
        If the conversation involves model updates, include the parameters and their impact.
        """
        current_time = datetime.now().isoformat()

        base_metadata = {
            "conversation_id": state["conversation_id"],
            "timestamp": current_time,
        }

        if state.get("routing_decision"):
            base_metadata["routing_decision"] = state["routing_decision"]

        # If this was a model update, store the parsed parameters (if available)
        if state.get("routing_decision") == "update" and state.get("metadata", {}).get(
            "model_parameters"
        ):
            base_metadata["model_parameters"] = json.dumps(
                state["metadata"]["model_parameters"]
            )

        if state["similar_conversations"]:
            similar_ids = [
                conv["metadata"]["conversation_id"]
                for conv in state["similar_conversations"]
            ]
            base_metadata["similar_conversations"] = json.dumps(similar_ids)
            base_metadata["similarity_scores"] = json.dumps(
                [conv["similarity_score"] for conv in state["similar_conversations"]]
            )

        human_metadata = {
            **base_metadata,
            "role": "human",
            "message_type": "query",
        }

        ai_metadata = {
            **base_metadata,
            "role": "ai",
            "message_type": "response",
        }

        base_id = f"conv_{state['conversation_id']}"
        self.collection.add(
            documents=[state["current_input"], state["final_response"]],  # type: ignore
            metadatas=[human_metadata, ai_metadata],
            ids=[f"{base_id}_human_{uuid4()}", f"{base_id}_ai_{uuid4()}"],
        )

        return state

    def _create_workflow(self) -> Graph:
        """Create the LangGraph workflow with routing.
        The workflow starts with routing the input. Based on the decision:
          - 'default': run the similarity search, summarization, response generation nodes.
          - 'insight': generate model improvement insights.
          - 'update': process model parameter updates.
        All branches eventually lead to storing the conversation.
        """
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("route_input", self._route_input)
        workflow.add_node("find_similar", self._find_similar_conversations)
        workflow.add_node("summarize", self._summarize_conversations)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("model_insight", self._model_insight)
        workflow.add_node("update_model_parameters", self._update_model_parameters)
        workflow.add_node("store", self._store_conversation)

        def route_decision(state: ConversationState):
            decision = state.get("routing_decision", "default")
            if decision == "insight":
                return "model_insight"
            elif decision == "update":
                return "update_model_parameters"
            else:
                return "find_similar"

        workflow.set_entry_point("route_input")
        workflow.add_conditional_edges(
            "route_input",
            route_decision,
            {
                "model_insight": "model_insight",
                "update_model_parameters": "update_model_parameters",
                "find_similar": "find_similar",
            },
        )

        workflow.add_edge("find_similar", "summarize")
        workflow.add_edge("summarize", "generate")
        workflow.add_edge("generate", "store")
        workflow.add_edge("model_insight", "store")
        workflow.add_edge("update_model_parameters", "store")

        return workflow.compile()  # type: ignore

    def process_message(
        self, message: str, conversation_id: Optional[str] = None
    ) -> Dict:
        """Process a message and return the response with conversation context."""
        print("\n=== Processing New Message ===")
        print(f"Message: {message}")
        print(f"Conversation ID: {conversation_id or 'New conversation'}")

        messages = []
        if conversation_id:
            previous_messages = self.collection.get(
                where={"conversation_id": conversation_id},
            )
            for doc, metadata in zip(
                previous_messages["documents"], previous_messages["metadatas"]  # type: ignore
            ):
                messages.append(
                    {
                        "content": doc,
                        "role": metadata["role"],
                        "timestamp": metadata["timestamp"],
                    }
                )
            messages.sort(key=lambda x: x["timestamp"])
            print(f"Found {len(messages)} previous messages in this conversation")
            print("Skipping routing for existing conversation")

        state: ConversationState = {
            "current_input": message,
            "conversation_id": conversation_id or str(uuid4()),
            "similar_conversations": [],
            "summaries": [],
            "metadata": {"timestamp": datetime.now().isoformat()},
            "final_response": None,
            "should_use_context": not bool(messages),
            "messages": messages,
            "routing_decision": None,
        }

        if messages:
            print("\nGenerating response with conversation history...")
            template = """Here is the conversation history:
{conversation_history}

Current query: {query}

Please provide a response that takes into account the conversation history 
and maintains continuity with the ongoing discussion. Make references to 
previous parts of the conversation when relevant.

Response:"""

            conversation_history = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}" for msg in messages
            )

            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()

            response = chain.invoke(
                {"conversation_history": conversation_history, "query": message}
            )

            state["final_response"] = response
            self._store_conversation(state)
        else:
            print("\nRunning workflow for new conversation with routing...")
            state = self.workflow.invoke(state)  # type: ignore

        return {
            "response": state["final_response"],
            "found_similar": len(state["similar_conversations"]) > 0,
            "similar_conversations": state["similar_conversations"],
            "conversation_id": state["conversation_id"],
            "routing_decision": state.get("routing_decision"),
        }


graph = ConversationMemoryGraph().workflow
