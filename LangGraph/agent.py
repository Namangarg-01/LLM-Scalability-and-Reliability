from typing import Annotated 
from typing_extensions import TypedDict 
from langchain_groq import ChatGroq
from langgraph.graph import START, END 
from langgraph.graph.state import StateGraph 
from langgraph.prebuilt import ToolNode 
from langgraph.graph.message import add_messages 
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage 
from langchain.chat_models import init_chat_model
from langgraph.graph import START, END 
from langgraph.graph.state import StateGraph
from langgraph.prebuilt import tools_condition
from langchain.tools import tool
import os
from dotenv import load_dotenv
load_dotenv() # To load our env variable

os.environ["LANGSMITH_PROJECT"] = "TestProject"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

llm = init_chat_model("groq:openai/gpt-oss-120b")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def make_tool_graph():
    
    @tool
    def add(a:float, b:float):
        """Add two numbers. Use this tool every time the user asks to add, sum or total two numbers,
        even simple ones. Never do the addition yourself."""
        return a + b

    tool_node = ToolNode([add])

    llm_with_tool = llm.bind_tools([add])

    def call_llm_model(state: State):
        return {"messages": [llm_with_tool.invoke(state['messages'])]}


    builder = StateGraph(State)
    builder.add_node("call_llm_model", call_llm_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "call_llm_model")
    builder.add_conditional_edges(
        "call_llm_model",
        tools_condition
    )
    builder.add_edge("tools", "call_llm_model")

    graph = builder.compile()
    return graph

tool_agent = make_tool_graph()

