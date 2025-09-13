from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")


os.environ["OPENAI_API_KEY"]=""

#os.environ["LANGSMITH_API_KEY"]="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"

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_PROJECT"] = "awsome-proj2"

# Optional: Background callbacks (default: true)
# Set to false for serverless environments to ensure traces complete
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "true"

class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

# model=ChatOpenAI(temperature=0)
model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        #api_key=os.environ["GEMINI_API_KEY"]  # uses GOOGLE_API_KEY env var by default
    )
def make_default_graph():
    graph_workflow=StateGraph(State)

    def call_model(state):
        return {"messages":[model.invoke(state['messages'])]}
    
    graph_workflow.add_node("agent2", call_model)
    graph_workflow.add_edge("agent2", END)
    graph_workflow.add_edge(START, "agent2")

    agent=graph_workflow.compile()
    return agent

def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent2", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent2")
    graph_workflow.add_edge(START, "agent2")
    graph_workflow.add_conditional_edges("agent2", should_continue)

    agent = graph_workflow.compile()
    return agent

agent=make_alternative_graph()

