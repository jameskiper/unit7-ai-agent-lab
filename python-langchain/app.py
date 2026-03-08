import os
import asyncio
import json
from typing import Annotated, Literal
from typing_extensions import TypedDict
#from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.types import Command


# Shared state for the workflow graph.
# This defines the structure of the data that all nodes (agents/tools)
# will read from and write to.
#
# messages:
#   - Stores the full conversation history (user messages, AI responses,
#     and tool outputs).
#   - Annotated with add_messages so that when multiple nodes update the
#     state, new messages are appended instead of overwriting existing ones.
#
# This allows all agents in the workflow to share memory and build
# on each other's outputs safely.

class State(TypedDict):
    messages: Annotated[list, add_messages]

load_dotenv()

# Planned workflow:
# User input -> Researcher -> Writer -> Editor
# The editor will either approve the result and end the workflow,
# or send it back to the writer for revision.
#
# This uses:
# - shared state to accumulate messages
# - Command-based handoffs to choose the next node
# - dynamic routing so the workflow can loop when needed

# Global variable for the researcher agent (will be set in main)
researcher_agent = None


async def researcher_node(state: State) -> Command[Literal["writer", "__end__"]]:
    """Research node that hands off to writer."""
    print("\n" + "="*50)
    print("RESEARCHER NODE")
    print("="*50)
    
    # Use the global researcher agent created in main()
    response = await researcher_agent.ainvoke({"messages": state["messages"]})
    
    # Debug: Print search results and tool usage
    print("\n--- Research Results ---")
    for msg in response["messages"]:
        # Check for tool calls (AI messages with tool_calls)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"\nTool Called: {tool_call.get('name', 'Unknown')}")
                print(f"Arguments: {tool_call.get('args', {})}")
        
        # Check for tool responses (ToolMessage)
        if msg.type == "tool":
            print(f"\nTool Response from: {getattr(msg, 'name', 'Unknown Tool')}")
            content_preview = (
                str(msg.content)[:500] + "..."
                if len(str(msg.content)) > 500
                else str(msg.content)
            )
            print(f"Content: {content_preview}")
        
        # Print AI responses
        if msg.type == "ai" and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            print("\nResearcher Response:")
            print(f"{msg.content}")
    
    print("\n" + "="*50 + "\n")
    
    return Command(
        update={"messages": response["messages"]},
        goto="writer"
    )


async def main():
    """Run the multi-agent content creation workflow."""
    
    # Check for required API keys
    if not os.getenv("GITHUB_TOKEN"):
        print("Error: GITHUB_TOKEN not found.")
        print("Add GITHUB_TOKEN=your-token to a .env file")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not found.")
        print("Add TAVILY_API_KEY=your-key to a .env file")
        print("Get your API key from: https://app.tavily.com/")
        return
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.7,
        base_url="https://models.github.ai/inference",
        api_key=os.getenv("GITHUB_TOKEN")
    )
        # Get Tavily API key from environment
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    # Create MCP client for Tavily
    research_client = MultiServerMCPClient({
        "tavily": {
            "transport": "http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}",
        }
    })
    
    # Get tools from the client (await because it's async)
    researcher_tools = await research_client.get_tools()
    
    print(f"Research tools: {[tool.name for tool in researcher_tools]}")  
        # Load prompts from your local filesystem
    with open("templates/researcher.json", "r") as f:
        researcher_data = json.load(f)
        researcher_prompt = researcher_data.get("template", "You are a helpful research assistant.")
    # We'll add more here in the next steps
    
    print("\nOrchestration setup complete!")


if __name__ == "__main__":
    asyncio.run(main())