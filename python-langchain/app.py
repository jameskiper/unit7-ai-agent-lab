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

# Shared state for the workflow graph.
# This defines the data that all agents in the workflow can access.
#
# messages:
#   - Stores the full conversation history between agents.
#   - Includes user input, AI responses, and tool outputs.
#   - The add_messages annotation ensures new messages are appended
#     instead of overwriting the previous conversation.
#
# latest_draft:
#   - Stores the most recent article created by the writer agent.
#   - This allows the workflow to print the final article at the end
#     instead of the editor's approval message.
#   - Each time the writer runs (including revisions), this value is updated
#     with the newest version of the article.
# ================================================================
# Multi-Agent Content Creation Workflow
# ================================================================
#
# This program demonstrates a multi-agent AI workflow using LangGraph.
# Each agent performs a specialized task and passes the result to the
# next agent in the pipeline.
#
# Workflow Pipeline:
#
# User Input
#      │
#      ▼
# Researcher Agent
#   - Uses Tavily search tools via MCP
#   - Gathers external information
#   - Summarizes key findings
#
#      │ Command(goto="writer")
#      ▼
# Writer Agent
#   - Converts research into a structured article
#   - Formats headings and explanations
#   - Saves the article into `latest_draft`
#
#      │ Command(goto="fact_checker")
#      ▼
# Fact-Checker Agent
#   - Reviews the writer’s content for factual accuracy
#   - If issues are found → returns "REVISE" and sends the workflow
#     back to the writer for correction
#
#      │ Command(goto="editor")
#      ▼
# Editor Agent
#   - Reviews writing quality, clarity, and structure
#   - If improvements are needed → returns "REVISE" and sends the
#     workflow back to the writer
#   - If the article is acceptable → workflow ends
#
#      │
#      ▼
# Final Output
#   - The system prints `latest_draft`, which stores the most recent
#     article written by the writer agent.
#   - This ensures the program outputs the finished article instead
#     of the editor's approval message.
#
# Key Concepts Demonstrated:
#   • Multi-agent orchestration
#   • LangGraph state sharing
#   • Tool usage via MCP (Tavily search)
#   • Conditional routing between agents
#   • Revision loops using "REVISE"
#   • Token/context management by storing only final outputs
#
# This architecture simulates a real editorial pipeline:
# Researcher → Writer → Fact-Checker → Editor
# ================================================================



# revision_count:
#   - Tracks how many times the workflow has been sent back for revision.
#   - Used to prevent infinite loops between the writer and editor.
#   - If the count reaches the maximum allowed revisions, the workflow ends.


class State(TypedDict):
    messages: Annotated[list, add_messages]
    latest_draft: str
    revision_count: int


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

# Global variables for agents (will be set in main)
researcher_agent = None
writer_agent = None
fact_checker_agent = None
editor_agent = None


async def researcher_node(state: State) -> Command[Literal["writer", "__end__"]]:
    """Research node that hands off to writer."""
    print("\n" + "="*50)
    print("RESEARCHER NODE")
    print("="*50)
    
    # Only pass the initial user message to the researcher
    response = await researcher_agent.ainvoke({"messages": [state["messages"][0]]})
    
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


async def writer_node(state: State) -> Command[Literal["fact_checker", "__end__"]]:
    """Writer node that hands off to fact-checker."""
    print("\n" + "="*50)
    print("WRITER NODE")
    print("="*50)
    
    # Only pass the researcher's final response to the writer
    final_message = state["messages"][-1]
    response = await writer_agent.ainvoke({"messages": [final_message]})
    
    # Save the writer's most recent article in latest_draft
    # so the workflow can print the finished article at the end.
    # Print the written content
    final_message = response["messages"][-1]
    print(f"\nWriter Output:")
    print(f"{final_message.content}")
    print("\n" + "="*50 + "\n")

    # Save the latest writer draft in state and hand off to fact-checker
    return Command(
        update={
            "messages": state["messages"] + [final_message],
            "latest_draft": final_message.content
        },
        goto="fact_checker"
    )


async def fact_checker_node(state: State) -> Command[Literal["editor", "writer", "__end__"]]:
    """Fact-checker node that reviews writer output before editor."""
    print("\n" + "="*50)
    print("FACT-CHECKER NODE")
    print("="*50)

    # Only pass the writer's response to the fact-checker
    final_message = state["messages"][-1]
    response = await fact_checker_agent.ainvoke({"messages": [final_message]})

    final_message = response["messages"][-1]
    print("\nFact-Checker Output:")
    print(final_message.content)

    if "REVISE" in str(final_message.content):
        print("\n⚠️ Fact-checker requested REVISION - routing back to writer")
        print("="*50 + "\n")
        return Command(
            update={"messages": state["messages"] + [final_message]},
            goto="writer"
        )

    print("\n✓ Fact-check passed - routing to editor")
    print("="*50 + "\n")
    return Command(
        update={"messages": state["messages"] + [final_message]},
        goto="editor"
    )
    
async def editor_node(state: State) -> Command[Literal["writer", "__end__"]]:
    """Editor node that can hand back to writer or end."""
    print("\n" + "="*50)
    print("EDITOR NODE")
    print("="*50)
    
    # Only pass the fact-checker's response to the editor
    final_message = state["messages"][-1]
    response = await editor_agent.ainvoke({"messages": [final_message]})
    
    # Debug: Print editor feedback
    final_message = response["messages"][-1]
    print(f"\nEditor Feedback:")
    print(f"{final_message.content}")
    
    # Example logic: if editor finds an error, hand back to writer
    current_revisions = state.get("revision_count", 0)

    if "REVISE" in str(final_message.content):
        if current_revisions >= 2:
            print("\n⚠️ Maximum revision limit reached - ending workflow")
            print("="*50 + "\n")
            return Command(
                update={
                    "messages": state["messages"] + [final_message],
                    "revision_count": current_revisions
                },
                goto="__end__"
            )

        print("\n⚠️  Editor requested REVISION - routing back to writer")
        print("="*50 + "\n")
        return Command(
            update={
                "messages": state["messages"] + [final_message],
                "revision_count": current_revisions + 1
            },
            goto="writer"
        )
    
    print("\n✓ Editor approved - workflow complete")
    print("="*50 + "\n")
    return Command(
        update={
            "messages": state["messages"] + [final_message],
            "revision_count": current_revisions
        },
        goto="__end__"
    )
    
async def main():
    """Run the multi-agent content creation workflow."""
    global researcher_agent, writer_agent, fact_checker_agent, editor_agent
    # ... rest of the code
    
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

    # Load prompts from your local filesystem
    with open("templates/researcher.json", "r") as f:
        researcher_data = json.load(f)
        researcher_prompt = researcher_data.get(
            "template",
            "You are a helpful research assistant."
        )

    with open("templates/writer.json", "r") as f:
        writer_data = json.load(f)
        writer_prompt = writer_data.get(
            "template",
            "You are a helpful writing assistant."
        )

    with open("templates/editor.json", "r") as f:
        editor_data = json.load(f)
        editor_prompt = editor_data.get(
            "template",
            "You are a helpful editing assistant."
        )
    
    with open("templates/fact_checker.json", "r") as f:
        fact_checker_data = json.load(f)
        fact_checker_prompt = fact_checker_data.get(
            "template",
            "You are a helpful fact checking assistant."
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
    
    # Get tools from the client
    researcher_tools = await research_client.get_tools()
    
    print(f"Research tools: {[tool.name for tool in researcher_tools]}")
    
    # Create researcher agent
    researcher_agent = create_agent(
        llm,
        tools=researcher_tools,
        system_prompt=researcher_prompt
    )
    # Writer and editor don't need tools
    writer_agent = create_agent(
        llm, 
        tools=[],
        system_prompt=writer_prompt
    )

    editor_agent = create_agent(
        llm, 
        tools=[], 
        system_prompt=editor_prompt
    )
    
    fact_checker_agent = create_agent(
        llm,
        tools=[],
        system_prompt=fact_checker_prompt
    )
        
    # Build the Graph without manual edges (Edgeless Handoff)
    builder = StateGraph(State)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("fact_checker", fact_checker_node)
    builder.add_node("editor", editor_node)
    
    # Only need to set the entry point
    builder.add_edge(START, "researcher")
    graph = builder.compile()
        
    # Run the workflow
        
    print("\n" + "="*50)
    print("Starting Multi-Agent Content Creation Workflow")
    print("="*50 + "\n")

    user_input = input("Enter the topic that you would like to research: ")
    initial_message = HumanMessage(content=user_input)
    result = await graph.ainvoke({
    "messages": [initial_message],
    "latest_draft": "",
    "revision_count": 0
})

    # Print the final result of the workflow.
    #
    # Instead of printing the last message in the conversation history,
    # we print "latest_draft", which stores the most recent article
    # produced by the writer agent.
    #
    # The editor is the final node in the workflow, so the last message
    # would normally be the editor's approval or revision feedback.
    # By storing the writer's article in "latest_draft", we can display
    # the finished content instead of the editor's comments.


    print("\n" + "="*50)
    print("Workflow Complete")
    print("="*50 + "\n")
    print("Final Output:")
    print(result.get("latest_draft", "No draft available"))
    
if __name__ == "__main__":
    asyncio.run(main())