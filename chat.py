import asyncio
from graph import graph
import sys
from rich.console import Console

console = Console()

async def main():
    print("===================================================")
    print("                 JARVIS IS ONLINE                  ")
    print("   Your personal AI assistant at your service.     ")
    print("===================================================")
    print("Type 'exit' or 'quit' to stop.")
    print("===================================================")

    # We use a specific thread ID so the supervisor checkpointer 
    # maintains your conversation history throughout the session.
    thread_id = "jarvis_supervisor_session"
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        try:
            # Safely get user input without blocking the async event loop
            user_input = await asyncio.to_thread(input, "\nYou: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.lower() in ['exit', 'quit']:
            print("\nJARVIS: Powering down. Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        try:
            print("\nJARVIS: ", end="", flush=True)
            
            # Use astream_events to get real-time updates from the graph
            async for event in graph.astream_events(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                version="v2"
            ):
                kind = event["event"]
                
                # Detect when a tool (sub-agent) starts running
                if kind == "on_tool_start":
                    tool_name = event["name"]
                    print(f"\n\n[JARVIS is delegating to sub-agent: {tool_name}...]", end="", flush=True)
                    
                # Detect when a tool (sub-agent) finishes
                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    print(f"\n[Sub-agent {tool_name} finished!]\n\nJARVIS: ", end="", flush=True)
                    
                # Stream the actual text tokens from the LLM word-by-word
                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        if isinstance(content, str):
                            console.print(content, end="")
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    console.print(block.get("text", ""), end="")
                                elif isinstance(block, str):
                                    console.print(block, end="")
                        
            print() # Print a final newline when the stream completes
            
        except Exception as e:
            print(f"\nJARVIS Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
