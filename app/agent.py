from datetime import datetime

from langchain.agents import AgentType, Tool, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun

import llm
from llm import memory

# Initialize tools
search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or general knowledge"
    ),
    Tool(
        name="Current Time",
        func=lambda _: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="Useful for when you need to know the current time and date"
    )
]

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# Example conversation function
def chat_with_agent(user_input: str) -> str:
    try:
        response = agent.run(input=user_input)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Example conversation
    queries = [
        "What's the current time?",
        "Can you tell me about the weather in London?",
        "Remember that I like coffee in the morning.",
        "What did I tell you about my morning preferences?"
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = chat_with_agent(query)
        print(f"Agent: {response}")
