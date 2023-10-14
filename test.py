from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.5)
tools = load_tools(["arxiv"])
agent_chain = initialize_agent(
    tools,
    llm,
    max_iterations=5,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)
agent_chain.run("What is blackhole?")