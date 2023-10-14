from dotenv import load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, load_tools, initialize_agent, AgentType
import os
import chainlit as cl

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0.8)
    tools = load_tools(
        [
            "arxiv",
        ]
    )
    agent_chain = initialize_agent(
        tools,
        llm,
        max_iterations=10,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    cl.user_session.set("agent", agent_chain)


@cl.on_message
async def main(message: str):
    agent: AgentExecutor = cl.user_session.get("agent")  # type: ignore
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    await cl.make_async(agent.run)(message, callbacks=[cb])
