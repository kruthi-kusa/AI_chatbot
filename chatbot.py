from transformers import pipeline
from langchain import HuggingFaceHub, LLMChain
from langchain_core.prompts import PromptTemplate
import os
from getpass import getpass
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint 
import chainlit as cl


# HUGGINFACEHUB_API_TOKEN=getpass()
# os.environ['HUGGINFACEHUB_API_TOKEN']=HUGGINFACEHUB_API_TOKEN

load_dotenv()

model_id = "gpt2-medium"
conv_model = HuggingFaceEndpoint(
    huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'
                                        ],  # Note this parameter name
    repo_id=model_id,
    temperature=0.9,
    max_new_tokens=100
)

template= """"You are a helpful AI assistant. Please respond to the following message in a clear and engaging way {query}
"""
#message_content = message.content if hasattr(message, 'content') else str(message)

@cl.on_chat_start
def main():
    prompt=PromptTemplate(template=template, input_variables=['query'])

    conv_chain=LLMChain(llm=conv_model,
                        prompt=prompt,
                        verbose=True)
    cl.user_session.set("llm_chain", conv_chain)

@cl.on_message
async def main(message:str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message, callbacks = [cl.AsyncLangchainCallbackHandler()])
    print(res)

    
    if res and "text" in res:
        await cl.Message(content=res["text"]).send()
    else:
        await cl.Message(content="I apologize, but I couldn't generate a response. Please try again.").send()
            

    #await cl.Message(content=res["text"]).send()

#print(conv_chain.run({"query": "Hello, all"}))
#print(response)