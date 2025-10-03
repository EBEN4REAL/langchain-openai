from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import argparse
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language",  default="python")
args = parser.parse_args()

# Initialize the LLM
llm = OpenAI()

# Define the prompt
code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very very short {language} function that will {task}.",
)

# New syntax: chain = prompt | llm
chain = code_prompt | llm # Create the chain by piping the prompt to the LLM

# Use .invoke instead of .run
result = chain.invoke({"language": args.language, "task": args.task})

print(result)
