from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda
import argparse
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language",  default="python")
args = parser.parse_args()

# Initialize the LLM
llm = OpenAI()

# Define the prompts
code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="Write a very very short {language} function that will {task}.",
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}",
)

# Step 1: Chain to generate code
code_chain = code_prompt | llm

# Step 2: Map the result into a dict { "language": ..., "code": ... }
wrap_code = RunnableLambda(
    lambda output: {"language": args.language, "code": output}
)

# Step 3: Chain to generate test from code
test_chain = test_prompt | llm

# Sequential chain: code → wrap → test
sequential_chain = code_chain | wrap_code | test_chain

# Run it
final_test = sequential_chain.invoke({"language": args.language, "task": args.task})

print("=== Generated Code ===")
print(code_chain.invoke({"language": args.language, "task": args.task}))
print("\n=== Generated Test ===")
print(final_test)
