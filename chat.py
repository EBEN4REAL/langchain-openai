from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, ChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  


# Define messages
system_message = ChatMessagePromptTemplate.from_template(
    role="system",
    template="You are an expert {language} developer."
)

human_message = HumanMessagePromptTemplate.from_template(
    "Write a short {language} function that {task}."
)

# Combine them into a chat prompt
chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

# Format with variables
prompt_value = chat_prompt.format_prompt(language="python", task="adds two numbers")

# This gives you a ChatPromptValue, which can be passed directly to a chat model
# print(prompt_value.to_messages())

chat = ChatOpenAI()

response = chat.invoke(prompt_value.to_messages())

while True:
    content = input("You: ")

    result = chat.invoke([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
    ])

    print("AI: ", result.content)
