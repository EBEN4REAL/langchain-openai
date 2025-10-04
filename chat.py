from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import json
import os
from datetime import datetime

load_dotenv()

# ============================================================
# HELPER FUNCTIONS FOR SAVING/LOADING
# ============================================================

def save_history_to_json(store, filename="messages.json"):
    """Save conversation history to a JSON file"""
    data = {}
    
    for session_id, history in store.items():
        messages_list = []
        for msg in history.messages:
            messages_list.append({
                "type": msg.type,
                "content": msg.content,
                "timestamp": datetime.now().isoformat()  # Add timestamp
            })
        data[session_id] = messages_list
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ History saved to {filename}")


def load_history_from_json(filename="messages.json"):
    """Load conversation history from a JSON file"""
    if not os.path.exists(filename):
        print(f"⚠️  No existing history file found ({filename})")
        return {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    store = {}
    for session_id, messages_list in data.items():
        history = InMemoryChatMessageHistory()
        for msg_data in messages_list:
            if msg_data["type"] == "human":
                history.add_message(HumanMessage(content=msg_data["content"]))
            elif msg_data["type"] == "ai":
                history.add_message(AIMessage(content=msg_data["content"]))
        store[session_id] = history
    
    print(f"✅ History loaded from {filename}")
    return store


# ============================================================
# PART 1: Simple Format Demo (without memory)
# ============================================================
print("="*60)
print("DEMONSTRATION: How Messages are Combined")
print("="*60)
print("\n📌 PART 1: Simple Message Formatting Demo\n")

# Define messages
system_msg = ChatMessagePromptTemplate.from_template(
    role="system",
    template="You are an expert {language} developer."
)

human_msg = HumanMessagePromptTemplate.from_template(
    "Write a short {language} function that {task}."
)

# Combine them
simple_prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])

# Format with variables
prompt_value = simple_prompt.format_prompt(language="python", task="adds two numbers")

print("📝 Formatted Messages:")
for i, msg in enumerate(prompt_value.to_messages(), 1):
    print(f"\n{i}. {msg.__class__.__name__}:")
    print(f"   {msg.content}")

# ============================================================
# PART 2: Chat with Memory Demo
# ============================================================
print("\n" + "="*60)
print("📌 PART 2: Chat with Memory (with JSON saving)\n")

# 1️⃣ Define custom system message with variables
system_message = ChatMessagePromptTemplate.from_template(
    role="system",
    template="You are an expert {language} developer with years of experience. Your specialty is {specialty}."
)

# 2️⃣ Define custom human message template
human_message = HumanMessagePromptTemplate.from_template(
    "As a {language} expert, please help me with: {input}"
)

# 3️⃣ Build chat prompt with all components
chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="history"),
    human_message,
])

# 4️⃣ Chain the prompt and model
chat = ChatOpenAI(temperature=0.7)
chain = chat_prompt | chat

# 5️⃣ Load existing history or create new store
store = load_history_from_json("messages.json")

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 6️⃣ Wrap with message history
chat_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 7️⃣ Interactive loop with dynamic settings
session_id = "ebenezer_session_1"

# Configure the assistant
language = input("Programming language (e.g., Python, JavaScript): ").strip() or "Python"
specialty = input(f"Specialty (e.g., web development, data science): ").strip() or "general development"

print(f"\n✅ I'm now an expert {language} developer specializing in {specialty}!")
print("Ask me anything (type 'exit' to quit, 'save' to save without quitting)\n")

while True:
    content = input("You: ")
    
    if content.lower() in ["exit", "quit", "bye"]:
        # Save before exiting
        save_history_to_json(store, "messages.json")
        print("AI: Goodbye 👋")
        break
    
    if content.lower() == "save":
        # Save without exiting
        save_history_to_json(store, "messages.json")
        continue

    # Invoke with all variables
    result = chat_with_memory.invoke(
        {
            "language": language,
            "specialty": specialty,
            "input": content
        },
        config={"configurable": {"session_id": session_id}},
    )

    print(f"AI: {result.content}\n")

# 8️⃣ Show the final message structure
print("\n" + "="*60)
print("📋 FINAL MESSAGE STRUCTURE")
print("="*60)

print("\n🔹 System Message:")
print(f"   'You are an expert {language} developer with years of experience. Your specialty is {specialty}.'")

print("\n🔹 Chat History:")
history = store.get(session_id)
if history and history.messages:
    for i, msg in enumerate(history.messages, 1):
        role = "👤 Human" if msg.type == "human" else "🤖 AI"
        content_preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        print(f"   {i}. {role}: {content_preview}")
else:
    print("   (No history)")

print("\n🔹 Human Message Template:")
print(f"   'As a {language} expert, please help me with: {{user_input}}'")

print("\n" + "="*60)
print("💡 These messages are combined and sent to the LLM together!")
print("="*60)