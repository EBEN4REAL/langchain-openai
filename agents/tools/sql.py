from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

# Load the db path dynamically
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "db.sqlite")

# NEW: Function to get database schema
def get_database_schema():
    """Get the complete schema of the database."""
    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_info = []
        for (table_name,) in tables:
            # Get column info for each table
            cursor.execute(f"PRAGMA table_info({table_name})") # pragma to get table columns info
            columns = cursor.fetchall()
            
            col_details = []
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, is_pk = col
                col_details.append(f"{col_name} ({col_type})")
            
            schema_info.append(f"{table_name}: {', '.join(col_details)}")
        
        conn.close()
        return "\n".join(schema_info)
    except Exception as e:
        return f"Error getting schema: {str(e)}"

# Get schema at startup
database_schema = get_database_schema()
print("Database Schema:")
print(database_schema)
print()

# Define the tool
@tool
def run_sqlite_query(query: str) -> str:
    """
    Run a SQL query against the SQLite database and return the results.
    
    Args:
        query: A valid SQL query string (e.g., SELECT, COUNT, etc.)
        
    Returns:
        Query results as a formatted string
    """
    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return "Query executed successfully but returned no results."
        
        return str(results)
    except Exception as e:
        return f"Error executing query: {str(e)}"

# Setup the agent with database schema in system prompt
tools = [run_sqlite_query]

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are a helpful database assistant. You can execute SQL queries using the run_sqlite_query tool.

DATABASE SCHEMA:
{database_schema}

IMPORTANT INSTRUCTIONS:
1. Always refer to the schema above to understand the database structure
2. Users and addresses are in separate tables - you need to JOIN them
3. The addresses table likely has a user_id foreign key to link to users
4. Use appropriate SQL queries (SELECT COUNT(*), SELECT *, JOINs, etc.)
5. Always provide clear, formatted answers

Common queries:
- Count users: SELECT COUNT(*) FROM users
- Count users with addresses: SELECT COUNT(DISTINCT user_id) FROM addresses
- List tables: SELECT name FROM sqlite_master WHERE type='table'
- See table structure: PRAGMA table_info(table_name)
- Join users and addresses: SELECT * FROM users JOIN addresses ON users.id = addresses.user_id"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Run the query to count users
if __name__ == "__main__":
    print("="*60)
    print("COUNTING USERS WITH SHIPPING ADDRESS")
    print("="*60 + "\n")
    
    response = agent_executor.invoke({
        "input": "How many users have provided a shipping address?"
    })
    
    print("\n" + "="*60)
    print("ANSWER:")
    print(response["output"])
    print("="*60)