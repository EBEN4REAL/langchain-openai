from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# 1. Define input schema
class WriteReportArgsSchema(BaseModel):
    filename: str = Field(description="Report filename")
    html: str = Field(description="Report content")

# 2. Define the function
def write_report(filename: str, html: str) -> str:
    """Write a report to a file"""
    with open(filename, 'w') as f:
        f.write(html)
    return f"✅ Report saved to {filename}"

# 3. Create the tool
report_tool = StructuredTool.from_function(
    func=write_report,
    name="write_report",
    description="Write an  HTML file to disk. USe this tools whenever asks for a report",
    args_schema=WriteReportArgsSchema
)

# 4. Use it
result = report_tool.run({
    "filename": "my_report.txt",
    "html": "This is my report content"
})
print(result)