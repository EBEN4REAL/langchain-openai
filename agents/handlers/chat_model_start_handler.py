from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.schema.messages import BaseMessage
from typing import Any, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def print_panel(*args, **kwargs):
    """
    A reusable wrapper around rich Panel printing.

    Args:
        *args: The first positional argument should be the message/content.
        **kwargs: Optional rich.Panel arguments like 'title', 'style', 'padding', etc.
    """
    # Default values
    message = args[0] if args else ""
    title = kwargs.pop("title", "Info")
    style = kwargs.pop("style", "bold green")
    padding = kwargs.pop("padding", (1, 2))

    panel = Panel(message, title=title, style=style, padding=padding, **kwargs)
    console.print(panel)

class ChatModelStartHandler(BaseCallbackHandler):
    """Complete callback handler for chat models with all relevant methods."""
    
    # Message type to color/style mapping
    MESSAGE_STYLES = {
        'SystemMessage': {
            'color': 'cyan',
            'icon': '⚙️',
            'border_style': 'cyan'
        },
        'HumanMessage': {
            'color': 'green',
            'icon': '👤',
            'border_style': 'green'
        },
        'AIMessage': {
            'color': 'blue',
            'icon': '🤖',
            'border_style': 'blue'
        },
        'FunctionMessage': {
            'color': 'magenta',
            'icon': '🔧',
            'border_style': 'magenta'
        },
        'ToolMessage': {
            'color': 'yellow',
            'icon': '🛠️',
            'border_style': 'yellow'
        }
    }
    
    def __init__(self):
        self.call_count = 0
        self.total_tokens = 0
    
    def _get_message_style(self, msg_type: str) -> dict:
        """Get the style configuration for a message type."""
        return self.MESSAGE_STYLES.get(msg_type, {
            'color': 'white',
            'icon': '💬',
            'border_style': 'white'
        })
    
    def _print_message(self, msg: BaseMessage, index: int):
        """Print a message with rich formatting."""
        msg_type = msg.__class__.__name__
        style_config = self._get_message_style(msg_type)
        
        # Get message content (truncate if too long)
        content = msg.content
        if len(content) > 500:
            content = content[:500] + "..."
        
        # Create rich text with proper formatting
        message_text = Text()
        message_text.append(f"{style_config['icon']} ", style="bold")
        message_text.append(f"{msg_type}\n\n", style=f"bold {style_config['color']}")
        message_text.append(content, style=style_config['color'])
        
        # Print as panel with matching border color
        panel = Panel(
            message_text,
            title=f"Message {index}",
            border_style=style_config['border_style'],
            padding=(1, 2)
        )
        console.print(panel)
    
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any
    ) -> None:
        """Called when chat model starts - more specific than on_llm_start."""
        self.call_count += 1
        
        # Print header
        console.print(f"\n{'='*60}", style="bold white")
        console.print(f"💬 Chat Model Call #{self.call_count}", style="bold magenta")
        console.print(f"{'='*60}", style="bold white")
        
        # Extract model information
        model_info = serialized.get('id', [])
        model_name = model_info[-1] if model_info else 'unknown'
        console.print(f"Model: {model_name}", style="bold yellow")
        
        # Display messages with rich formatting
        console.print(f"\n📨 Input Messages:", style="bold white")
        
        message_counter = 1
        for batch_idx, message_batch in enumerate(messages):
            if len(messages) > 1:
                console.print(f"\n  Batch #{batch_idx + 1}:", style="bold white")
            
            for msg in message_batch:
                self._print_message(msg, message_counter)
                message_counter += 1
        
        console.print()  # Add spacing
    
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Fallback for regular LLMs (not chat models)."""
        console.print(f"\n🤖 LLM Started (fallback)", style="bold yellow")
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated."""
        console.print(token, end="", style="cyan")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM/chat model ends."""
        console.print(f"\n{'='*60}", style="bold white")
        
        # Token usage
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            if usage:
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
                
                self.total_tokens += total_tokens
                
                # Create a nice token usage panel
                token_info = Text()
                token_info.append("✅ Tokens Used:\n\n", style="bold green")
                token_info.append(f"Prompt: ", style="bold white")
                token_info.append(f"{prompt_tokens}\n", style="yellow")
                token_info.append(f"Completion: ", style="bold white")
                token_info.append(f"{completion_tokens}\n", style="yellow")
                token_info.append(f"Total: ", style="bold white")
                token_info.append(f"{total_tokens}\n", style="yellow")
                token_info.append(f"Session Total: ", style="bold white")
                token_info.append(f"{self.total_tokens}", style="bold green")
                
                panel = Panel(
                    token_info,
                    title="Token Usage",
                    border_style="green",
                    padding=(1, 2)
                )
                console.print(panel)
        
        console.print(f"{'='*60}\n", style="bold white")
    
    def on_llm_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Called when LLM/chat model errors."""
        error_panel = Panel(
            f"[bold red]{str(error)}[/bold red]",
            title="❌ Chat Model Error",
            border_style="red",
            padding=(1, 2)
        )
        console.print(error_panel)