"""
Original source: https://github.com/sooperset/mcp-client-slackbot
License: MIT
"""

import asyncio
import json
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, List

import mcp.types
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from .logger import logger


class Configuration:
    """Manages configuration and environment variables for the MCP Slackbot."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.slack_app_token = os.getenv("SLACK_APP_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the appropriate LLM API key based on the model.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If no API key is found for the selected model.
        """
        if "gpt" in self.llm_model.lower() and self.openai_api_key:
            return self.openai_api_key
        elif "llama" in self.llm_model.lower() and self.groq_api_key:
            return self.groq_api_key
        elif "claude" in self.llm_model.lower() and self.anthropic_api_key:
            return self.anthropic_api_key

        # Fallback to any available key
        if self.openai_api_key:
            return self.openai_api_key
        elif self.groq_api_key:
            return self.groq_api_key
        elif self.anthropic_api_key:
            return self.anthropic_api_key

        raise ValueError("No API key found for any LLM provider")


class Tool:
    """Represents a tool with its properties and formatting.

    NOTE: 重新设计了Tool类以支持更灵活的工具描述和参数处理
    """

    def __init__(
        self, name: str, description: str, input_schema: Dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.inputSchema: Dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.inputSchema:
            for param_name, param_info in self.inputSchema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.inputSchema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class BaseClient:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.session_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        pass

    async def _list_tools(self) -> List[Tool]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(
                        Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def _execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int,
        delay: float,
    ) -> mcp.types.CallToolResult:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logger.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                logger.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries reached. Failing.")
                    raise

    async def _cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.session_context = None
            except Exception as e:
                logger.error(
                    f"Error during cleanup of server {self.name}: {e}")

    def list_tools(self) -> List[Tool]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        async def _list_tools() -> List[Tool]:
            try:
                await self.initialize()
                return await self._list_tools()
            finally:
                await self._cleanup()
        return asyncio.run(_list_tools())

    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> mcp.types.CallToolResult:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        async def _execute_tool() -> Any:
            try:
                await self.initialize()
                return await self._execute_tool(tool_name, arguments, retries, delay)
            finally:
                await self._cleanup()
        return asyncio.run(_execute_tool())


class StdioClient(BaseClient):
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError(
                "The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
            cwd=self.config.get("cwd", None),
            encoding=self.config.get("encoding", "utf-8"),
            encoding_error_handler=self.config.get(
                "encoding_error_handler", "replace"),
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logger.error(f"Error initializing server {self.name}: {e}")
            await self._cleanup()
            raise


class SSEClient(BaseClient):
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """
        Args:
            name: Name of the server.
            config: Configuration dictionary containing server parameters.
                - url: URL of the SSE server.
                - headers: Optional headers for the SSE connection.
                - timeout: Optional timeout for the SSE connection.
                - sse_read_timeout: Optional read timeout for SSE events.
                - httpx_client_factory: Optional factory for creating HTTPX client.
                - auth: Optional authentication credentials.
        """
        super().__init__(name, config)

    async def initialize(self) -> None:
        """Initialize the server connection."""
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                sse_client(**self.config))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logger.error(f"Error initializing server {self.name}: {e}")
            await self._cleanup()
            raise



if __name__ == "__main__":

    # client = SSEClient(
    #     "FramelinkFigmaMCP",
    #     {"url": "http://localhost:3333/sse"},
    # )
    client = StdioClient("f2c-mcp", {
        "command": "npx",
        "args": [
            "@f2c/mcp",
        ],
        "env": {
            "figma_api_token": "your_figma_api_token_here",
        }
    })

    tools = client.list_tools()
    for tool in tools:
        print(tool.format_for_llm())

    code = client.execute_tool(
        "get_code",
        {
            "fileKey": "your_file_key_here",
            "ids": "0:1",
            "personalToken": "your_figma_api_token_here",
            "localPath": "your_local_path_here",
            "imgFormat": "png",
            "scaleSize": 3})
    print(code)
