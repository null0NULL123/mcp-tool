from .basic_client import BaseClient, SSEClient, StdioClient


class FramelinkFigmaMCP(SSEClient):
    """Framelink Figma MCP client for managing Figma API interactions."""

    def __init__(self, url="http://localhost:3333/sse"):
        super().__init__("FramelinkFigmaMCP", {"url": url})

    def download_figma_image(self, _):
        """See figmapy.FigmaPy.get_file_images."""
        pass

    def get_figma_data(self, file_key: str, node_id: str) -> str:
        """Get Figma data for a specific file and node."""
        response = self.execute_tool(
            "get_figma_data",
            arguments={
                "fileKey": file_key,
                "nodeId": node_id,
            }
        )
        return response.content[0].text
