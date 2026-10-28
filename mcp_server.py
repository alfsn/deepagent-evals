"""A tiny MCP server (stdio) exposing brand data, so the agent mixes local tools with MCP tools.

Run standalone to inspect it:  uv run mcp dev mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("brands")

BRANDS = {
    "glossier": {
        "values": ["inclusive", "skin-first", "authentic"],
        "target_audience": "US women 18-34",
        "min_engagement_rate": 2.5,
        "exclusions": ["anyone who has ever partnered with competitor Fenty Beauty"],
    },
    "gymshark": {
        "values": ["community", "self-improvement"],
        "target_audience": "18-34, UK/EU/US",
        "min_engagement_rate": 2.0,
        "exclusions": ["supplement brands with unverified claims"],
    },
}


@mcp.tool()
def get_brand_guidelines(brand: str) -> dict:
    """Return a brand's values, target audience, minimum engagement rate and exclusion rules."""
    return BRANDS.get(brand.lower(), {"error": f"no guidelines for {brand}"})


@mcp.tool()
def list_brands() -> list[str]:
    """List brands that have guidelines on file."""
    return sorted(BRANDS)


if __name__ == "__main__":
    mcp.run(transport="stdio")
