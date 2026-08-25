import customer_support_agent.tools as tools_package


def test_tools_package_exposes_only_public_tool_declaration_api() -> None:
    public_names = {
        "Tool",
        "ToolContext",
        "ToolDefinition",
        "Toolset",
        "tool",
    }

    assert set(tools_package.__all__) == public_names
    assert all(hasattr(tools_package, name) for name in public_names)
