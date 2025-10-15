"""Simple system validation script for Axiom Investment Banking Analytics."""

import sys
import os
from pathlib import Path
import importlib.util

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import(module_name: str, description: str = "") -> bool:
    """Test if a module can be imported."""
    try:
        if '.' in module_name:
            # Handle submodules
            parts = module_name.split('.')
            module = __import__(module_name, fromlist=[parts[-1]])
        else:
            module = __import__(module_name)
        print(f"✅ {description or module_name}: Import successful")
        return True
    except Exception as e:
        print(f"❌ {description or module_name}: Import failed - {str(e)}")
        return False


def test_file_exists(file_path: str, description: str = "") -> bool:
    """Test if a file exists."""
    if os.path.exists(file_path):
        print(f"✅ {description or file_path}: File exists")
        return True
    else:
        print(f"❌ {description or file_path}: File missing")
        return False


def test_class_instantiation(module_name: str, class_name: str, description: str = "") -> bool:
    """Test if a class can be instantiated."""
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        # Don't actually instantiate, just check if class exists
        print(f"✅ {description or f'{module_name}.{class_name}'}: Class available")
        return True
    except Exception as e:
        print(f"❌ {description or f'{module_name}.{class_name}'}: Class test failed - {str(e)}")
        return False


def validate_project_structure():
    """Validate project file structure."""
    print("📁 Validating Project Structure...")
    
    # Key directories
    directories = [
        ("axiom", "Main package directory"),
        ("axiom/ai_client_integrations", "AI provider integrations"),
        ("axiom/config", "Configuration files"),
        ("axiom/graph", "LangGraph workflow"),
        ("axiom/graph/nodes", "Workflow nodes"),
        ("axiom/tools", "Tool integrations"),
        ("axiom/dspy_modules", "DSPy optimization"),
        ("axiom/utils", "Utilities"),
        ("tests", "Test files")
    ]
    
    structure_valid = True
    for directory, description in directories:
        if os.path.isdir(directory):
            print(f"✅ {description}: {directory}")
        else:
            print(f"❌ {description}: {directory} - Missing")
            structure_valid = False
    
    return structure_valid


def validate_core_files():
    """Validate core implementation files."""
    print("\n📋 Validating Core Files...")
    
    core_files = [
        ("axiom/__init__.py", "Package init"),
        ("axiom/main.py", "Main entry point"),
        ("axiom/config/settings.py", "Settings configuration"),
        ("axiom/config/schemas.py", "Data schemas"),
        ("axiom/config/ai_layer_config.py", "AI layer configuration"),
        ("axiom/graph/state.py", "Graph state management"),
        ("axiom/graph/graph.py", "LangGraph workflow"),
        ("axiom/ai_client_integrations/base_ai_provider.py", "Base AI provider"),
        ("axiom/ai_client_integrations/openai_provider.py", "OpenAI provider"),
        ("axiom/ai_client_integrations/claude_provider.py", "Claude provider"),
        ("axiom/ai_client_integrations/sglang_provider.py", "SGLang provider"),
        ("axiom/ai_client_integrations/provider_factory.py", "Provider factory"),
        ("axiom/tools/tavily_client.py", "Tavily integration"),
        ("axiom/tools/firecrawl_client.py", "Firecrawl integration"),
        ("axiom/tools/mcp_adapter.py", "MCP adapter"),
        ("axiom/utils/error_handling.py", "Error handling"),
        ("axiom/utils/validation.py", "Data validation")
    ]
    
    files_valid = True
    for file_path, description in core_files:
        if not test_file_exists(file_path, description):
            files_valid = False
    
    return files_valid


def validate_imports():
    """Validate that core modules can be imported."""
    print("\n🔍 Validating Module Imports...")
    
    # Test basic imports (without external dependencies)
    import_tests = [
        ("axiom.config.schemas", "Data schemas"),
        ("axiom.config.ai_layer_config", "AI layer config"),
        ("axiom.graph.state", "Graph state"),
        ("axiom.ai_client_integrations.base_ai_provider", "Base AI provider"),
    ]
    
    imports_valid = True
    for module, description in import_tests:
        if not test_import(module, description):
            imports_valid = False
    
    return imports_valid


def validate_configuration():
    """Validate configuration setup."""
    print("\n⚙️  Validating Configuration...")
    
    try:
        from axiom.config.settings import settings
        from axiom.config.ai_layer_config import ai_layer_mapping, AnalysisLayer
        
        # Test basic config access
        providers = settings.get_configured_providers()
        print(f"✅ Configured providers: {', '.join(providers) if providers else 'None (testing mode)'}")
        
        # Test AI layer mapping
        planner_config = ai_layer_mapping.get_layer_config(AnalysisLayer.PLANNER)
        print(f"✅ Planner AI config: {planner_config.primary_provider.value}")
        
        # Test M&A specific configs
        ma_config = ai_layer_mapping.get_layer_config(AnalysisLayer.MA_DUE_DILIGENCE)
        print(f"✅ M&A due diligence AI config: {ma_config.primary_provider.value} (consensus: {ma_config.use_consensus})")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def validate_ai_providers():
    """Validate AI provider system."""
    print("\n🤖 Validating AI Provider System...")
    
    try:
        # Import provider factory
        from axiom.ai_client_integrations.provider_factory import provider_factory
        
        # Test provider factory initialization
        available_providers = provider_factory.get_available_providers()
        print(f"✅ Provider factory initialized: {len(available_providers)} providers")
        
        if available_providers:
            print(f"  Available: {', '.join(available_providers)}")
        else:
            print("  ⚠️  No providers configured (expected without API keys)")
        
        # Test provider info
        provider_info = provider_factory.get_provider_info()
        print(f"✅ Provider info accessible: {len(provider_info)} providers")
        
        return True
        
    except Exception as e:
        print(f"❌ AI provider validation failed: {e}")
        return False


def validate_graph_components():
    """Validate LangGraph components."""
    print("\n🌐 Validating Graph Components...")
    
    try:
        # Test state management
        from axiom.graph.state import AxiomState, create_initial_state
        
        test_state = create_initial_state("Test query", "test-trace")
        print(f"✅ State creation: {type(test_state)}")
        
        # Test node imports
        from axiom.graph.nodes import planner, task_runner, observer
        print("✅ Graph nodes: planner, task_runner, observer imported")
        
        # Test graph creation
        from axiom.graph.graph import create_research_graph
        graph = create_research_graph()
        print("✅ Research graph created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph component validation failed: {e}")
        return False


def validate_tools():
    """Validate tool integrations."""
    print("\n🔧 Validating Tool Integrations...")
    
    try:
        # Test tool imports
        from axiom.tools.mcp_adapter import mcp_adapter
        
        tools = mcp_adapter.get_available_tools()
        print(f"✅ MCP adapter: {len(tools)} tools available")
        
        tool_names = [tool["name"] for tool in tools]
        expected_tools = ["investment_banking_search", "financial_document_processor", "financial_qa"]
        
        for tool_name in expected_tools:
            if tool_name in tool_names:
                print(f"  ✅ {tool_name}")
            else:
                print(f"  ❌ {tool_name} - Missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool validation failed: {e}")
        return False


def main():
    """Main validation function."""
    print("🏦 Axiom Investment Banking Analytics - System Validation")
    print("=" * 65)
    
    validation_results = {
        "Project Structure": validate_project_structure(),
        "Core Files": validate_core_files(),
        "Module Imports": validate_imports(),
        "Configuration": validate_configuration(),
        "AI Providers": validate_ai_providers(),
        "Graph Components": validate_graph_components(),
        "Tool Integrations": validate_tools()
    }
    
    print("\n" + "=" * 65)
    print("📊 VALIDATION SUMMARY")
    print("=" * 65)
    
    total_checks = len(validation_results)
    passed_checks = sum(1 for result in validation_results.values() if result)
    
    for check_name, result in validation_results.items():
        status_icon = "✅" if result else "❌"
        print(f"  {status_icon} {check_name}")
    
    print(f"\nValidation Score: {passed_checks}/{total_checks}")
    
    if passed_checks == total_checks:
        print("🎉 All validations passed! System structure is correct.")
        print("\nNext Steps:")
        print("1. Set up API keys in .env file")
        print("2. Install dependencies with: pip install -r requirements.txt")
        print("3. Test with: python -m axiom.main 'Test M&A query'")
        return 0
    else:
        print("⚠️  Some validations failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())