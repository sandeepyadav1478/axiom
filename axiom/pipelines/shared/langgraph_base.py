"""
Base LangGraph Pipeline Framework
All pipelines inherit from this class
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, TypedDict
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)


class BaseLangGraphPipeline(ABC):
    """
    Base class for all LangGraph-powered pipelines.
    
    Features:
    - Claude AI integration
    - LangGraph workflow management
    - Continuous execution mode
    - Error handling and retries
    """
    
    def __init__(self, pipeline_name: str):
        """Initialize pipeline."""
        self.pipeline_name = pipeline_name
        
        # Initialize Claude client
        self.claude = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            max_tokens=4096
        )
        
        # Pipeline configuration
        self.interval = int(os.getenv('PIPELINE_INTERVAL', '60'))
        
        logger.info(f"✅ {pipeline_name} initialized with Claude Sonnet 4")
        
        # Build workflow
        self.workflow = self.build_workflow()
        self.app = self.workflow.compile()
    
    @abstractmethod
    def build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def process_item(self, item: Any) -> Dict[str, Any]:
        """
        Process a single item through the workflow.
        Must be implemented by subclasses.
        """
        pass
    
    async def run_continuous(self, items: List[Any]):
        """
        Run pipeline in continuous mode.
        
        Args:
            items: List of items to process (symbols, events, etc.)
        """
        logger.info(f"Starting {self.pipeline_name} - processing {len(items)} items every {self.interval}s")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                logger.info(f"=== Cycle {cycle_count} ===")
                
                results = []
                for item in items:
                    try:
                        result = await self.process_item(item)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {item}: {e}")
                
                # Log cycle summary
                successful = len([r for r in results if r.get('success', False)])
                logger.info(f"Cycle {cycle_count} complete: {successful}/{len(items)} successful")
                
                # Wait for next cycle
                await asyncio.sleep(self.interval)
                
            except KeyboardInterrupt:
                logger.info(f"Stopping {self.pipeline_name}")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await asyncio.sleep(self.interval)
    
    def invoke_claude(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Invoke Claude with a prompt.
        
        Args:
            prompt: User prompt
            system: Optional system prompt
            
        Returns:
            Claude's response content
        """
        messages = [{"role": "user", "content": prompt}]
        
        if system:
            response = self.claude.invoke(
                messages,
                system=system
            )
        else:
            response = self.claude.invoke(messages)
        
        return response.content
    
    def close(self):
        """Cleanup resources."""
        logger.info(f"{self.pipeline_name} cleanup complete")


class PipelineState(TypedDict):
    """Base state for all pipelines."""
    item: Any
    success: bool
    error: Optional[str]
    metadata: Dict[str, Any]


# Common LangGraph node functions

def log_start(state: Dict) -> Dict:
    """Log pipeline start."""
    logger.info(f"Processing: {state.get('item')}")
    return state


def log_end(state: Dict) -> Dict:
    """Log pipeline completion."""
    if state.get('success'):
        logger.info(f"✅ Success: {state.get('item')}")
    else:
        logger.error(f"❌ Failed: {state.get('item')} - {state.get('error')}")
    return state


def handle_error(state: Dict) -> Dict:
    """Error handling node."""
    state['success'] = False
    if not state.get('error'):
        state['error'] = "Unknown error"
    return state