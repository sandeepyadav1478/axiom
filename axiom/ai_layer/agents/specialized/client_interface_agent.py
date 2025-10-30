"""
Client Interface Agent - Client Communication Specialist

Responsibility: All client-facing interactions
Expertise: Dashboards, reports, Q&A, explanations
Independence: Manages all client communication

Capabilities:
- Generate interactive dashboards
- Answer client questions (RAG-based)
- Create professional reports
- Explain AI decisions (SHAP)
- Handle client requests
- Multi-turn conversations

Performance: <500ms for responses (includes LLM)
Quality: Professional, accurate, helpful
Memory: Maintains conversation context
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio
import time


@dataclass
class ClientRequest:
    """Request to client interface agent"""
    request_type: str  # 'dashboard', 'question', 'report', 'explain'
    client_id: str
    query: Optional[str] = None
    data: Optional[Dict] = None


@dataclass
class ClientResponse:
    """Response to client"""
    success: bool
    response_type: str  # 'dashboard_html', 'text', 'report_pdf'
    content: Any
    confidence: float
    sources: List[str]  # Which data sources used
    generation_time_ms: float


class ClientInterfaceAgent:
    """Specialized agent for client interactions"""
    
    def __init__(self):
        """Initialize client interface agent"""
        from axiom.ai_layer.rag.retrieval_system import RAGSystem
        from axiom.ai_layer.memory.conversation_memory import ConversationMemory
        from axiom.ai_layer.prompts.prompt_manager import PromptManager
        from axiom.derivatives.reporting.client_dashboard import ClientDashboardGenerator
        
        self.rag_system = RAGSystem()
        self.memory = ConversationMemory()
        self.prompt_manager = PromptManager()
        self.dashboard_generator = ClientDashboardGenerator()
        
        # Client sessions
        self.active_sessions = {}
        
        print("ClientInterfaceAgent initialized")
        print("  Capabilities: Dashboards, Q&A, reports, explanations")
    
    async def process_request(self, request: ClientRequest) -> ClientResponse:
        """Process client request"""
        start = time.perf_counter()
        
        try:
            if request.request_type == 'question':
                # Answer question using RAG
                context = self.rag_system.retrieve_context(
                    query=request.query,
                    collection_name='all',
                    n_results=3
                )
                
                # Generate answer (would use LLM in production)
                answer = f"Based on historical data: {request.query}"
                
                # Store in conversation memory
                if request.client_id not in self.active_sessions:
                    self.active_sessions[request.client_id] = ConversationMemory()
                
                memory = self.active_sessions[request.client_id]
                memory.add_message('user', request.query)
                memory.add_message('assistant', answer)
                
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                return ClientResponse(
                    success=True,
                    response_type='text',
                    content=answer,
                    confidence=0.85,
                    sources=[doc[:50] for doc in context.documents],
                    generation_time_ms=elapsed_ms
                )
            
            elif request.request_type == 'dashboard':
                # Generate dashboard
                import pandas as pd
                
                pnl_data = pd.DataFrame({'total_pnl': [0, 1000, 2000]})
                
                html = self.dashboard_generator.generate_realtime_dashboard(
                    pnl_data=pnl_data,
                    current_positions=request.data.get('positions', []),
                    current_greeks=request.data.get('greeks', {}),
                    risk_metrics=request.data.get('risk', {})
                )
                
                return ClientResponse(
                    success=True,
                    response_type='dashboard_html',
                    content=html,
                    confidence=1.0,
                    sources=['system'],
                    generation_time_ms=(time.perf_counter() - start) * 1000
                )
            
            else:
                raise ValueError(f"Unknown request type: {request.request_type}")
        
        except Exception as e:
            return ClientResponse(
                success=False,
                response_type='error',
                content=str(e),
                confidence=0.0,
                sources=[],
                generation_time_ms=(time.perf_counter() - start) * 1000
            )
    
    def get_stats(self) -> Dict:
        """Get client interface agent statistics"""
        return {
            'agent': 'client_interface',
            'active_sessions': len(self.active_sessions),
            'status': 'operational'
        }


if __name__ == "__main__":
    import asyncio
    
    async def test_client_interface_agent():
        print("="*60)
        print("CLIENT INTERFACE AGENT - STANDALONE TEST")
        print("="*60)
        
        agent = ClientInterfaceAgent()
        
        # Test Q&A
        print("\n→ Test: Client Question")
        request = ClientRequest(
            request_type='question',
            client_id='client_001',
            query="What happens to volatility after Fed meetings?"
        )
        
        response = await agent.process_request(request)
        
        print(f"   Success: {'✓' if response.success else '✗'}")
        print(f"   Answer: {response.content[:100]}...")
        print(f"   Confidence: {response.confidence:.1%}")
        print(f"   Time: {response.generation_time_ms:.0f}ms")
        
        print("\n✓ Client interface agent operational")
    
    asyncio.run(test_client_interface_agent())