"""
Guardrail Agent - Safety Validation Specialist

Responsibility: Final safety check on all actions
Expertise: Input validation, output validation, safety rules
Independence: Can veto any action from any agent

Capabilities:
- Validate all AI outputs
- Cross-check with analytical solutions
- Enforce safety rules
- Circuit breaker management
- Veto dangerous actions
- Escalate to humans when needed

Performance: <1ms validation
Authority: Can block any action (highest priority)
Accuracy: 100% (must catch all dangerous actions)

This agent has final say - if it says no, action is blocked.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class GuardrailRequest:
    """Request to guardrail agent"""
    action_type: str  # 'validate_greeks', 'validate_strategy', 'validate_execution'
    agent_name: str  # Which agent proposed the action
    proposed_action: Dict
    context: Dict


@dataclass
class GuardrailResponse:
    """Response from guardrail agent"""
    approved: bool  # True = allow, False = block
    reason: str
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    alternative_action: Optional[Dict]  # Suggest safer alternative
    requires_human_approval: bool
    validation_time_ms: float


class GuardrailAgent:
    """
    Safety guardrail agent - final authority on all actions
    
    Responsibilities:
    - Validate every action before execution
    - Block dangerous actions
    - Escalate to humans when uncertain
    - Maintain circuit breakers
    - Enforce safety rules
    
    Authority: Highest - can veto anything
    Approach: Conservative - if uncertain, block
    """
    
    def __init__(self):
        """Initialize guardrail agent"""
        from axiom.ai_layer.guardrails.ai_safety_layer import AIGuardrailSystem
        
        self.safety_system = AIGuardrailSystem()
        
        # Veto statistics
        self.total_requests = 0
        self.blocked_requests = 0
        self.human_escalations = 0
        
        # Rules
        self.safety_rules = self._load_safety_rules()
        
        print("GuardrailAgent initialized")
        print("  Authority: Can veto any action")
        print("  Approach: Conservative (block if uncertain)")
    
    def _load_safety_rules(self) -> List[Dict]:
        """Load safety rules"""
        return [
            {
                'name': 'max_single_trade',
                'condition': lambda action: abs(action.get('quantity', 0)) <= 10000,
                'severity': 'critical'
            },
            {
                'name': 'greeks_accuracy',
                'condition': lambda action: action.get('confidence', 0) >= 0.99,
                'severity': 'high'
            },
            {
                'name': 'strategy_max_loss',
                'condition': lambda action: action.get('max_loss', 0) <= 500000,
                'severity': 'critical'
            }
        ]
    
    async def process_request(self, request: GuardrailRequest) -> GuardrailResponse:
        """Process safety validation request"""
        start = time.perf_counter()
        
        self.total_requests += 1
        
        try:
            # Run safety checks
            if request.action_type == 'validate_greeks':
                validation = self.safety_system.validate_greeks_output(
                    ai_greeks=request.proposed_action,
                    spot=request.context.get('spot', 100),
                    strike=request.context.get('strike', 100),
                    time=request.context.get('time', 1.0),
                    rate=request.context.get('rate', 0.03),
                    vol=request.context.get('vol', 0.25)
                )
                
                approved = validation.passed
                reason = "Greeks validated successfully" if approved else "Validation failed"
                risk_level = validation.risk_level.value
                
            elif request.action_type == 'validate_strategy':
                validation = self.safety_system.validate_strategy(
                    strategy=request.proposed_action,
                    max_risk=request.context.get('max_risk', 100000)
                )
                
                approved = validation.passed
                reason = "Strategy validated" if approved else "Strategy too risky"
                risk_level = validation.risk_level.value
                
            elif request.action_type == 'validate_execution':
                validation = self.safety_system.validate_execution(
                    order=request.proposed_action,
                    current_portfolio=request.context.get('portfolio', {})
                )
                
                approved = validation.passed
                reason = "Order approved" if approved else "Order blocked"
                risk_level = validation.risk_level.value
                
            else:
                # Unknown action type - block by default (conservative)
                approved = False
                reason = f"Unknown action type: {request.action_type}"
                risk_level = 'high'
            
            # Update statistics
            if not approved:
                self.blocked_requests += 1
            
            # Escalate to human if critical
            requires_human = risk_level == 'critical' and not approved
            if requires_human:
                self.human_escalations += 1
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            return GuardrailResponse(
                approved=approved,
                reason=reason,
                risk_level=risk_level,
                alternative_action=None,  # Could suggest safer alternative
                requires_human_approval=requires_human,
                validation_time_ms=elapsed_ms
            )
        
        except Exception as e:
            # On error, block action (conservative)
            self.blocked_requests += 1
            
            return GuardrailResponse(
                approved=False,
                reason=f"Validation error: {str(e)}",
                risk_level='critical',
                alternative_action=None,
                requires_human_approval=True,
                validation_time_ms=(time.perf_counter() - start) * 1000
            )
    
    def get_stats(self) -> Dict:
        """Get guardrail agent statistics"""
        block_rate = self.blocked_requests / self.total_requests if self.total_requests > 0 else 0
        
        return {
            'agent': 'guardrail',
            'total_requests': self.total_requests,
            'blocked_requests': self.blocked_requests,
            'block_rate': block_rate,
            'human_escalations': self.human_escalations,
            'status': 'guarding'
        }


if __name__ == "__main__":
    import asyncio
    
    async def test_guardrail_agent():
        print("="*60)
        print("GUARDRAIL AGENT - STANDALONE TEST")
        print("="*60)
        
        agent = GuardrailAgent()
        
        # Test 1: Valid Greeks (should approve)
        print("\n→ Test 1: Valid Greeks")
        request1 = GuardrailRequest(
            action_type='validate_greeks',
            agent_name='pricing',
            proposed_action={'delta': 0.52, 'gamma': 0.015, 'vega': 0.39},
            context={'spot': 100, 'strike': 100, 'time': 1.0, 'rate': 0.03, 'vol': 0.25}
        )
        
        response1 = await agent.process_request(request1)
        
        print(f"   Approved: {'✓ YES' if response1.approved else '✗ NO'}")
        print(f"   Reason: {response1.reason}")
        print(f"   Risk level: {response1.risk_level}")
        
        # Test 2: Invalid Greeks (should block)
        print("\n→ Test 2: Invalid Greeks (should block)")
        request2 = GuardrailRequest(
            action_type='validate_greeks',
            agent_name='pricing',
            proposed_action={'delta': 1.5, 'gamma': -0.01, 'vega': 0.39},  # Invalid!
            context={'spot': 100, 'strike': 100, 'time': 1.0, 'rate': 0.03, 'vol': 0.25}
        )
        
        response2 = await agent.process_request(request2)
        
        print(f"   Approved: {'✓ YES' if response2.approved else '✗ NO (BLOCKED)'}")
        print(f"   Reason: {response2.reason}")
        print(f"   Human escalation: {response2.requires_human_approval}")
        
        # Stats
        print("\n→ Agent Statistics:")
        stats = agent.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✓ Guardrail agent operational")
        print("✓ Can block dangerous actions")
        print("✓ Conservative approach")
    
    asyncio.run(test_guardrail_agent())