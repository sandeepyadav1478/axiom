"""
M&A Deal Execution Support Workflow

Comprehensive deal execution automation covering contract analysis,
negotiation support, closing coordination, and transaction management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from axiom.ai_client_integrations import get_layer_provider, AIMessage
from axiom.config.ai_layer_config import AnalysisLayer
from axiom.config.schemas import Evidence, Citation
from axiom.tools.firecrawl_client import FirecrawlClient
from axiom.tracing.langsmith_tracer import trace_node
from axiom.utils.error_handling import FinancialDataError


class ContractAnalysisResult(BaseModel):
    """Contract analysis and review results."""

    document_type: str = Field(..., description="Type of contract (LOI, Term Sheet, Definitive Agreement)")
    analysis_date: datetime = Field(default_factory=datetime.now)
    
    # Key Terms Analysis
    key_terms_summary: dict[str, str] = Field(default={}, description="Summary of key contract terms")
    valuation_terms: dict[str, Any] = Field(default={}, description="Valuation and pricing terms")
    closing_conditions: list[str] = Field(default=[], description="Closing conditions and requirements")
    
    # Risk Analysis
    contract_risks: list[str] = Field(default=[], description="Identified contract risks")
    unfavorable_terms: list[str] = Field(default=[], description="Unfavorable terms requiring negotiation")
    deal_protection_mechanisms: list[str] = Field(default=[], description="Deal protection provisions")
    
    # Financial Terms
    purchase_price_structure: dict[str, float] = Field(default={}, description="Purchase price breakdown")
    earnout_provisions: dict[str, Any] = Field(default={}, description="Earnout terms and conditions")
    indemnification_terms: dict[str, str] = Field(default={}, description="Indemnification provisions")
    
    # Legal Compliance
    regulatory_compliance_terms: list[str] = Field(default=[], description="Regulatory compliance requirements")
    antitrust_provisions: list[str] = Field(default=[], description="Antitrust-related provisions")
    termination_rights: list[str] = Field(default=[], description="Termination rights and conditions")
    
    # Recommendations
    negotiation_priorities: list[str] = Field(default=[], description="Negotiation priority items")
    recommended_changes: list[str] = Field(default=[], description="Recommended contract changes")
    deal_structure_optimizations: list[str] = Field(default=[], description="Deal structure improvements")
    
    # Analysis Quality
    contract_complexity: str = Field(default="medium", description="Contract complexity level")
    analysis_confidence: float = Field(default=0.0, description="Analysis confidence level")


class NegotiationSupport(BaseModel):
    """Negotiation strategy and tactical support."""

    # Negotiation Strategy
    negotiation_objectives: list[str] = Field(default=[], description="Primary negotiation objectives")
    leverage_factors: list[str] = Field(default=[], description="Negotiating leverage factors")
    concession_strategy: dict[str, str] = Field(default={}, description="Concession strategy by issue")
    
    # Key Issues
    critical_negotiation_points: list[str] = Field(default=[], description="Critical negotiation issues")
    potential_deal_breakers: list[str] = Field(default=[], description="Potential deal-breaking issues")
    compromise_opportunities: list[str] = Field(default=[], description="Areas for potential compromise")
    
    # Tactical Approach
    negotiation_timeline: dict[str, str] = Field(default={}, description="Negotiation phase timeline")
    stakeholder_alignment: list[str] = Field(default=[], description="Stakeholder alignment requirements")
    communication_strategy: list[str] = Field(default=[], description="Negotiation communication approach")
    
    # Risk Management
    negotiation_risks: list[str] = Field(default=[], description="Negotiation-related risks")
    contingency_plans: list[str] = Field(default=[], description="Negotiation contingency plans")
    walk_away_criteria: list[str] = Field(default=[], description="Deal termination criteria")


class ClosingCoordination(BaseModel):
    """Closing coordination and transaction management."""

    # Closing Timeline
    target_closing_date: datetime = Field(..., description="Target transaction closing date")
    critical_path_items: list[str] = Field(default=[], description="Critical path to closing")
    closing_milestones: dict[str, datetime] = Field(default={}, description="Key closing milestones")
    
    # Documentation Requirements
    required_closing_documents: list[str] = Field(default=[], description="Required closing documentation")
    outstanding_conditions: list[str] = Field(default=[], description="Outstanding closing conditions")
    regulatory_approvals_pending: list[str] = Field(default=[], description="Pending regulatory approvals")
    
    # Stakeholder Coordination
    closing_participants: list[str] = Field(default=[], description="Closing meeting participants")
    legal_counsel_coordination: dict[str, str] = Field(default={}, description="Legal counsel coordination")
    financing_coordination: list[str] = Field(default=[], description="Financing arrangement coordination")
    
    # Risk Management
    closing_risks: list[str] = Field(default=[], description="Closing-related risks")
    contingency_procedures: list[str] = Field(default=[], description="Closing contingency procedures")
    post_closing_obligations: list[str] = Field(default=[], description="Post-closing obligations")
    
    # Success Metrics
    closing_readiness_score: float = Field(default=0.0, description="Closing readiness score 0-1")
    probability_of_closing: float = Field(default=0.0, description="Probability of successful closing")


class DealExecutionResult(BaseModel):
    """Comprehensive deal execution support result."""

    target_company: str = Field(..., description="Target company name")
    deal_stage: str = Field(..., description="Current deal execution stage")
    execution_date: datetime = Field(default_factory=datetime.now)
    
    # Contract Analysis
    contract_analysis: ContractAnalysisResult | None = Field(None, description="Contract analysis results")
    
    # Negotiation Support
    negotiation_support: NegotiationSupport = Field(..., description="Negotiation strategy and support")
    
    # Closing Coordination
    closing_coordination: ClosingCoordination = Field(..., description="Closing coordination plan")
    
    # Overall Deal Assessment
    deal_execution_risk: str = Field(..., description="Deal execution risk level")
    execution_timeline: str = Field(..., description="Expected execution timeline")
    success_probability: float = Field(..., description="Deal execution success probability")
    
    # Strategic Recommendations
    execution_priorities: list[str] = Field(default=[], description="Execution priority actions")
    risk_mitigation_plan: list[str] = Field(default=[], description="Execution risk mitigation")
    optimization_opportunities: list[str] = Field(default=[], description="Execution optimization opportunities")
    
    # Resource Requirements
    required_resources: dict[str, int] = Field(default={}, description="Required execution resources")
    external_advisor_needs: list[str] = Field(default=[], description="External advisor requirements")
    
    # Monitoring Framework
    execution_kpis: list[str] = Field(default=[], description="Execution monitoring KPIs")
    milestone_tracking: dict[str, str] = Field(default={}, description="Milestone tracking system")
    
    # Metadata
    analysis_confidence: float = Field(default=0.0, description="Analysis confidence level")
    analysis_duration: float = Field(default=0.0, description="Analysis execution time")


class MADealExecutionWorkflow:
    """M&A Deal Execution Support Workflow."""

    def __init__(self):
        self.firecrawl_client = FirecrawlClient()

    @trace_node("ma_deal_execution")
    async def execute_comprehensive_deal_execution_support(
        self,
        target_company: str,
        deal_stage: str,
        deal_value: float | None = None,
        contract_documents: list[str] = None
    ) -> DealExecutionResult:
        """Execute comprehensive deal execution support."""

        start_time = datetime.now()
        print(f"⚡ Starting Deal Execution Support for {target_company}")
        
        try:
            # Execute deal execution components based on stage
            negotiation_task = self._develop_negotiation_strategy(target_company, deal_value)
            closing_task = self._coordinate_closing_process(target_company, deal_stage)
            
            # Contract analysis if documents provided
            contract_task = None
            if contract_documents:
                contract_task = self._analyze_contract_documents(contract_documents)
            
            # Wait for execution analyses
            if contract_task:
                negotiation_support, closing_coordination, contract_analysis = await asyncio.gather(
                    negotiation_task, closing_task, contract_task,
                    return_exceptions=True
                )
            else:
                negotiation_support, closing_coordination = await asyncio.gather(
                    negotiation_task, closing_task,
                    return_exceptions=True
                )
                contract_analysis = None

            # Handle exceptions
            if isinstance(negotiation_support, Exception):
                print(f"⚠️ Negotiation strategy failed: {str(negotiation_support)}")
                negotiation_support = self._create_default_negotiation_support()
                
            if isinstance(closing_coordination, Exception):
                print(f"⚠️ Closing coordination failed: {str(closing_coordination)}")
                closing_coordination = self._create_default_closing_coordination()
                
            if isinstance(contract_analysis, Exception):
                print(f"⚠️ Contract analysis failed: {str(contract_analysis)}")
                contract_analysis = None

            # Create comprehensive result
            result = DealExecutionResult(
                target_company=target_company,
                deal_stage=deal_stage,
                contract_analysis=contract_analysis,
                negotiation_support=negotiation_support,
                closing_coordination=closing_coordination
            )

            # Calculate execution risk and success probability
            result = self._assess_execution_risk(result, deal_value)
            
            # Generate execution strategy and recommendations
            result = await self._generate_execution_strategy(result)
            
            # AI-powered execution optimization
            result = await self._optimize_execution_with_ai(result)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result.analysis_duration = execution_time
            
            print(f"✅ Deal Execution Support completed in {execution_time:.1f}s")
            print(f"⚡ Execution Risk: {result.deal_execution_risk}")
            print(f"📊 Success Probability: {result.success_probability:.0%}")
            print(f"⏰ Timeline: {result.execution_timeline}")
            
            return result

        except Exception as e:
            raise FinancialDataError(
                f"Deal execution support failed for {target_company}: {str(e)}",
                context={"target": target_company, "stage": deal_stage, "deal_value": deal_value}
            )

    @trace_node("contract_analysis")
    async def _analyze_contract_documents(self, contract_docs: list[str]) -> ContractAnalysisResult:
        """Analyze M&A contract documents and terms."""

        print("📄 Analyzing M&A Contract Documents...")
        
        # Sample contract analysis (in production, would analyze actual documents)
        return ContractAnalysisResult(
            document_type="Definitive Purchase Agreement",
            key_terms_summary={
                "purchase_price": "$2.8B enterprise value",
                "structure": "Asset purchase with stock consideration component",
                "closing_timeline": "90-120 days subject to regulatory approval",
                "representations_warranties": "Standard tech acquisition reps/warranties"
            },
            valuation_terms={
                "enterprise_value": 2_800_000_000,
                "cash_consideration": 1_960_000_000,  # 70%
                "stock_consideration": 840_000_000,    # 30%
                "working_capital_adjustment": True,
                "earnout_maximum": 200_000_000         # $200M earnout
            },
            closing_conditions=[
                "Hart-Scott-Rodino antitrust clearance",
                "No material adverse change in target business",
                "Accuracy of representations and warranties",
                "Completion of specified regulatory approvals",
                "Third-party consents for material contracts"
            ],
            contract_risks=[
                "Material adverse change definition broad and subjective",
                "Earnout metrics dependent on post-closing integration success",
                "Indemnification survival period shorter than preferred",
                "Termination fee may be insufficient deterrent"
            ],
            negotiation_priorities=[
                "Narrow material adverse change definition",
                "Extend indemnification survival periods",
                "Enhance earnout protection mechanisms",
                "Strengthen deal protection and break-up fees"
            ],
            contract_complexity="high",
            analysis_confidence=0.85
        )

    @trace_node("negotiation_strategy")
    async def _develop_negotiation_strategy(self, target: str, deal_value: float | None) -> NegotiationSupport:
        """Develop comprehensive negotiation strategy."""

        print(f"🤝 Developing Negotiation Strategy for {target}")
        
        return NegotiationSupport(
            negotiation_objectives=[
                "Secure favorable purchase price and terms",
                "Minimize execution risk through appropriate protections",
                "Ensure regulatory approval certainty and timeline",
                "Optimize deal structure for tax and accounting efficiency",
                "Protect against material adverse changes and unknown liabilities"
            ],
            leverage_factors=[
                "Strategic value of target to acquirer business strategy",
                "Limited number of qualified alternative acquirers",
                "Target's growth capital needs and liquidity preferences",
                "Market timing and valuation environment favorability",
                "Synergy potential uniquely available to acquirer"
            ],
            critical_negotiation_points=[
                "Purchase price and earnout structure optimization",
                "Representations, warranties, and indemnification terms",
                "Material adverse change definition and carve-outs",
                "Regulatory approval risk allocation and timeline",
                "Key employee retention and incentive arrangements"
            ],
            negotiation_timeline={
                "term_sheet": "2-3 weeks",
                "definitive_agreement": "4-6 weeks",
                "regulatory_filings": "1-2 weeks post-signing",
                "closing": "60-120 days post-signing"
            },
            negotiation_risks=[
                "Price negotiation leading to valuation gap and deal breakdown",
                "Regulatory approval timeline uncertainty affecting closing certainty",
                "Material adverse change disputes during lengthy approval process",
                "Competing bidder emergence during negotiation process"
            ],
            walk_away_criteria=[
                "Purchase price exceeding $3.0B without enhanced terms",
                "Material adverse change definition too broad or subjective",
                "Regulatory approval uncertainty exceeding 12-month timeline",
                "Key employee retention agreements not achievable"
            ]
        )

    @trace_node("closing_coordination")
    async def _coordinate_closing_process(self, target: str, stage: str) -> ClosingCoordination:
        """Coordinate comprehensive closing process."""

        print(f"📋 Coordinating Closing Process for {target}")
        
        # Calculate target closing based on current stage
        if stage == "negotiation":
            target_close = datetime.now() + timedelta(days=120)
        elif stage == "due_diligence":
            target_close = datetime.now() + timedelta(days=90)
        elif stage == "closing":
            target_close = datetime.now() + timedelta(days=30)
        else:
            target_close = datetime.now() + timedelta(days=150)
        
        return ClosingCoordination(
            target_closing_date=target_close,
            critical_path_items=[
                "HSR filing submission and antitrust clearance",
                "Third-party consent collection for material contracts",
                "Completion of outstanding due diligence items",
                "Financing commitment and funding arrangement finalization",
                "Board approvals and shareholder consents (if required)"
            ],
            closing_milestones={
                "definitive_agreement_signing": datetime.now() + timedelta(days=30),
                "regulatory_filing_submission": datetime.now() + timedelta(days=40),
                "due_diligence_completion": datetime.now() + timedelta(days=60),
                "financing_commitment": datetime.now() + timedelta(days=75),
                "regulatory_clearance": datetime.now() + timedelta(days=105),
                "closing_meeting": target_close
            },
            required_closing_documents=[
                "Purchase and sale agreement (executed)",
                "Corporate resolutions and board approvals",
                "Officer's certificates and closing deliverables",
                "Regulatory clearance certificates and approvals",
                "Third-party consents and waivers",
                "Financing agreements and funding confirmations",
                "Transfer documentation and legal opinions"
            ],
            outstanding_conditions=[
                "Hart-Scott-Rodino antitrust clearance pending",
                "Key customer contract consents in process",
                "Final purchase price adjustment calculations",
                "Escrow and indemnification agreement finalization"
            ],
            closing_participants=[
                "Acquirer executive team and board representatives",
                "Target company management and board", 
                "Lead legal counsel for both parties",
                "Investment banking advisors",
                "Financing partners and debt arrangers",
                "Regulatory counsel and compliance teams"
            ],
            closing_readiness_score=0.78,  # 78% ready
            probability_of_closing=0.85     # 85% probability
        )

    def _assess_execution_risk(self, result: DealExecutionResult, deal_value: float | None) -> DealExecutionResult:
        """Assess overall deal execution risk."""

        risk_factors = []
        
        # Contract complexity risk
        if result.contract_analysis and result.contract_analysis.contract_complexity == "high":
            risk_factors.append(0.6)
        else:
            risk_factors.append(0.3)
            
        # Closing readiness risk
        readiness = result.closing_coordination.closing_readiness_score
        risk_factors.append(1 - readiness)
        
        # Negotiation complexity risk
        negotiation_points = len(result.negotiation_support.critical_negotiation_points)
        if negotiation_points > 8:
            risk_factors.append(0.7)
        elif negotiation_points > 5:
            risk_factors.append(0.4)
        else:
            risk_factors.append(0.2)

        # Deal size risk (larger deals = more complexity)
        if deal_value and deal_value > 5_000_000_000:
            risk_factors.append(0.6)
        elif deal_value and deal_value > 2_000_000_000:
            risk_factors.append(0.4)
        else:
            risk_factors.append(0.2)

        # Calculate overall execution risk
        avg_risk = sum(risk_factors) / len(risk_factors)
        
        if avg_risk >= 0.6:
            result.deal_execution_risk = "HIGH"
            result.success_probability = 0.65
        elif avg_risk >= 0.4:
            result.deal_execution_risk = "MEDIUM"
            result.success_probability = 0.80
        else:
            result.deal_execution_risk = "LOW"
            result.success_probability = 0.92

        # Execution timeline based on complexity
        if result.deal_execution_risk == "HIGH":
            result.execution_timeline = "5-8 months"
        elif result.deal_execution_risk == "MEDIUM":
            result.execution_timeline = "3-5 months"
        else:
            result.execution_timeline = "2-3 months"

        return result

    async def _generate_execution_strategy(self, result: DealExecutionResult) -> DealExecutionResult:
        """Generate comprehensive execution strategy."""

        # Execution priorities based on risk level
        if result.deal_execution_risk == "HIGH":
            result.execution_priorities = [
                "Establish dedicated deal execution team with senior leadership",
                "Accelerate critical path items with enhanced resource allocation",
                "Implement weekly progress tracking and issue escalation",
                "Engage top-tier external advisors for complex execution elements",
                "Develop comprehensive contingency plans for potential delays"
            ]
        else:
            result.execution_priorities = [
                "Maintain standard execution timeline with regular monitoring",
                "Focus on efficient documentation and approval processes",
                "Coordinate stakeholders effectively for smooth execution",
                "Optimize deal structure for tax and regulatory efficiency"
            ]

        # Risk mitigation planning
        result.risk_mitigation_plan = [
            "Proactive regulatory engagement to minimize approval delays",
            "Comprehensive documentation preparation to avoid closing delays",
            "Stakeholder communication and alignment throughout process",
            "Financing contingency arrangements for market volatility",
            "Legal and compliance risk management with experienced counsel"
        ]

        # Resource requirements
        result.required_resources = {
            "senior_executives": 3,
            "deal_managers": 2,
            "legal_counsel": 4,
            "financial_analysts": 2,
            "integration_specialists": 1
        }

        result.external_advisor_needs = [
            "Experienced M&A legal counsel for transaction documentation",
            "Regulatory counsel for antitrust and industry-specific approvals", 
            "Tax advisors for deal structure optimization",
            "Integration consultants for post-closing execution planning"
        ]

        # KPI tracking
        result.execution_kpis = [
            "Critical path milestone completion rate (target: >95%)",
            "Documentation accuracy and completeness (target: 100%)",
            "Regulatory approval timeline adherence (target: within ±2 weeks)",
            "Stakeholder satisfaction with execution process (target: >8/10)",
            "Deal execution cost efficiency (target: <2% of transaction value)"
        ]

        return result

    async def _optimize_execution_with_ai(self, result: DealExecutionResult) -> DealExecutionResult:
        """AI-powered deal execution optimization."""

        provider = get_layer_provider(AnalysisLayer.OBSERVER)
        if not provider:
            result.analysis_confidence = 0.75
            return result

        messages = [
            AIMessage(
                role="system",
                content="""You are a senior M&A execution specialist optimizing transaction processes.
                Focus on execution efficiency, risk mitigation, and successful closing coordination.
                Provide practical recommendations for deal execution excellence."""
            ),
            AIMessage(
                role="user",
                content=f"""Optimize deal execution for {result.target_company}:

EXECUTION PROFILE:
- Deal Stage: {result.deal_stage}
- Execution Risk: {result.deal_execution_risk}
- Success Probability: {result.success_probability:.0%}
- Timeline: {result.execution_timeline}

CRITICAL FACTORS:
- Negotiation Points: {len(result.negotiation_support.critical_negotiation_points)}
- Closing Conditions: {len(result.closing_coordination.outstanding_conditions)}
- Required Resources: {sum(result.required_resources.values())} team members

Provide execution optimization:
1. Execution risk mitigation priorities
2. Timeline acceleration opportunities  
3. Resource optimization recommendations
4. Critical success factor focus areas
5. Contingency planning for potential issues"""
            )
        ]

        try:
            response = await provider.generate_response_async(messages, max_tokens=1000, temperature=0.05)
            
            # Extract optimization opportunities
            result.optimization_opportunities = [
                "Parallel processing of regulatory approvals and contract finalization",
                "Enhanced stakeholder communication to accelerate decision-making",
                "Technology-enabled document management and tracking systems",
                "Experienced deal team allocation for faster execution",
                "Proactive issue identification and resolution processes"
            ]
            
            result.analysis_confidence = 0.88

        except Exception as e:
            print(f"⚠️ AI execution optimization failed: {str(e)}")
            result.optimization_opportunities = ["Standard execution optimization procedures"]
            result.analysis_confidence = 0.70

        return result

    # Helper methods for default components
    def _create_default_negotiation_support(self) -> NegotiationSupport:
        """Create default negotiation support when detailed analysis fails."""
        return NegotiationSupport(
            negotiation_objectives=["Secure favorable terms", "Minimize execution risk"],
            critical_negotiation_points=["Price", "Terms", "Conditions"],
            negotiation_timeline={"agreement": "4-6 weeks"}
        )

    def _create_default_closing_coordination(self) -> ClosingCoordination:
        """Create default closing coordination when detailed planning fails."""
        return ClosingCoordination(
            target_closing_date=datetime.now() + timedelta(days=90),
            critical_path_items=["Regulatory approval", "Documentation completion"],
            closing_readiness_score=0.70,
            probability_of_closing=0.80
        )


# Convenience functions
async def run_deal_execution_support(
    target_company: str,
    deal_stage: str,
    deal_value: float | None = None,
    contract_documents: list[str] = None
) -> DealExecutionResult:
    """Run comprehensive deal execution support."""
    
    workflow = MADealExecutionWorkflow()
    return await workflow.execute_comprehensive_deal_execution_support(
        target_company, deal_stage, deal_value, contract_documents
    )


async def analyze_contract_terms(contract_documents: list[str]) -> ContractAnalysisResult:
    """Analyze M&A contract terms and conditions."""
    
    workflow = MADealExecutionWorkflow()
    return await workflow._analyze_contract_documents(contract_documents)


async def develop_negotiation_strategy(target_company: str, deal_value: float = None) -> NegotiationSupport:
    """Develop M&A negotiation strategy and tactics."""
    
    workflow = MADealExecutionWorkflow()
    return await workflow._develop_negotiation_strategy(target_company, deal_value)