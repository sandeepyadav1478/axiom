"""
Post-Merger Integration (PMI) Planning & Execution Workflow

Comprehensive post-merger integration automation covering Day 1 readiness,
systems integration, cultural integration, synergy realization, and PMO management.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from axiom.integrations.ai_providers import AIMessage, get_layer_provider
from axiom.config.ai_layer_config import AnalysisLayer
from axiom.integrations.search_tools.tavily_client import TavilyClient
from axiom.tracing.langsmith_tracer import trace_node
from axiom.core.validation.error_handling import FinancialDataError


class IntegrationWorkstream(BaseModel):
    """Individual integration workstream definition."""

    workstream_name: str = Field(..., description="Integration workstream name")
    pmo_lead: str = Field(..., description="PMO workstream lead")
    duration_months: int = Field(..., description="Integration duration in months")

    # Key Milestones
    phase1_milestones: list[str] = Field(default=[], description="Phase 1 milestones (0-3 months)")
    phase2_milestones: list[str] = Field(default=[], description="Phase 2 milestones (3-6 months)")
    phase3_milestones: list[str] = Field(default=[], description="Phase 3 milestones (6-12 months)")

    # Success Metrics
    success_kpis: list[str] = Field(default=[], description="Key performance indicators")
    completion_criteria: list[str] = Field(default=[], description="Completion criteria")
    risk_mitigation: list[str] = Field(default=[], description="Workstream risk mitigation")

    # Resource Requirements
    team_size: int = Field(default=5, description="Required team size")
    budget_allocation: float = Field(default=1000000, description="Budget allocation ($)")
    external_support: bool = Field(default=False, description="External consultant required")

    # Dependencies
    critical_dependencies: list[str] = Field(default=[], description="Critical workstream dependencies")
    integration_complexity: str = Field(default="medium", description="Integration complexity level")


class Day1ReadinessPlan(BaseModel):
    """Day 1 operational readiness plan."""

    # Critical Day 1 Requirements
    systems_access: list[str] = Field(default=[], description="Required systems access")
    employee_communications: list[str] = Field(default=[], description="Employee communication checklist")
    customer_notifications: list[str] = Field(default=[], description="Customer notification requirements")
    vendor_communications: list[str] = Field(default=[], description="Vendor and supplier notifications")
    regulatory_notifications: list[str] = Field(default=[], description="Regulatory notification requirements")

    # Day 1 Success Metrics
    success_metrics: dict[str, str] = Field(default={}, description="Day 1 success criteria")
    contingency_plans: list[str] = Field(default=[], description="Day 1 contingency plans")

    # Communication Plan
    stakeholder_communications: dict[str, str] = Field(default={}, description="Stakeholder communication plan")
    press_release_timeline: str = Field(default="Day 1 morning", description="Press release timing")

    # Operational Readiness
    business_continuity_plan: list[str] = Field(default=[], description="Business continuity requirements")
    day1_leadership_structure: dict[str, str] = Field(default={}, description="Day 1 leadership assignments")


class SynergyRealizationPlan(BaseModel):
    """Synergy realization planning and tracking."""

    # Revenue Synergies
    revenue_synergies_timeline: dict[str, float] = Field(default={}, description="Revenue synergy realization timeline")
    cross_selling_plan: list[str] = Field(default=[], description="Cross-selling implementation plan")
    market_expansion_strategy: list[str] = Field(default=[], description="Market expansion approach")

    # Cost Synergies
    cost_synergies_timeline: dict[str, float] = Field(default={}, description="Cost synergy realization timeline")
    headcount_optimization: dict[str, Any] = Field(default={}, description="Headcount optimization plan")
    overhead_elimination: list[str] = Field(default=[], description="Overhead elimination initiatives")
    procurement_synergies: list[str] = Field(default=[], description="Procurement optimization plan")

    # Synergy Tracking
    synergy_kpis: list[str] = Field(default=[], description="Synergy realization KPIs")
    tracking_frequency: str = Field(default="monthly", description="Synergy tracking frequency")
    variance_thresholds: dict[str, float] = Field(default={}, description="Synergy variance alert thresholds")

    # Risk Management
    synergy_risks: list[str] = Field(default=[], description="Synergy realization risks")
    mitigation_strategies: list[str] = Field(default=[], description="Synergy risk mitigation")
    contingency_plans: list[str] = Field(default=[], description="Synergy shortfall contingencies")


class PMIExecutionPlan(BaseModel):
    """Comprehensive Post-Merger Integration execution plan."""

    target_company: str = Field(..., description="Target company name")
    acquirer_company: str = Field(..., description="Acquiring company name")
    integration_start_date: datetime = Field(..., description="Integration start date")
    integration_duration: int = Field(default=12, description="Integration duration (months)")

    # Integration Governance
    pmo_structure: dict[str, Any] = Field(default={}, description="PMO organizational structure")
    steering_committee: list[str] = Field(default=[], description="Integration steering committee")
    escalation_procedures: list[str] = Field(default=[], description="Issue escalation procedures")

    # Day 1 Readiness
    day1_plan: Day1ReadinessPlan = Field(..., description="Day 1 readiness plan")

    # Integration Workstreams
    technology_integration: IntegrationWorkstream = Field(..., description="Technology integration workstream")
    human_capital_integration: IntegrationWorkstream = Field(..., description="HR integration workstream")
    commercial_integration: IntegrationWorkstream = Field(..., description="Commercial integration workstream")
    financial_integration: IntegrationWorkstream = Field(..., description="Financial integration workstream")
    legal_compliance_integration: IntegrationWorkstream = Field(..., description="Legal/compliance workstream")

    # Synergy Realization
    synergy_plan: SynergyRealizationPlan = Field(..., description="Synergy realization plan")

    # Risk Management
    integration_risks: list[str] = Field(default=[], description="Integration risks")
    risk_mitigation_plan: dict[str, str] = Field(default={}, description="Integration risk mitigation")
    success_probability: float = Field(default=0.80, description="Integration success probability")

    # Budget and Resources
    total_integration_budget: float = Field(default=25000000, description="Total integration budget")
    budget_allocation: dict[str, float] = Field(default={}, description="Budget allocation by workstream")
    resource_requirements: dict[str, int] = Field(default={}, description="Resource requirements")

    # Success Metrics
    integration_kpis: list[str] = Field(default=[], description="Integration success KPIs")
    milestone_tracking: dict[str, str] = Field(default={}, description="Milestone tracking system")

    # Metadata
    plan_confidence: float = Field(default=0.0, description="Integration plan confidence")
    plan_creation_date: datetime = Field(default_factory=datetime.now)


class MAPMIPlanningWorkflow:
    """M&A Post-Merger Integration Planning Workflow."""

    def __init__(self):
        self.tavily_client = TavilyClient()

    @trace_node("ma_pmi_planning")
    async def execute_comprehensive_pmi_planning(
        self,
        target_company: str,
        acquirer_company: str = "Acquiring Company",
        deal_value: float | None = None,
        integration_context: dict[str, Any] = None
    ) -> PMIExecutionPlan:
        """Execute comprehensive post-merger integration planning."""

        start_time = datetime.now()
        integration_start = start_time + timedelta(days=60)  # Typical 60-day close to integration

        print(f"🤝 Starting PMI Planning for {target_company}")

        try:
            # Execute PMI planning components in parallel
            day1_task = self._develop_day1_readiness_plan(target_company, acquirer_company)
            workstreams_task = self._design_integration_workstreams(target_company, deal_value)
            synergy_task = self._develop_synergy_realization_plan(target_company, deal_value)
            governance_task = self._design_pmo_governance_structure(target_company, acquirer_company)

            # Wait for all planning components
            day1_plan, workstreams, synergy_plan, governance = await asyncio.gather(
                day1_task, workstreams_task, synergy_task, governance_task,
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(day1_plan, Exception):
                print(f"⚠️ Day 1 planning failed: {str(day1_plan)}")
                day1_plan = self._create_default_day1_plan()

            if isinstance(workstreams, Exception):
                print(f"⚠️ Workstream planning failed: {str(workstreams)}")
                workstreams = self._create_default_workstreams()

            if isinstance(synergy_plan, Exception):
                print(f"⚠️ Synergy planning failed: {str(synergy_plan)}")
                synergy_plan = self._create_default_synergy_plan()

            if isinstance(governance, Exception):
                print(f"⚠️ Governance planning failed: {str(governance)}")
                governance = self._create_default_governance()

            # Create comprehensive PMI plan
            pmi_plan = PMIExecutionPlan(
                target_company=target_company,
                acquirer_company=acquirer_company,
                integration_start_date=integration_start,
                day1_plan=day1_plan,
                technology_integration=workstreams["technology"],
                human_capital_integration=workstreams["human_capital"],
                commercial_integration=workstreams["commercial"],
                financial_integration=workstreams["financial"],
                legal_compliance_integration=workstreams["legal_compliance"],
                synergy_plan=synergy_plan,
                pmo_structure=governance
            )

            # Calculate budget allocation
            pmi_plan = self._calculate_integration_budget(pmi_plan, deal_value)

            # Define success metrics and KPIs
            pmi_plan = self._define_integration_success_metrics(pmi_plan)

            # AI-powered integration optimization
            pmi_plan = await self._optimize_integration_plan_with_ai(pmi_plan)

            execution_time = (datetime.now() - start_time).total_seconds()

            print(f"✅ PMI Planning completed in {execution_time:.1f}s")
            print(f"🎯 Integration Duration: {pmi_plan.integration_duration} months")
            print(f"💰 Total Budget: ${pmi_plan.total_integration_budget/1e6:.1f}M")
            print(f"📊 Success Probability: {pmi_plan.success_probability:.0%}")

            return pmi_plan

        except Exception as e:
            raise FinancialDataError(
                f"PMI planning failed for {target_company}: {str(e)}",
                context={"target": target_company, "deal_value": deal_value}
            )

    async def _develop_day1_readiness_plan(self, target: str, acquirer: str) -> Day1ReadinessPlan:
        """Develop comprehensive Day 1 readiness plan."""

        return Day1ReadinessPlan(
            systems_access=[
                "IT systems access and security protocols",
                "Email and communication platform integration",
                "Financial systems access for reporting",
                "Customer management system integration",
                "Employee portal and HR system access"
            ],
            employee_communications=[
                "Welcome message from acquiring company leadership",
                "Integration timeline and process overview",
                "Reporting structure and organizational changes",
                "Benefits and compensation information",
                "Q&A sessions with HR and management"
            ],
            customer_notifications=[
                "Acquisition announcement and continuity assurance",
                "Service level commitment and contact information",
                "Account management transition plan",
                "Product roadmap and enhancement commitments",
                "Customer success team expansion announcement"
            ],
            vendor_communications=[
                "Vendor notification of ownership change",
                "Contract review and novation requirements",
                "Payment and procurement process updates",
                "Supplier diversity and inclusion commitments",
                "Strategic partnership evaluation and enhancement"
            ],
            regulatory_notifications=[
                "Regulatory body notification of closing",
                "Compliance framework updates and certifications",
                "Industry-specific regulatory filing updates",
                "Data privacy and security compliance verification",
                "International regulatory jurisdiction notifications"
            ],
            success_metrics={
                "employee_notification": "100% completion by Day 1 morning",
                "system_availability": "99.9% uptime for critical systems",
                "customer_incidents": "Zero customer-impacting incidents",
                "regulatory_compliance": "100% regulatory notifications submitted",
                "communication_effectiveness": "95% stakeholder acknowledgment rate"
            },
            stakeholder_communications={
                "employees": "Town hall meeting Day 1 at 10 AM",
                "customers": "Email announcement Day 1 at 8 AM",
                "investors": "Press release Day 1 at 6 AM",
                "media": "Press conference Day 1 at 9 AM",
                "regulators": "Filing submissions Day 1 by 5 PM"
            },
            business_continuity_plan=[
                "Maintain separate operational systems during initial transition",
                "Dual reporting capabilities for financial consolidation",
                "Customer service continuity with expanded support teams",
                "Supply chain continuity with vendor relationship management",
                "Risk management continuity with enhanced monitoring"
            ]
        )

    async def _design_integration_workstreams(self, target: str, deal_value: float | None) -> dict[str, IntegrationWorkstream]:
        """Design comprehensive integration workstreams."""

        # Scale workstream complexity based on deal size
        complexity_factor = "high" if deal_value and deal_value > 2_000_000_000 else "medium"
        duration_months = 12 if complexity_factor == "high" else 9

        return {
            "technology": IntegrationWorkstream(
                workstream_name="Technology Systems Integration",
                pmo_lead="CTO Integration Team",
                duration_months=duration_months,
                phase1_milestones=[
                    "IT infrastructure assessment and integration planning",
                    "Security framework harmonization and access provisioning",
                    "Critical system integration priorities identification"
                ],
                phase2_milestones=[
                    "Core business system integration and data migration",
                    "Application portfolio consolidation and optimization",
                    "Cybersecurity framework integration and validation"
                ],
                phase3_milestones=[
                    "Full technology stack integration and optimization",
                    "Legacy system decommissioning and cost optimization",
                    "Innovation platform integration and R&D consolidation"
                ],
                success_kpis=[
                    "System uptime >99.9%",
                    "Data migration accuracy >99.5%",
                    "User adoption rate >95%",
                    "Security incident count = 0"
                ],
                team_size=15,
                budget_allocation=8000000,  # $8M for technology integration
                external_support=True,
                integration_complexity=complexity_factor
            ),

            "human_capital": IntegrationWorkstream(
                workstream_name="Human Capital & Cultural Integration",
                pmo_lead="CHRO Integration Team",
                duration_months=8,
                phase1_milestones=[
                    "Organizational design and role mapping completion",
                    "Compensation and benefits harmonization plan",
                    "Key talent retention agreements execution"
                ],
                phase2_milestones=[
                    "Cultural integration program launch and execution",
                    "Performance management system integration",
                    "Learning and development program consolidation"
                ],
                phase3_milestones=[
                    "Full cultural integration assessment and optimization",
                    "Leadership development program integration",
                    "Employee engagement and retention optimization"
                ],
                success_kpis=[
                    "Employee retention >90%",
                    "Key talent retention >95%",
                    "Employee satisfaction >7.5/10",
                    "Cultural integration score >8/10"
                ],
                team_size=8,
                budget_allocation=3000000,  # $3M for HR integration
                external_support=True,
                integration_complexity=complexity_factor
            ),

            "commercial": IntegrationWorkstream(
                workstream_name="Commercial & Market Integration",
                pmo_lead="Chief Commercial Officer",
                duration_months=10,
                phase1_milestones=[
                    "Customer retention strategy execution and communication",
                    "Sales team integration and territory optimization",
                    "Product portfolio integration and roadmap alignment"
                ],
                phase2_milestones=[
                    "Cross-selling program launch and revenue synergy realization",
                    "Marketing integration and brand strategy execution",
                    "Channel partner integration and optimization"
                ],
                phase3_milestones=[
                    "Market expansion initiatives and geographic optimization",
                    "Customer success program integration and enhancement",
                    "Revenue synergy optimization and performance tracking"
                ],
                success_kpis=[
                    "Customer retention >95%",
                    "Revenue synergy realization >75%",
                    "Cross-selling success rate >15%",
                    "Market share maintenance/growth"
                ],
                team_size=12,
                budget_allocation=2500000,  # $2.5M for commercial integration
                integration_complexity=complexity_factor
            ),

            "financial": IntegrationWorkstream(
                workstream_name="Financial Systems & Reporting Integration",
                pmo_lead="CFO Integration Team",
                duration_months=6,
                phase1_milestones=[
                    "Financial reporting consolidation and chart of accounts harmonization",
                    "Treasury and cash management integration",
                    "Tax structure optimization and compliance integration"
                ],
                phase2_milestones=[
                    "ERP system integration and financial process standardization",
                    "Budget and planning process integration",
                    "Financial controls and audit framework harmonization"
                ],
                phase3_milestones=[
                    "Cost center optimization and synergy realization tracking",
                    "Investment and capital allocation process integration",
                    "Financial performance optimization and variance analysis"
                ],
                success_kpis=[
                    "Financial close timeline ≤5 days",
                    "Cost synergy realization >85%",
                    "Budget accuracy >95%",
                    "Audit findings = 0"
                ],
                team_size=10,
                budget_allocation=2000000,  # $2M for financial integration
                integration_complexity="medium"
            ),

            "legal_compliance": IntegrationWorkstream(
                workstream_name="Legal & Regulatory Compliance Integration",
                pmo_lead="General Counsel Integration",
                duration_months=4,
                phase1_milestones=[
                    "Legal entity consolidation planning and execution",
                    "Contract portfolio review and novation planning",
                    "Regulatory compliance framework harmonization"
                ],
                phase2_milestones=[
                    "Material contract novation and vendor consolidation",
                    "Intellectual property portfolio integration",
                    "Compliance monitoring and reporting integration"
                ],
                phase3_milestones=[
                    "Legal structure optimization and cost reduction",
                    "Risk management framework integration",
                    "Corporate governance integration and board optimization"
                ],
                success_kpis=[
                    "Contract migration >95%",
                    "Regulatory compliance 100%",
                    "Legal cost optimization >20%",
                    "IP portfolio integration 100%"
                ],
                team_size=6,
                budget_allocation=1500000,  # $1.5M for legal integration
                integration_complexity="low"
            )
        }

    async def _develop_synergy_realization_plan(self, target: str, deal_value: float | None) -> SynergyRealizationPlan:
        """Develop detailed synergy realization plan."""

        estimated_revenue_synergies = (deal_value * 0.08) if deal_value else 100_000_000
        estimated_cost_synergies = (deal_value * 0.05) if deal_value else 75_000_000

        return SynergyRealizationPlan(
            revenue_synergies_timeline={
                "month_3": estimated_revenue_synergies * 0.10,
                "month_6": estimated_revenue_synergies * 0.30,
                "month_12": estimated_revenue_synergies * 0.60,
                "month_18": estimated_revenue_synergies * 0.85,
                "month_24": estimated_revenue_synergies * 1.00
            },
            cost_synergies_timeline={
                "month_1": estimated_cost_synergies * 0.15,
                "month_3": estimated_cost_synergies * 0.35,
                "month_6": estimated_cost_synergies * 0.65,
                "month_12": estimated_cost_synergies * 0.85,
                "month_18": estimated_cost_synergies * 1.00
            },
            cross_selling_plan=[
                "Customer portfolio mapping and opportunity identification",
                "Sales team cross-training on combined product portfolio",
                "Joint customer success programs and account management",
                "Integrated marketing campaigns and lead generation"
            ],
            overhead_elimination=[
                "Duplicate function elimination and role consolidation",
                "Real estate optimization and facility consolidation",
                "Vendor consolidation and procurement optimization",
                "Administrative process automation and efficiency"
            ],
            synergy_kpis=[
                "Revenue synergy realization rate (target: 75% by month 12)",
                "Cost synergy achievement rate (target: 85% by month 12)",
                "Customer retention during synergy realization (target: >95%)",
                "Employee retention during optimization (target: >90%)"
            ],
            synergy_risks=[
                "Customer churn during integration affecting revenue synergies",
                "Key talent departure impacting synergy execution",
                "Integration complexity delaying synergy realization",
                "Market conditions affecting cross-selling opportunities"
            ]
        )

    def _calculate_integration_budget(self, plan: PMIExecutionPlan, deal_value: float | None) -> PMIExecutionPlan:
        """Calculate comprehensive integration budget allocation."""

        # Budget scaling based on deal value
        if deal_value:
            if deal_value > 5_000_000_000:  # >$5B deals
                plan.total_integration_budget = 50_000_000
            elif deal_value > 2_000_000_000:  # $2-5B deals
                plan.total_integration_budget = 30_000_000
            elif deal_value > 1_000_000_000:  # $1-2B deals
                plan.total_integration_budget = 20_000_000
            else:  # <$1B deals
                plan.total_integration_budget = 15_000_000

        # Budget allocation by workstream
        plan.budget_allocation = {
            "technology_integration": plan.total_integration_budget * 0.40,  # 40%
            "human_capital": plan.total_integration_budget * 0.25,          # 25%
            "commercial_integration": plan.total_integration_budget * 0.20,  # 20%
            "financial_integration": plan.total_integration_budget * 0.10,   # 10%
            "contingency_reserve": plan.total_integration_budget * 0.05      # 5%
        }

        # Resource requirements
        plan.resource_requirements = {
            "full_time_integration_team": 25,
            "part_time_subject_matter_experts": 40,
            "external_consultants": 15,
            "executive_sponsors": 5
        }

        return plan

    def _define_integration_success_metrics(self, plan: PMIExecutionPlan) -> PMIExecutionPlan:
        """Define comprehensive integration success metrics."""

        plan.integration_kpis = [
            "Customer retention rate >95% throughout integration",
            "Employee retention rate >90% for key talent",
            "Revenue synergy realization >75% by month 12",
            "Cost synergy achievement >85% by month 18",
            "Integration milestone completion >90% on time",
            "System integration uptime >99.9%",
            "Cultural integration score >8/10 by month 6",
            "Regulatory compliance 100% maintained"
        ]

        plan.milestone_tracking = {
            "day_1_readiness": "100% completion by integration start",
            "month_1_milestones": "Critical system integration and communication",
            "month_3_milestones": "Core workstream integration and synergy initiation",
            "month_6_milestones": "Cultural integration and customer retention validation",
            "month_12_milestones": "Full operational integration and synergy optimization"
        }

        return plan

    async def _optimize_integration_plan_with_ai(self, plan: PMIExecutionPlan) -> PMIExecutionPlan:
        """AI-powered integration plan optimization."""

        provider = get_layer_provider(AnalysisLayer.OBSERVER)
        if not provider:
            plan.plan_confidence = 0.75
            return plan

        messages = [
            AIMessage(
                role="system",
                content="""You are a senior M&A integration expert optimizing post-merger integration plans.
                Analyze integration complexity, identify potential risks, and recommend optimization strategies.
                Focus on critical success factors and realistic timeline expectations."""
            ),
            AIMessage(
                role="user",
                content=f"""Optimize post-merger integration plan for {plan.target_company}:

INTEGRATION SCOPE:
- Duration: {plan.integration_duration} months
- Budget: ${plan.total_integration_budget/1e6:.1f}M
- Workstreams: 5 parallel integration tracks
- Team Size: {plan.resource_requirements.get('full_time_integration_team', 25)} FTE

INTEGRATION WORKSTREAMS:
- Technology: {plan.technology_integration.duration_months} months
- Human Capital: {plan.human_capital_integration.duration_months} months
- Commercial: {plan.commercial_integration.duration_months} months
- Financial: {plan.financial_integration.duration_months} months

Provide optimization recommendations:
1. Integration Success Probability (0.0-1.0)
2. Critical Risk Factors for integration failure
3. Timeline Optimization recommendations
4. Resource Optimization suggestions
5. Success Factor Prioritization"""
            )
        ]

        try:
            response = await provider.generate_response_async(messages, max_tokens=1200, temperature=0.1)

            # Parse success probability
            import re
            prob_match = re.search(r"success probability:?\s*([0-9.]+)", response.content.lower())
            if prob_match:
                plan.success_probability = min(max(float(prob_match.group(1)), 0.0), 1.0)
            else:
                plan.success_probability = 0.80  # Default conservative estimate

            plan.plan_confidence = 0.85

        except Exception as e:
            print(f"⚠️ AI integration optimization failed: {str(e)}")
            plan.success_probability = 0.75
            plan.plan_confidence = 0.70

        return plan

    # Helper methods for default planning components
    def _create_default_day1_plan(self) -> Day1ReadinessPlan:
        """Create default Day 1 plan when detailed planning fails."""
        return Day1ReadinessPlan(
            systems_access=["Basic IT access", "Communication systems"],
            employee_communications=["Welcome message", "Integration overview"],
            success_metrics={"basic_readiness": "Essential systems operational"}
        )

    def _create_default_workstreams(self) -> dict[str, IntegrationWorkstream]:
        """Create default workstreams when detailed planning fails."""
        return {
            "technology": IntegrationWorkstream(
                workstream_name="Technology Integration",
                pmo_lead="IT Integration Team",
                duration_months=12,
                success_kpis=["System integration completion"],
                budget_allocation=5000000
            ),
            "human_capital": IntegrationWorkstream(
                workstream_name="Human Capital Integration",
                pmo_lead="HR Integration Team",
                duration_months=6,
                success_kpis=["Employee retention >90%"],
                budget_allocation=2000000
            ),
            "commercial": IntegrationWorkstream(
                workstream_name="Commercial Integration",
                pmo_lead="Sales Integration Team",
                duration_months=8,
                success_kpis=["Customer retention >95%"],
                budget_allocation=2000000
            ),
            "financial": IntegrationWorkstream(
                workstream_name="Financial Integration",
                pmo_lead="Finance Integration Team",
                duration_months=4,
                success_kpis=["Financial close <5 days"],
                budget_allocation=1500000
            ),
            "legal_compliance": IntegrationWorkstream(
                workstream_name="Legal Integration",
                pmo_lead="Legal Integration Team",
                duration_months=3,
                success_kpis=["Compliance 100%"],
                budget_allocation=1000000
            )
        }

    def _create_default_synergy_plan(self) -> SynergyRealizationPlan:
        """Create default synergy plan when detailed planning fails."""
        return SynergyRealizationPlan(
            synergy_kpis=["Revenue synergy realization", "Cost synergy achievement"],
            tracking_frequency="monthly"
        )

    def _create_default_governance(self) -> dict[str, Any]:
        """Create default governance structure when detailed planning fails."""
        return {
            "pmo_director": "Senior Integration Executive",
            "steering_committee": ["CEO", "Division Heads", "PMO Director"],
            "meeting_frequency": "Weekly for first 3 months, bi-weekly thereafter"
        }


# Convenience functions
async def run_pmi_planning(
    target_company: str,
    acquirer_company: str = "Acquiring Company",
    deal_value: float | None = None
) -> PMIExecutionPlan:
    """Run comprehensive post-merger integration planning."""

    workflow = MAPMIPlanningWorkflow()
    return await workflow.execute_comprehensive_pmi_planning(
        target_company, acquirer_company, deal_value
    )


async def run_day1_planning(target_company: str, acquirer_company: str) -> Day1ReadinessPlan:
    """Run focused Day 1 readiness planning."""

    workflow = MAPMIPlanningWorkflow()
    return await workflow._develop_day1_readiness_plan(target_company, acquirer_company)
