"""
Supply Chain Intelligence for M&A Target Analysis

Analyzes:
- Import/export data for target companies
- Supplier/customer relationships  
- Supply chain disruption risks
- Geographic dependencies
- Vertical integration opportunities

Helps identify strategic fit and integration synergies.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class SupplyChainProfile:
    """Supply chain profile of company"""
    company_name: str
    top_suppliers: List[str]
    top_customers: List[str]
    import_countries: List[str]
    export_countries: List[str]
    supplier_concentration: float  # 0-1, higher = more concentrated
    customer_concentration: float
    geographic_diversity: float  # 0-1, higher = more diverse


class SupplyChainIntelligence:
    """Analyze supply chains for M&A opportunities"""
    
    def assess_strategic_fit(
        self,
        acquirer_supply_chain: SupplyChainProfile,
        target_supply_chain: SupplyChainProfile
    ) -> Dict:
        """Assess supply chain synergies"""
        
        # Supplier overlap (cost synergies)
        supplier_overlap = len(
            set(acquirer_supply_chain.top_suppliers) & 
            set(target_supply_chain.top_suppliers)
        )
        
        # Customer overlap (avoid cannibalization)
        customer_overlap = len(
            set(acquirer_supply_chain.top_customers) & 
            set(target_supply_chain.top_customers)
        )
        
        # Geographic complementarity
        new_markets = len(
            set(target_supply_chain.export_countries) - 
            set(acquirer_supply_chain.export_countries)
        )
        
        # Vertical integration opportunity
        vertical_integration = (
            target_supply_chain.company_name in acquirer_supply_chain.top_suppliers or
            acquirer_supply_chain.company_name in target_supply_chain.top_suppliers
        )
        
        synergy_score = 0.0
        
        # Scoring
        synergy_score += min(0.3, supplier_overlap * 0.1)  # Procurement synergies
        synergy_score += min(0.2, new_markets * 0.05)  # Market expansion
        if vertical_integration:
            synergy_score += 0.3  # Vertical integration value
        if customer_overlap < 3:
            synergy_score += 0.1  # Low cannibalization
        
        return {
            'synergy_score': synergy_score,
            'supplier_synergies': supplier_overlap > 0,
            'market_expansion': new_markets,
            'vertical_integration': vertical_integration,
            'customer_conflict': customer_overlap > 5
        }


if __name__ == "__main__":
    print("Supply Chain Intelligence for M&A")
    
    intel = SupplyChainIntelligence()
    
    acquirer = SupplyChainProfile(
        company_name='Acme Corp',
        top_suppliers=['Supplier A', 'Supplier B'],
        top_customers=['Customer 1', 'Customer 2'],
        import_countries=['China', 'Mexico'],
        export_countries=['US', 'Canada'],
        supplier_concentration=0.4,
        customer_concentration=0.3,
        geographic_diversity=0.7
    )
    
    target = SupplyChainProfile(
        company_name='Target Inc',
        top_suppliers=['Supplier B', 'Supplier C'],
        top_customers=['Customer 3', 'Customer 4'],
        import_countries=['Vietnam'],
        export_countries=['US', 'EU'],
        supplier_concentration=0.6,
        customer_concentration=0.5,
        geographic_diversity=0.6
    )
    
    assessment = intel.assess_strategic_fit(acquirer, target)
    
    print(f"Synergy score: {assessment['synergy_score']:.2f}")
    print(f"Supplier synergies: {assessment['supplier_synergies']}")
    print(f"New markets: {assessment['market_expansion']}")
    print("✓ Supply chain intelligence")