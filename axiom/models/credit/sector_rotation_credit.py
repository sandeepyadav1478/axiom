"""
Sector Rotation Credit Model

Adjusts credit risk based on sector performance and rotation patterns.
Different sectors have different default rates in different economic cycles.
"""

import numpy as np

class SectorRotationCredit:
    """Credit model incorporating sector dynamics"""
    
    def __init__(self):
        self.sector_base_rates = {
            'Technology': 0.08,
            'Healthcare': 0.10,
            'Finance': 0.12,
            'Energy': 0.15,
            'Consumer': 0.11
        }
    
    def adjust_for_cycle(self, sector: str, economic_indicator: float) -> float:
        """Adjust default probability for economic cycle"""
        base = self.sector_base_rates.get(sector, 0.12)
        
        # Positive indicator = expansionary = lower defaults
        adjustment = -economic_indicator * 0.05
        
        return max(0.01, base + adjustment)

if __name__ == "__main__":
    model = SectorRotationCredit()
    prob = model.adjust_for_cycle('Technology', 0.3)
    print(f"Default prob: {prob:.1%}")