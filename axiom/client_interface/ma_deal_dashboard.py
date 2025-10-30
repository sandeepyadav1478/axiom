"""
M&A Deal Dashboard - Client-Facing Interface

Professional dashboard for M&A deals showing:
- Deal overview and timeline
- Target screening results (from ML Target Screener)
- Valuation analysis (DCF, Comps, Precedents)
- Synergy breakdown
- Risk assessment
- Success probability (from MA Success Predictor)
- Key recommendations

This is what investment banking clients SEE.
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class MADealDashboard:
    """
    Professional M&A deal dashboard for clients
    
    Shows comprehensive deal analysis in executive-friendly format.
    """
    
    def __init__(self, deal_data: Dict):
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required")
        
        self.deal = deal_data
    
    def create_executive_summary(self) -> go.Figure:
        """
        Create executive summary dashboard
        
        Top-level view for C-suite/Board presentations.
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Deal Overview',
                'Valuation Range',
                'Synergy Breakdown',
                'Success Probability',
                'Key Risks',
                'Recommendations'
            ),
            specs=[
                [{'type': 'table'}, {'type': 'bar'}],
                [{'type': 'pie'}, {'type': 'indicator'}],
                [{'type': 'bar'}, {'type': 'table'}]
            ],
            row_heights=[0.25, 0.35, 0.40]
        )
        
        # 1. Deal Overview (executive info)
        overview = self.deal.get('overview', {})
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[
                    ['Target', 'Deal Value', 'Structure', 'Timeline'],
                    [overview.get('target', 'Target Inc'),
                     f"${overview.get('value', 2.8)}B",
                     f"{overview.get('cash_pct', 65)}% Cash / {overview.get('stock_pct', 35)}% Stock",
                     overview.get('timeline', '6-9 months')]
                ])
            ),
            row=1, col=1
        )
        
        # 2. Valuation Range
        valuations = self.deal.get('valuations', {})
        fig.add_trace(
            go.Bar(
                x=['DCF', 'Comps', 'Precedents', 'Recommended'],
                y=[
                    valuations.get('dcf', 2.5),
                    valuations.get('comps', 2.6),
                    valuations.get('precedents', 2.7),
                    valuations.get('recommended', 2.8)
                ],
                marker_color=['lightblue', 'lightgreen', 'lightyellow', 'orange'],
                text=[f"${v}B" for v in [2.5, 2.6, 2.7, 2.8]],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Synergy Breakdown
        synergies = self.deal.get('synergies', {})
        fig.add_trace(
            go.Pie(
                labels=['Revenue Synergies', 'Cost Synergies', 'Financial Synergies'],
                values=[
                    synergies.get('revenue', 180),
                    synergies.get('cost', 120),
                    synergies.get('financial', 50)
                ],
                marker_colors=['#90EE90', '#87CEEB', '#FFD700'],
                textinfo='label+value',
                texttemplate='%{label}<br>$%{value}M'
            ),
            row=2, col=1
        )
        
        # 4. Success Probability (from ML model)
        success_prob = self.deal.get('success_probability', 0.75)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=success_prob * 100,
                title={'text': "Deal Success Probability<br>(ML Predicted)"},
                delta={'reference': 70, 'increasing': {'color': 'green'}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': 'darkblue'},
                    'steps': [
                        {'range': [0, 50], 'color': 'lightgray'},
                        {'range': [50, 70], 'color': 'yellow'},
                        {'range': [70, 100], 'color': 'lightgreen'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 65
                    }
                }
            ),
            row=2, col=2
        )
        
        # 5. Key Risks
        risks = self.deal.get('risks', [])
        risk_names = [r['name'] for r in risks[:5]]
        risk_scores = [r['score'] for r in risks[:5]]
        
        fig.add_trace(
            go.Bar(
                y=risk_names,
                x=risk_scores,
                orientation='h',
                marker_color=['red' if s > 7 else 'orange' if s > 5 else 'yellow' for s in risk_scores]
            ),
            row=3, col=1
        )
        
        # 6. Recommendations
        recommendations = self.deal.get('recommendations', [
            ['Proceed with enhanced DD', 'High Priority'],
            ['Secure financing commitments', 'Critical'],
            ['Talent retention plan required', 'Medium Priority']
        ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Action Item', 'Priority']),
                cells=dict(values=list(zip(*recommendations)))
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title_text=f"M&A Deal Analysis: {self.deal.get('target_name', 'Target Company')}",
            showlegend=False,
            height=1000
        )
        
        return fig
    
    def export_to_pdf(self, filename: str = 'ma_deal_report.pdf'):
        """Export dashboard to PDF for client presentation"""
        
        fig = self.create_executive_summary()
        
        # Would use kaleido for PDF export
        # fig.write_image(filename)
        
        # For now, HTML
        fig.write_html(filename.replace('.pdf', '.html'))
        
        return filename.replace('.pdf', '.html')


# Example
if __name__ == "__main__":
    print("M&A Deal Dashboard - Client Interface")
    print("=" * 60)
    
    if PLOTLY_AVAILABLE:
        sample_deal = {
            'target_name': 'DataRobot Inc',
            'overview': {
                'target': 'DataRobot Inc',
                'value': 2.8,
                'cash_pct': 65,
                'stock_pct': 35,
                'timeline': '6-9 months'
            },
            'valuations': {
                'dcf': 2.5,
                'comps': 2.6,
                'precedents': 2.7,
                'recommended': 2.8
            },
            'synergies': {
                'revenue': 180,
                'cost': 120,
                'financial': 50
            },
            'success_probability': 0.75,
            'risks': [
                {'name': 'Integration Risk', 'score': 7},
                {'name': 'Customer Retention', 'score': 6},
                {'name': 'Regulatory', 'score': 4},
                {'name': 'Cultural Fit', 'score': 5},
                {'name': 'Talent Retention', 'score': 6}
            ],
            'recommendations': [
                ['Proceed with enhanced DD', 'High'],
                ['Secure financing', 'Critical'],
                ['Talent retention plan', 'Medium']
            ]
        }
        
        dashboard = MADealDashboard(sample_deal)
        filename = dashboard.export_to_pdf('ma_deal_analysis.pdf')
        
        print(f"✓ Executive dashboard created: {filename}")
        print("\nThis is what clients see:")
        print("  • Professional visualizations")
        print("  • ML-powered insights")
        print("  • Actionable recommendations")
        print("  • Ready for board presentation")