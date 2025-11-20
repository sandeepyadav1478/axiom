# 🎨 Neo4j Graph Visualization Guide

## ✅ Neo4j Browser UI (Already Available)

Neo4j comes with a **built-in web UI** that's already running!

**Access**:
```
URL: http://localhost:7474
Username: neo4j
Password: axiom_neo4j
```

**Features**:
- Interactive graph visualization
- Run Cypher queries
- Explore relationships
- Export visualizations

---

## 🚀 Quick Start: View Your Knowledge Graph

### 1. Open Neo4j Browser
```bash
# In your web browser, navigate to:
http://localhost:7474
```

### 2. Connect
```
Connect URL: bolt://localhost:7687
Username: neo4j  
Password: axiom_neo4j
```

### 3. Run Queries

**View all companies**:
```cypher
MATCH (c:Company)
RETURN c
```

**View companies with relationships**:
```cypher
MATCH (c:Company)-[r]-(other)
RETURN c, r, other
LIMIT 50
```

**View full graph**:
```cypher
MATCH (n)-[r]-(m)
RETURN n, r, m
LIMIT 100
```

**Explore specific company**:
```cypher
MATCH path = (aapl:Company {symbol: 'AAPL'})-[*1..2]-(other)
RETURN path
LIMIT 50
```

---

## 🎨 Enhanced Visualization (Optional - Add if Needed)

If you want more advanced visualization beyond Neo4j Browser, here are options:

### Option 1: Streamlit Graph Viewer (Recommended)

**Why**: Simple Python UI, interactive, customizable

**Create**: `axiom/ui/graph_viewer.py`
```python
import streamlit as st
from neo4j import GraphDatabase
import pandas as pd

st.title("Axiom Knowledge Graph Viewer")

# Connect to Neo4j
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "axiom_neo4j")
)

# Sidebar
st.sidebar.header("Graph Queries")
query_type = st.sidebar.selectbox(
    "Select Query",
    ["All Companies", "Competitors Network", "Event Impact", "Correlations"]
)

# Main area
if query_type == "All Companies":
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Company)
            RETURN c.symbol as Symbol, c.name as Name, c.sector as Sector
            ORDER BY c.market_cap DESC
        """)
        df = pd.DataFrame([dict(record) for record in result])
        st.dataframe(df)

elif query_type == "Competitors Network":
    symbol = st.text_input("Enter symbol", "AAPL")
    
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Company {symbol: $symbol})-[:COMPETES_WITH]-(comp)
            RETURN comp.symbol as Competitor, comp.name as Name
        """, symbol=symbol)
        df = pd.DataFrame([dict(record) for record in result])
        st.dataframe(df)
        
# Add more query types...
```

**Run**:
```bash
streamlit run axiom/ui/graph_viewer.py
# Opens at http://localhost:8501
```

### Option 2: Grafana with Neo4j Plugin

**Why**: Professional dashboards, monitoring integration

**Deploy**:
```yaml
# Add to docker-compose
grafana:
  image: grafana/grafana:latest
  ports:
    - "3000:3000"
  environment:
    - GF_INSTALL_PLUGINS=hadesarchitect-cassandra-datasource
  volumes:
    - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
```

**Configure Neo4j datasource**:
```json
{
  "datasources": [{
    "name": "Neo4j",
    "type": "neo4j-datasource",
    "url": "bolt://neo4j:7687",
    "basicAuth": true,
    "basicAuthUser": "neo4j",
    "secureJsonData": {
      "basicAuthPassword": "axiom_neo4j"
    }
  }]
}
```

### Option 3: Custom Web UI with Neovis.js

**Why**: Full control, embeddable, JavaScript-based

**Create**: `axiom/ui/index.html`
```html
<!DOCTYPE html>
<html>
<head>
    <title>Axiom Knowledge Graph</title>
    <script src="https://unpkg.com/neovis.js@2.0.2"></script>
    <style>
        #viz {
            width: 100%;
            height: 800px;
            border: 1px solid lightgray;
        }
    </style>
</head>
<body>
    <h1>Axiom Knowledge Graph - Live View</h1>
    <div id="viz"></div>
    
    <script>
        const config = {
            containerId: "viz",
            neo4j: {
                serverUrl: "bolt://localhost:7687",
                serverUser: "neo4j",
                serverPassword: "axiom_neo4j"
            },
            visConfig: {
                nodes: {
                    shape: 'dot',
                    size: 25
                },
                edges: {
                    arrows: {
                        to: {enabled: true}
                    }
                }
            },
            labels: {
                Company: {
                    label: "name",
                    value: "market_cap",
                    community: "sector"
                }
            },
            relationships: {
                COMPETES_WITH: {
                    value: "intensity"
                },
                CORRELATED_WITH: {
                    value: "coefficient"
                }
            },
            initialCypher: "MATCH (n)-[r]-(m) RETURN n,r,m LIMIT 100"
        };
        
        const viz = new NeoVis(config);
        viz.render();
    </script>
</body>
</html>
```

**Serve**:
```bash
cd axiom/ui
python -m http.server 8888
# Open http://localhost:8888
```

---

## 🎯 Recommended: Use Built-in Neo4j Browser

**Reasons**:
1. ✅ Already available (no setup needed)
2. ✅ Professional tool from Neo4j
3. ✅ Full Cypher support
4. ✅ Interactive exploration
5. ✅ Export capabilities

**Access Now**:
```
http://localhost:7474

Connect with:
- URL: bolt://localhost:7687  
- Username: neo4j
- Password: axiom_neo4j
```

**Only add custom UI if**:
- Need embedding in your own application
- Want custom styling/branding
- Need specific visualizations not in Browser
- Want to share with non-technical users

---

## 📊 Useful Neo4j Browser Queries

### Explore the Graph

**1. See everything**:
```cypher
MATCH (n)
RETURN n
LIMIT 25
```

**2. Company network**:
```cypher
MATCH (c:Company)-[r]-(other)
RETURN c, r, other
```

**3. Find Apple's network**:
```cypher
MATCH path = (aapl:Company {symbol: 'AAPL'})-[*1..2]-(other)
RETURN path
```

**4. Competition landscape**:
```cypher
MATCH (c1:Company)-[:COMPETES_WITH {intensity: >0.7}]-(c2:Company)
RETURN c1, c2
```

**5. Correlations with explanations**:
```cypher
MATCH (s1:Stock)-[r:CORRELATED_WITH]-(s2:Stock)
WHERE r.coefficient > 0.7
RETURN s1.symbol, s2.symbol, r.coefficient, r.explanation
```

---

## 🎨 Visualization Tips

### In Neo4j Browser:

**1. Click nodes** to see properties
**2. Double-click** to expand relationships
**3. Use graph style**:
- Nodes colored by label (Company = blue, Sector = green)
- Edge thickness by relationship strength
- Node size by property (market_cap)

**4. Export**:
- PNG image
- SVG vector
- JSON data

---

## ✅ Current Status

**Neo4j Browser**: ✅ Already available at http://localhost:7474
**Custom UI**: Not needed yet (use browser first)
**Enhanced viz**: Documented above if needed later

**Access your knowledge graph now: http://localhost:7474**