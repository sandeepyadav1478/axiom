# 🎯 Lightweight 3D Graph Visualization Options

## Best Lightweight Solutions (All <100MB)

### Option 1: Static HTML with 3d-force-graph (SMALLEST - 0 MB!)

**Size**: Just HTML file (~10 KB)
**Setup**: Copy HTML file, open in browser
**No server, no Docker, no installation**

Create `axiom/ui/graph_viewer.html`:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Axiom 3D Knowledge Graph</title>
    <script src="https://unpkg.com/3d-force-graph"></script>
    <style>
        body { margin: 0; background: #000; }
        #3d-graph { width: 100vw; height: 100vh; }
    </style>
</head>
<body>
    <div id="3d-graph"></div>
    <script>
        // Fetch from Neo4j (via simple Python server)
        fetch('http://localhost:8889/graph-data')
            .then(r => r.json())
            .then(data => {
                ForceGraph3D()(document.getElementById('3d-graph'))
                    .graphData(data)
                    .nodeLabel('name')
                    .nodeColor(n => n.group === 'Company' ? '#4CAF50' : '#2196F3');
            });
    </script>
</body>
</html>
```

**Serve with simple Python**:
```bash
# Create tiny API server (30 lines of Python)
python axiom/ui/serve_graph.py
# Opens at localhost:8889
```

**Total size**: <1 MB (just Python script + HTML)

---

### Option 2: PyVis (Python Library - 5MB)

**Size**: 5 MB pip install
**Setup**: 2-line Python script

**Install**:
```bash
pip install pyvis
```

**Use**:
```python
from pyvis.network import Network
from neo4j import GraphDatabase

# Fetch from Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "axiom_neo4j"))
with driver.session() as session:
    result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
    
    # Create network
    net = Network(height='800px', width='100%', bgcolor='#222', font_color='white')
    net.barnes_hut()  # Physics simulation
    
    for record in result:
        n = record['n']
        m = record['m']
        r = record['r']
        
        net.add_node(n.element_id, label=dict(n).get('name', 'Node'))
        net.add_node(m.element_id, label=dict(m).get('name', 'Node'))
        net.add_edge(n.element_id, m.element_id, title=r.type)
    
    # Save HTML
    net.show('axiom/ui/graph.html')
    print("Open axiom/ui/graph.html in browser!")
```

**Result**: Interactive 3D graph in HTML file (auto-opens in browser)

---

### Option 3: Gephi (Free Desktop App)

**Size**: User downloads (not our problem)
**Setup**: 5 minutes
**Best for**: Professional visualization

**Steps**:
```bash
# 1. User downloads Gephi from https://gephi.org/

# 2. Export from Neo4j:
docker exec axiom_neo4j cypher-shell -u neo4j -p axiom_neo4j \
  "CALL apoc.export.graphml.all('/tmp/graph.graphml', {})"

# 3. Copy from container:
docker cp axiom_neo4j:/tmp/graph.graphml ./graph.graphml

# 4. Import to Gephi:
# File → Open → graph.graphml
# Layout → Force Atlas 2 → Run
# Beautiful 3D visualization!
```

**Size for us**: 0 MB (user installs Gephi themselves)

---

### Option 4: Lightweight Docker with Nginx + Static Files

**Size**: ~50 MB (nginx + Python script)

**Create lightweight server**:
```python
# axiom/ui/minimal_server.py (20 lines)
from http.server import HTTPServer, SimpleHTTPRequestHandler
from neo4j import GraphDatabase
import json

class GraphHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/data':
            # Fetch from Neo4j
            driver = GraphDatabase.driver("bolt://localhost:7687", 
                                        auth=("neo4j", "axiom_neo4j"))
            with driver.session() as session:
                result = session.run("MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100")
                
                nodes, links = [], []
                for record in result:
                    # Convert to JSON
                    # ... simple conversion logic
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'nodes': nodes, 'links': links}).encode())
        else:
            super().do_GET()

HTTPServer(('0.0.0.0', 8889), GraphHandler).serve_forever()
```

**Dockerfile** (~20 MB):
```dockerfile
FROM python:3.13-alpine
RUN pip install neo4j
COPY minimal_server.py graph.html /app/
CMD ["python", "/app/minimal_server.py"]
```

---

## 🎯 RECOMMENDED: PyVis (Best Balance)

**Why PyVis**:
- ✅ Only 5 MB
- ✅ Pure Python (pip install)
- ✅ Generates beautiful interactive 3D HTML
- ✅ No server needed (static HTML file)
- ✅ Works offline
- ✅ Very fast

**Installation**:
```bash
pip install pyvis
```

**Usage** (Create once, view anytime):
```python
# Run this script whenever you want to visualize
python axiom/ui/generate_graph_viz.py

# Opens graph.html in browser
# 3D interactive graph ready!
```

---

## ✅ Action Plan

**For 3D Visualization**:

**Option A** (Simplest - 30 seconds):
```bash
pip install pyvis
python -c "from pyvis.network import Network; import neo4j; ..." 
# Generates graph.html
open graph.html
```

**Option B** (Desktop - 5 minutes):
```bash
# Download Gephi from https://gephi.org/
# Export from Neo4j
# Import to Gephi
# Professional 3D visualization
```

**Option C** (Web-based lightweight - 2 minutes):
```bash
# Create simple Python server (20 lines)
# Serve static HTML with 3d-force-graph CDN
# Total size: <1 MB
```

**All are <100 MB. All provide 3D visualization. Pick one.**

**Which would you prefer?**