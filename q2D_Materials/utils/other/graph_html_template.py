"""
Beautiful interactive graph HTML template with custom node shapes and property panels.

This module generates an HTML file for NetworkX graph visualization using vis.js
with custom node shapes, interactive property panels, and enhanced legend.
"""

from typing import Dict, Any, List, Tuple
import json


def generate_html_template(
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]],
    title: str = "Structure Graph",
    width: str = "100%",
    height: str = "900px"
) -> str:
    """
    Generate a complete HTML document for graph visualization.
    
    Parameters
    ----------
    nodes_data : list
        List of node dictionaries with id, label, color, size, node_type, title
    edges_data : list
        List of edge dictionaries with from, to, color, title
    title : str
        Title of the graph
    width : str
        Width of the graph container
    height : str
        Height of the graph container
        
    Returns
    -------
    str
        Complete HTML document
    """
    
    # Prepare data for JavaScript
    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
    
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            overflow: hidden;
            height: 100vh;
        }}
        
        #container {{
            display: flex;
            width: 100%;
            height: 100%;
        }}
        
        #graph-container {{
            flex: 1;
            position: relative;
            background: #0f172a;
        }}
        
        #graph-canvas {{
            width: {width};
            height: {height};
        }}
        
        #property-panel {{
            width: 320px;
            background: #1c1c24;
            border-left: 1px solid rgba(255, 255, 255, 0.1);
            overflow-y: auto;
            padding: 20px;
            display: none;
            z-index: 100;
            box-shadow: -5px 0 20px rgba(0, 0, 0, 0.3);
            animation: slideIn 0.3s ease-out;
        }}
        
        #property-panel.visible {{
            display: block;
        }}
        
        @keyframes slideIn {{
            from {{
                transform: translateX(100%);
                opacity: 0;
            }}
            to {{
                transform: translateX(0);
                opacity: 1;
            }}
        }}
        
        .property-header {{
            font-size: 14px;
            font-weight: 700;
            color: #83d3dc;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(131, 211, 220, 0.2);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .property-section {{
            margin-bottom: 15px;
        }}
        
        .property-section-title {{
            font-size: 12px;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        
        .property-item {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 8px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 6px;
            margin-bottom: 6px;
            font-size: 12px;
            word-break: break-all;
        }}
        
        .property-label {{
            color: #64748b;
            font-weight: 500;
            flex-shrink: 0;
            margin-right: 10px;
        }}
        
        .property-value {{
            color: #cbd5e1;
            font-family: 'Courier New', monospace;
            flex: 1;
            text-align: right;
        }}
        
        .property-copy {{
            background: transparent;
            border: none;
            color: #83d3dc;
            cursor: pointer;
            font-size: 10px;
            padding: 2px 6px;
            opacity: 0.7;
            transition: opacity 0.2s;
            margin-left: 8px;
        }}
        
        .property-copy:hover {{
            opacity: 1;
        }}
        
        #legend-panel {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: #1c1c24;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            z-index: 50;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            max-width: 280px;
        }}
        
        .legend-title {{
            font-size: 13px;
            font-weight: 700;
            color: #83d3dc;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .legend-section {{
            margin-bottom: 15px;
        }}
        
        .legend-section-title {{
            font-size: 11px;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 12px;
            color: #cbd5e1;
        }}
        
        .legend-sample {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            margin-right: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 9px;
            font-weight: bold;
            color: white;
        }}
        
        #top-toolbar {{
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
            z-index: 40;
        }}
        
        .toolbar-button {{
            background: #1c1c24;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #cbd5e1;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: all 0.2s;
        }}
        
        .toolbar-button:hover {{
            background: #262630;
            border-color: #83d3dc;
            color: #83d3dc;
        }}
        
        /* Scrollbar styling */
        #property-panel::-webkit-scrollbar {{
            width: 6px;
        }}
        
        #property-panel::-webkit-scrollbar-track {{
            background: transparent;
        }}
        
        #property-panel::-webkit-scrollbar-thumb {{
            background: #64748b;
            border-radius: 3px;
        }}
        
        #property-panel::-webkit-scrollbar-thumb:hover {{
            background: #83d3dc;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="graph-container">
            <canvas id="graph-canvas"></canvas>
            <div id="legend-panel">
                <div class="legend-title">Legend</div>
                
                <div class="legend-section">
                    <div class="legend-section-title">Node Types</div>
                    <div class="legend-item">
                        <div class="legend-sample" style="background: #C3D1D1;">■</div>
                        <span>Octahedron</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-sample" style="background: #49ADB6;">●</div>
                        <span>Layer</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-sample" style="background: #F5B102;">●</div>
                        <span>A-site / Molecule</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-sample" style="background: #039FE5;">▲</div>
                        <span>Spacer</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-sample" style="background: #8F26AB;">●</div>
                        <span>X-site</span>
                    </div>
                </div>
                
                <div class="legend-section">
                    <div class="legend-section-title">Edge Types</div>
                    <div class="legend-item">
                        <div style="width: 20px; height: 2px; background: #64748b; margin-right: 10px;"></div>
                        <span>Contains</span>
                    </div>
                    <div class="legend-item">
                        <div style="width: 20px; height: 2px; background: #f59e0b; margin-right: 10px;"></div>
                        <span>Bonded To</span>
                    </div>
                    <div class="legend-item">
                        <div style="width: 20px; height: 2px; background: #ef4444; border-top: 2px dashed; margin-right: 10px;"></div>
                        <span>Shares Atoms</span>
                    </div>
                </div>
            </div>
            
            <div id="top-toolbar">
                <button class="toolbar-button" onclick="toggleLegend()">Legend</button>
                <button class="toolbar-button" onclick="resetCamera()">Reset</button>
            </div>
        </div>
        
        <div id="property-panel"></div>
    </div>
    
    <script>
        // Data embedded in the HTML
        const nodesData = {nodes_json};
        const edgesData = {edges_json};
        
        // Create nodes for vis.js
        const nodes = new vis.DataSet(nodesData.map(n => ({{
            id: n.id,
            label: n.label,
            title: n.title,
            color: {{
                background: n.color,
                border: '#000000',
                highlight: {{
                    background: n.color,
                    border: '#83d3dc'
                }},
                hover: {{
                    background: n.color,
                    border: '#83d3dc'
                }}
            }},
            font: {{
                color: '#1c1c24',
                face: 'Arial',
                size: 14,
                bold: {{
                    color: '#1c1c24'
                }}
            }},
            borderWidth: 2,
            borderWidthSelected: 3,
            shadow: true,
            x: Math.random() * 1000 - 500,
            y: Math.random() * 1000 - 500,
            node_type: n.node_type,
            size: n.size,
            _fullData: n._fullData || {{}}
        }})));
        
        // Create edges for vis.js
        const edges = new vis.DataSet(edgesData.map(e => ({{
            from: e.from,
            to: e.to,
            color: {{
                color: e.color,
                highlight: e.color
            }},
            width: 2.5,
            font: {{
                size: 12,
                color: '#cbd5e1'
            }},
            title: e.title,
            smooth: {{
                type: 'continuous',
                roundness: 0.5
            }},
            shadow: true
        }})));
        
        // Create network
        const container = document.getElementById('graph-container');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            physics: {{
                enabled: true,
                barnesHut: {{
                    gravitationalConstant: -3000,
                    centralGravity: 0.15,
                    springLength: 250,
                    springConstant: 0.05,
                    damping: 0.12,
                    avoidOverlap: 1.0
                }},
                stabilization: {{
                    enabled: true,
                    iterations: 200,
                    fit: true
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true,
                navigationButtons: true,
                keyboard: true
            }},
            nodes: {{
                shape: 'dot',
                scaling: {{
                    min: 15,
                    max: 50
                }}
            }}
        }};
        
        const network = new vis.Network(container, data, options);
        let selectedNode = null;
        
        // Handle node click
        network.on('click', function(params) {{
            if (params.nodes && params.nodes.length > 0) {{
                selectedNode = params.nodes[0];
                showNodeProperties(selectedNode);
            }} else {{
                hidePropertyPanel();
            }}
        }});
        
        // Handle node hover
        network.on('hoverNode', function(params) {{
            const node = nodes.get(params.node);
            if (node.title) {{
                console.log('Hovering:', node.label);
            }}
        }});
        
        function showNodeProperties(nodeId) {{
            const node = nodes.get(nodeId);
            const panel = document.getElementById('property-panel');
            
            let html = '<div class="property-header">' + node.label + '</div>';
            
            // Node type section
            html += '<div class="property-section">';
            html += '<div class="property-section-title">Node Info</div>';
            html += '<div class="property-item">';
            html += '<span class="property-label">Type:</span>';
            html += '<span class="property-value">' + node.node_type + '</span>';
            html += '</div>';
            html += '<div class="property-item">';
            html += '<span class="property-label">ID:</span>';
            html += '<span class="property-value">' + nodeId + '</span>';
            html += '</div>';
            html += '</div>';
            
            // All other properties
            const fullData = node._fullData || {{}};
            if (Object.keys(fullData).length > 0) {{
                html += '<div class="property-section">';
                html += '<div class="property-section-title">Properties</div>';
                
                for (const [key, value] of Object.entries(fullData)) {{
                    if (key === 'node_type') continue;
                    const formatted = formatValue(value);
                    html += '<div class="property-item">';
                    html += '<span class="property-label">' + key + ':</span>';
                    html += '<span class="property-value">' + formatted + '</span>';
                    html += '</div>';
                }}
                
                html += '</div>';
            }}
            
            panel.innerHTML = html;
            panel.classList.add('visible');
        }}
        
        function hidePropertyPanel() {{
            const panel = document.getElementById('property-panel');
            panel.classList.remove('visible');
            selectedNode = null;
        }}
        
        function formatValue(value) {{
            if (Array.isArray(value)) {{
                if (value.length > 5) {{
                    return '[' + value.slice(0, 5).join(', ') + ', ... (' + value.length + ' items)]';
                }}
                return '[' + value.join(', ') + ']';
            }}
            if (typeof value === 'object' && value !== null) {{
                return JSON.stringify(value).substring(0, 50) + '...';
            }}
            if (typeof value === 'number') {{
                return value.toFixed(4);
            }}
            return String(value).substring(0, 100);
        }}
        
        function toggleLegend() {{
            const legend = document.getElementById('legend-panel');
            legend.style.display = legend.style.display === 'none' ? 'block' : 'none';
        }}
        
        function resetCamera() {{
            network.fit({{
                animation: {{
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }}
            }});
        }}
        
        // Initial fit
        setTimeout(() => {{
            network.fit();
        }}, 1000);
        
        // Close panel on ESC
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                hidePropertyPanel();
            }}
        }});
    </script>
</body>
</html>
"""
    
    return html
