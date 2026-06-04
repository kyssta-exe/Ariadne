/* Ariadne Console — Knowledge Graph Visualization (vis-network) */

const AriadneGraph = {
  network: null,

  render(containerId, entities, edges, physics = true) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Convert entities to vis-network nodes
    const nodes = entities.map((e, i) => {
      const label = e.name || e.entity_name || `Entity ${i}`;
      const type = e.entity_type || e.type || 'unknown';
      const colors = {
        person: '#58a6ff',
        place: '#3fb950',
        concept: '#bc8cff',
        tool: '#d29922',
        organization: '#f85149',
        unknown: '#8b949e',
      };
      return {
        id: e.id || e.name || i,
        label: label.length > 24 ? label.substring(0, 22) + '…' : label,
        title: `${label}\nType: ${type}\nMemories: ${e.memory_count || e.count || 0}`,
        color: {
          background: colors[type] || colors.unknown,
          border: colors[type] || colors.unknown,
          highlight: { background: '#fff', border: colors[type] || colors.unknown },
        },
        font: { color: '#e6edf3', size: 12 },
        size: Math.min(30, 10 + (e.memory_count || e.count || 0)),
        shape: type === 'person' ? 'dot' : type === 'tool' ? 'diamond' : 'dot',
      };
    });

    // Convert edges to vis-network edges
    const edgeList = edges.map((e, i) => ({
      from: e.source || e.source_id || e.from,
      to: e.target || e.target_id || e.to,
      label: e.relation || e.label || '',
      color: { color: '#30363d', highlight: '#58a6ff' },
      font: { color: '#8b949e', size: 10, strokeWidth: 0 },
      arrows: 'to',
      width: Math.max(1, (e.weight || 1) * 2),
      smooth: { type: 'continuous' },
    }));

    // Filter: only show edges where both nodes exist
    const nodeIds = new Set(nodes.map(n => n.id));
    const validEdges = edgeList.filter(e => nodeIds.has(e.from) && nodeIds.has(e.to));

    const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(validEdges) };

    const options = {
      physics: {
        enabled: physics,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -50,
          centralGravity: 0.01,
          springLength: 100,
          springConstant: 0.08,
          damping: 0.4,
        },
        stabilization: { iterations: 200 },
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        zoomView: true,
        dragView: true,
        multiselect: false,
      },
      nodes: {
        borderWidth: 2,
        shadow: { enabled: true, size: 5, color: 'rgba(0,0,0,0.3)' },
      },
      edges: {
        smooth: { type: 'continuous' },
        shadow: false,
      },
      layout: {
        improvedLayout: true,
        hierarchical: false,
      },
    };

    // Destroy previous instance
    if (this.network) {
      this.network.destroy();
    }

    this.network = new vis.Network(container, data, options);

    // Click handler
    this.network.on('click', (params) => {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const node = nodes.find(n => n.id === nodeId);
        if (node) {
          console.log('Entity clicked:', node);
        }
      }
    });

    // Stabilization complete
    this.network.once('stabilizationIterationsDone', () => {
      this.network.setOptions({ physics: { enabled: false } });
    });
  },

  destroy() {
    if (this.network) {
      this.network.destroy();
      this.network = null;
    }
  },
};
