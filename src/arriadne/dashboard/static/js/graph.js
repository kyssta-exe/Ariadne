/* Ariadne Console — Knowledge Graph (vis-network) — Memory Graph */

const AriadneGraph = {
  network: null,

  render(containerId, nodes, edges, physics = true, callbacks = {}) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Count connections per node for sizing
    const connectionCount = {};
    edges.forEach(e => {
      const s = e.source || e.source_id || e.from;
      const t = e.target || e.target_id || e.to;
      connectionCount[s] = (connectionCount[s] || 0) + 1;
      connectionCount[t] = (connectionCount[t] || 0) + 1;
    });

    const maxConns = Math.max(1, ...Object.values(connectionCount));

    // Type colors — muted monochrome palette
    const typeColors = {
      architecture:  '#b0b0b0',
      infrastructure:'#909090',
      project:       '#a0a0a0',
      automation:    '#808080',
      preference:    '#c0c0c0',
      semantic:      '#a8a8a8',
      episodic:      '#989898',
      procedural:    '#888888',
      search:        '#787878',
      lifecycle:     '#707070',
      general:       '#a0a0a0',
    };

    const visNodes = nodes.map((n) => {
      const id = n.id;
      const rawLabel = n.label || n.content?.substring(0, 30) || `Memory ${id}`;
      const label = rawLabel.length > 28 ? rawLabel.substring(0, 26) + '...' : rawLabel;
      const memType = n.memory_type || n.entity_type || 'general';
      const conns = connectionCount[id] || 0;
      const ratio = conns / maxConns;

      // Size scales with connections — hubs are bigger
      const nodeSize = Math.max(6, Math.min(30, 6 + ratio * 24));

      // Brightness scales with connections
      const brightness = Math.round(100 + ratio * 155);
      const nodeColor = `rgb(${brightness},${brightness},${brightness})`;

      return {
        id: id,
        // Prefix with newlines to push label below the dot
        label: '\n' + label,
        title: `${rawLabel}\nType: ${memType}\nConnections: ${conns}`,
        shape: 'dot',
        color: {
          background: nodeColor,
          border: nodeColor,
          highlight: {
            background: '#ffffff',
            border: '#ffffff',
          },
          hover: {
            background: '#ffffff',
            border: '#ffffff',
          },
        },
        font: {
          color: '#b0b0b0',
          size: 11,
          face: "'Inter', system-ui, sans-serif",
          strokeWidth: 2,
          strokeColor: '#1a1a1a',
          align: 'center',
          vadjust: 8,
        },
        size: nodeSize,
        borderWidth: 0,
        shadow: {
          enabled: false,
        },
        margin: 6,
      };
    });

    // Edges — visible white/grey lines matching reference
    const visEdges = edges.map((e) => ({
      from: e.source || e.source_id || e.from,
      to: e.target || e.target_id || e.to,
      label: '',
      color: {
        color: 'rgba(200,200,200,0.6)',
        highlight: 'rgba(255,255,255,0.9)',
        hover: 'rgba(255,255,255,0.8)',
      },
      font: { enabled: false },
      arrows: { to: { enabled: false } },
      width: 0.5,
      smooth: false,
      hoverWidth: 1.5,
    }));

    // Filter edges to valid nodes
    const nodeIds = new Set(visNodes.map(n => n.id));
    const validEdges = visEdges.filter(e => nodeIds.has(e.from) && nodeIds.has(e.to));

    const data = {
      nodes: new vis.DataSet(visNodes),
      edges: new vis.DataSet(validEdges),
    };

    // Physics — force-directed with good clustering
    const options = {
      physics: {
        enabled: physics,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -100,
          centralGravity: 0.01,
          springLength: 150,
          springConstant: 0.025,
          damping: 0.45,
          theta: 0.5,
        },
        stabilization: {
          iterations: 300,
          updateInterval: 50,
        },
        maxVelocity: 50,
      },
      interaction: {
        hover: true,
        tooltipDelay: 60,
        zoomView: true,
        dragView: true,
        multiselect: false,
        navigationButtons: false,
        keyboard: false,
      },
      nodes: {
        borderWidth: 0,
        borderWidthSelected: 2,
      },
      edges: {
        smooth: false,
        shadow: false,
        hoverWidth: 2,
      },
      layout: {
        improvedLayout: true,
        hierarchical: false,
      },
    };

    // Destroy previous
    if (this.network) this.network.destroy();

    // Create network
    this.network = new vis.Network(container, data, options);

    // Click — highlight connected, dim the rest
    this.network.on('click', (params) => {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const connected = validEdges.filter(
          e => e.from === nodeId || e.to === nodeId
        );
        const connectedIds = new Set();
        connectedIds.add(nodeId);
        connected.forEach(e => {
          connectedIds.add(e.from);
          connectedIds.add(e.to);
        });

        visNodes.forEach(n => {
          const isActive = connectedIds.has(n.id);
          const c = connectionCount[n.id] || 0;
          const ratio = c / maxConns;
          const brightness = Math.round(80 + ratio * 175);
          data.nodes.update({
            id: n.id,
            opacity: isActive ? 1 : 0.1,
            color: {
              background: isActive ? '#ffffff' : `rgb(${brightness},${brightness},${brightness})`,
              border: isActive ? '#ffffff' : `rgb(${brightness},${brightness},${brightness})`,
            },
            font: { color: isActive ? '#ffffff' : '#333333' },
          });
        });

        validEdges.forEach(e => {
          const isConn = e.from === nodeId || e.to === nodeId;
          data.edges.update({
            id: e.id,
            color: { color: isConn ? 'rgba(255,255,255,0.8)' : 'rgba(255,255,255,0.02)' },
            width: isConn ? 1.5 : 0.2,
          });
        });

        // Fire callback for focus mode
        if (callbacks.onNodeClick) {
          callbacks.onNodeClick(nodeId);
        }
      }
    });

    // Double-click — reset
    this.network.on('doubleClick', () => {
      visNodes.forEach(n => {
        const c = connectionCount[n.id] || 0;
        const ratio = c / maxConns;
        const brightness = Math.round(80 + ratio * 175);
        data.nodes.update({
          id: n.id,
          opacity: 1,
          color: {
            background: `rgb(${brightness},${brightness},${brightness})`,
            border: `rgb(${brightness},${brightness},${brightness})`,
          },
          font: { color: '#d0d0d0' },
        });
      });
      validEdges.forEach(e => {
        data.edges.update({
          id: e.id,
          color: { color: 'rgba(200,200,200,0.6)' },
          width: 0.5,
        });
      });
    });

    // Freeze after stabilization
    this.network.once('stabilizationIterationsDone', () => {
      setTimeout(() => {
        this.network.setOptions({ physics: { enabled: false } });
      }, 800);
    });
  },

  destroy() {
    if (this.network) {
      this.network.destroy();
      this.network = null;
    }
  },
};
