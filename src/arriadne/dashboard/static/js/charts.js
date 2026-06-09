/* Ariadne Console — Charts (Chart.js) — dark theme */

const AriadneCharts = {
  instances: {},

  destroy(id) {
    if (this.instances[id]) {
      this.instances[id].destroy();
      delete this.instances[id];
    }
  },

  // ── Palette ──────────────────────────────────────────────
  colors: {
    brand:   '#6366f1',
    red:     '#ef4444',
    amber:   '#f59e0b',
    blue:    '#3b82f6',
    indigo:  '#6366f1',
    purple:  '#a78bfa',
    green:   '#22c55e',
    pink:    '#ec4899',
  },

  typePalette: [
    '#6366f1', '#a78bfa', '#3b82f6', '#f59e0b',
    '#22c55e', '#ec4899', '#ef4444', '#14b8a6',
    '#f97316', '#8b5cf6',
  ],

  gridColor:   '#27272a',
  tickColor:   '#71717a',
  borderColor: '#27272a',

  tooltipDefaults() {
    return {
      backgroundColor: '#18181b',
      borderColor:     '#27272a',
      borderWidth:     1,
      titleColor:      '#fafafa',
      bodyColor:       '#a1a1aa',
      titleFont:       { family: "'Inter', system-ui", weight: '600', size: 13 },
      bodyFont:        { family: "'Inter', system-ui", size: 12 },
      padding:         12,
      cornerRadius:    8,
    };
  },

  // ── Memory Composition: sorted horizontal bars ───────────
  // Individual bars (not stacked), sorted descending.
  // Labels on the Y axis = type name + count — no separate legend needed.
  renderTypeBars(canvasId, items) {
    this.destroy(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas || !items || items.length === 0) return;

    // Sort descending by count
    const sorted = [...items].sort((a, b) => b.count - a.count);

    const labels = sorted.map(i =>
      `${i.type.charAt(0).toUpperCase() + i.type.slice(1)}  ${i.count}`
    );
    const data = sorted.map(i => i.count);
    const bgColors = sorted.map((_, i) => this.typePalette[i % this.typePalette.length]);

    // Build a clean per-bar tooltip
    const tooltipLines = sorted.map(i =>
      `${i.type}: ${i.count} (${i.pct}%)`
    );

    this.instances[canvasId] = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data,
          backgroundColor: bgColors.map(c => c + 'b3'),
          borderColor:     bgColors,
          borderWidth:     1,
          borderRadius:    4,
          barPercentage:   0.65,
          categoryPercentage: 0.8,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            ...this.tooltipDefaults(),
            callbacks: {
              label: (ctx) => {
                const i = ctx.dataIndex;
                return ` ${tooltipLines[i]}`;
              },
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color:    this.tickColor,
              font:     { family: "'Inter', system-ui", size: 11 },
              stepSize: 1,
            },
            grid:   { color: this.gridColor },
            border: { color: this.borderColor },
            beginAtZero: true,
          },
          y: {
            ticks: {
              color:    '#d4d4d8',
              font:     { family: "'Inter', system-ui", size: 12, weight: '500' },
              crossAlign: 'far',
            },
            grid:   { display: false },
            border: { display: false },
          },
        },
      },
    });
  },

  // ── Activity timeline ────────────────────────────────────
  renderActivityTimeline(canvasId, timeline) {
    this.destroy(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas || !timeline || timeline.length === 0) return;

    const labels = timeline.map(t => {
      const d = new Date(t.day);
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    const data = timeline.map(t => t.cnt);

    this.instances[canvasId] = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Memories',
          data,
          backgroundColor: this.colors.indigo + 'b3',
          borderColor: this.colors.indigo,
          borderWidth: 1,
          borderRadius: 4,
          barPercentage: 0.6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: this.tooltipDefaults(),
        },
        scales: {
          x: {
            ticks: {
              color:  this.tickColor,
              font:   { family: "'Inter', system-ui", size: 11 },
            },
            grid:   { display: false },
            border: { color: this.borderColor },
          },
          y: {
            ticks: {
              color:  this.tickColor,
              font:   { family: "'Inter', system-ui", size: 11 },
              stepSize: 1,
            },
            grid:   { color: this.gridColor },
            border: { color: this.borderColor },
            beginAtZero: true,
          },
        },
      },
    });
  },

  // ── Lifecycle doughnut ───────────────────────────────────
  renderLifecycle(canvasId, stats) {
    this.destroy(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const lifecycle = stats.lifecycle || {};
    const hot  = lifecycle.hot  || lifecycle.tier_1 || 0;
    const warm = lifecycle.warm || lifecycle.tier_2 || 0;
    const cold = lifecycle.cold || lifecycle.tier_3 || 0;

    this.instances[canvasId] = new Chart(canvas, {
      type: 'doughnut',
      data: {
        labels: ['Hot', 'Warm', 'Cold'],
        datasets: [{
          data:              [hot, warm, cold],
          backgroundColor:   [this.colors.red, this.colors.amber, this.colors.blue],
          borderColor:       '#18181b',
          borderWidth:       3,
          hoverBorderColor:  '#fafafa',
          hoverBorderWidth:  2,
        }],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        cutout:              '70%',
        plugins: {
          legend: { display: false },
          tooltip: {
            ...this.tooltipDefaults(),
            callbacks: {
              label: (tooltipCtx) => {
                const total = hot + warm + cold || 1;
                const pct = ((tooltipCtx.raw / total) * 100).toFixed(1);
                return ` ${tooltipCtx.label}: ${tooltipCtx.raw} (${pct}%)`;
              },
            },
          },
        },
      },
    });
  },

  // ── Retention curve line chart ───────────────────────────
  renderRetentionCurve(canvasId, lifecycle) {
    this.destroy(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const days = Array.from({ length: 30 }, (_, i) => i + 1);
    const hotCurve  = days.map(d => Math.exp(-d / 14) * 100);
    const warmCurve = days.map(d => Math.exp(-d / 30) * 100);
    const coldCurve = days.map(d => Math.exp(-d / 60) * 100);

    const self = this;
    const makeGradient = (hex, alpha) => {
      return (chart) => {
        const ctx = chart.ctx;
        const gradient = ctx.createLinearGradient(0, 0, 0, chart.height);
        gradient.addColorStop(0,   hex + alpha);
        gradient.addColorStop(1,   hex + '00');
        return gradient;
      };
    };

    this.instances[canvasId] = new Chart(canvas, {
      type: 'line',
      data: {
        labels: days.map(d => `Day ${d}`),
        datasets: [
          {
            label:                    'Hot (Tier 1)',
            data:                     hotCurve,
            borderColor:              this.colors.red,
            backgroundColor:          makeGradient(this.colors.red, '4d'),
            fill:                     true,
            tension:                  0.4,
            pointRadius:              0,
            pointHoverRadius:         5,
            pointHoverBackgroundColor: this.colors.red,
            borderWidth:              2,
          },
          {
            label:                    'Warm (Tier 2)',
            data:                     warmCurve,
            borderColor:              this.colors.amber,
            backgroundColor:          makeGradient(this.colors.amber, '33'),
            fill:                     true,
            tension:                  0.4,
            pointRadius:              0,
            pointHoverRadius:         5,
            pointHoverBackgroundColor: this.colors.amber,
            borderWidth:              2,
          },
          {
            label:                    'Cold (Tier 3)',
            data:                     coldCurve,
            borderColor:              this.colors.blue,
            backgroundColor:          makeGradient(this.colors.blue, '33'),
            fill:                     true,
            tension:                  0.4,
            pointRadius:              0,
            pointHoverRadius:         5,
            pointHoverBackgroundColor: this.colors.blue,
            borderWidth:              2,
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        interaction:         { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...this.tooltipDefaults(),
            callbacks: {
              label: (tooltipCtx) => ` ${tooltipCtx.dataset.label}: ${tooltipCtx.raw.toFixed(1)}%`,
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color:         this.tickColor,
              maxTicksLimit: 10,
              font:          { family: "'Inter', system-ui", size: 11 },
            },
            grid:   { color: this.gridColor },
            border: { color: this.borderColor },
          },
          y: {
            ticks: {
              color:    this.tickColor,
              callback: (v) => v + '%',
              font:     { family: "'Inter', system-ui", size: 11 },
            },
            grid:   { color: this.gridColor },
            border: { color: this.borderColor },
            min:    0,
            max:    100,
          },
        },
      },
    });
  },
};
