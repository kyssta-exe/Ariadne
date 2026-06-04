/* Ariadne Console — Charts (Chart.js) */

const AriadneCharts = {
  instances: {},

  destroy(id) {
    if (this.instances[id]) {
      this.instances[id].destroy();
      delete this.instances[id];
    }
  },

  renderLifecycle(canvasId, stats) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const lifecycle = stats.lifecycle || {};
    const hot = lifecycle.hot || lifecycle.tier_1 || 0;
    const warm = lifecycle.warm || lifecycle.tier_2 || 0;
    const cold = lifecycle.cold || lifecycle.tier_3 || 0;
    const total = hot + warm + cold || 1;

    this.instances[canvasId] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Hot', 'Warm', 'Cold'],
        datasets: [{
          data: [hot, warm, cold],
          backgroundColor: ['#f85149', '#d29922', '#58a6ff'],
          borderColor: '#161b22',
          borderWidth: 3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: { color: '#e6edf3', padding: 12, font: { size: 13 } },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const pct = ((ctx.raw / total) * 100).toFixed(1);
                return `${ctx.label}: ${ctx.raw} (${pct}%)`;
              },
            },
          },
        },
      },
    });
  },

  renderTypes(canvasId, stats) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const types = stats.memory_types || stats.types || {};
    const labels = Object.keys(types).length > 0
      ? Object.keys(types)
      : ['semantic', 'episodic', 'procedural'];
    const data = Object.keys(types).length > 0
      ? Object.values(types)
      : [stats.memories || 0, 0, 0];

    const colors = ['#58a6ff', '#bc8cff', '#3fb950', '#d29922', '#f85149'];

    this.instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Memories',
          data,
          backgroundColor: labels.map((_, i) => colors[i % colors.length]),
          borderRadius: 4,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: {
            ticks: { color: '#8b949e' },
            grid: { color: '#21262d' },
          },
          y: {
            ticks: { color: '#8b949e' },
            grid: { color: '#21262d' },
            beginAtZero: true,
          },
        },
      },
    });
  },

  renderRetentionCurve(canvasId, lifecycle) {
    this.destroy(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Ebbinghaus forgetting curve: R = e^(-t/S)
    // Where S = stability (higher = slower forgetting)
    const days = Array.from({ length: 30 }, (_, i) => i + 1);
    const hotCurve = days.map(d => Math.exp(-d / 14) * 100);
    const warmCurve = days.map(d => Math.exp(-d / 30) * 100);
    const coldCurve = days.map(d => Math.exp(-d / 60) * 100);

    this.instances[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: days.map(d => `Day ${d}`),
        datasets: [
          {
            label: 'Hot (Tier 1)',
            data: hotCurve,
            borderColor: '#f85149',
            backgroundColor: 'rgba(248,81,73,0.1)',
            fill: true,
            tension: 0.4,
          },
          {
            label: 'Warm (Tier 2)',
            data: warmCurve,
            borderColor: '#d29922',
            backgroundColor: 'rgba(210,153,34,0.1)',
            fill: true,
            tension: 0.4,
          },
          {
            label: 'Cold (Tier 3)',
            data: coldCurve,
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88,166,255,0.1)',
            fill: true,
            tension: 0.4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#e6edf3' },
          },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(1)}% retention`,
            },
          },
        },
        scales: {
          x: {
            ticks: { color: '#8b949e', maxTicksLimit: 10 },
            grid: { color: '#21262d' },
          },
          y: {
            ticks: { color: '#8b949e', callback: (v) => v + '%' },
            grid: { color: '#21262d' },
            min: 0,
            max: 100,
          },
        },
      },
    });
  },
};
