/* Ariadne Console — API Client (local mode) */

const AriadneAPI = {
  baseUrl: '',
  apiKey: '',

  init(url, key) {
    this.baseUrl = url.replace(/\/+$/, '');
    this.apiKey = key || '';
  },

  headers() {
    const h = { 'Content-Type': 'application/json' };
    if (this.apiKey) h['Authorization'] = `Bearer ${this.apiKey}`;
    return h;
  },

  async request(method, path, body = null) {
    const opts = { method, headers: this.headers() };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(`${this.baseUrl}${path}`, opts);
    if (res.status === 429) throw new Error('Rate limited — try again in 60s');
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },

  // Health — hits our /api/stats endpoint
  async health() { return this.request('GET', '/api/stats'); },
  async ready() { return this.request('GET', '/api/stats'); },

  // Memories
  async store(content, opts = {}) {
    return this.request('POST', '/api/memories', {
      content,
      type: opts.topic || 'semantic',
      importance: (opts.importance || 5) / 10,
      entities: opts.entities || [],
      metadata: opts.metadata || {},
    });
  },
  async getMemory(id) { return this.request('GET', `/api/memories/${id}`); },
  async updateMemory(id, data) { return this.request('PUT', `/api/memories/${id}`, data); },
  async deleteMemory(id) { return this.request('DELETE', `/api/memories/${id}`); },

  // Search
  async search(query, opts = {}) {
    const params = new URLSearchParams({ q: query, k: String(opts.limit || 10) });
    if (opts.memory_type) params.set('type', opts.memory_type);
    return this.request('GET', `/api/search?${params}`);
  },

  // Stats
  async stats() { return this.request('GET', '/api/stats'); },
  async metrics() { return this.request('GET', '/api/stats'); },

  // Lifecycle — map to our endpoints
  async lifecycle() { return this.request('GET', '/api/stats'); },
  async lifecycleRun() { return this.request('POST', '/api/maintenance'); },

  // Graph
  async graphEntities() {
    const data = await this.request('GET', '/api/graph/all');
    return { entities: data.nodes || [] };
  },
  async graphEntity(name) {
    return this.request('GET', `/api/graph?entity=${encodeURIComponent(name)}`);
  },
  async graphEdges() {
    const data = await this.request('GET', '/api/graph/all');
    return { edges: data.edges || [] };
  },
  async graphConnect(source, target, relation, weight) {
    return this.request('POST', '/api/memories', {
      content: `${source} ${relation} ${target}`,
      type: 'semantic',
      importance: 0.5,
      entities: [source, target],
    });
  },

  // Consolidation
  async consolidate() { return this.request('POST', '/api/consolidate'); },

  // Import/Export
  async exportData() { return this.request('GET', '/api/stats'); },
  async importData(data) { return this.request('POST', '/api/memories', data); },

  // Keys (admin) — not applicable in local mode
  async createKey() { return { key: 'local-mode-no-keys' }; },
  async listKeys() { return { keys: [] }; },
  async revokeKey() { return true; },
  async rotateKey() { return true; },
};
