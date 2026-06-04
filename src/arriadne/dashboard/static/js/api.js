/* Ariadne Console — API Client */

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

  // Health
  async health() { return this.request('GET', '/health'); },
  async ready() { return this.request('GET', '/ready'); },

  // Memories
  async store(content, opts = {}) {
    return this.request('POST', '/memories', {
      content,
      topic: opts.topic || 'general',
      importance: opts.importance || 5,
      entities: opts.entities || [],
      metadata: opts.metadata || {},
      user_id: opts.user_id,
      agent_id: opts.agent_id,
    });
  },
  async getMemory(id) { return this.request('GET', `/memories/${id}`); },
  async updateMemory(id, data) { return this.request('PATCH', `/memories/${id}`, data); },
  async deleteMemory(id) { return this.request('DELETE', `/memories/${id}`); },

  // Search
  async search(query, opts = {}) {
    return this.request('POST', '/search', {
      query,
      limit: opts.limit || 10,
      threshold: opts.threshold || 0.3,
      use_hybrid: opts.use_hybrid !== false,
      memory_type: opts.memory_type || undefined,
      community_id: opts.community_id || undefined,
    });
  },
  async searchStream(query, opts = {}) {
    // SSE streaming search
    const headers = this.headers();
    const res = await fetch(`${this.baseUrl}/search/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ query, limit: opts.limit || 10 }),
    });
    return res;
  },

  // Stats
  async stats() { return this.request('GET', '/stats'); },
  async metrics() { return this.request('GET', '/metrics'); },

  // Lifecycle
  async lifecycle() { return this.request('GET', '/lifecycle'); },
  async lifecycleRun() { return this.request('POST', '/lifecycle/run'); },

  // Graph
  async graphEntities() { return this.request('GET', '/graph/entities'); },
  async graphEntity(name) { return this.request('GET', `/graph/entity/${encodeURIComponent(name)}`); },
  async graphEdges() { return this.request('GET', '/graph/edges'); },
  async graphConnect(source, target, relation, weight) {
    return this.request('POST', '/graph/connect', { source, target, relation, weight });
  },

  // Communities
  async communities() { return this.request('GET', '/communities'); },
  async detectCommunities() { return this.request('POST', '/communities/detect'); },

  // Temporal
  async temporalFacts(subject) {
    const q = subject ? `?subject=${encodeURIComponent(subject)}` : '';
    return this.request('GET', `/temporal/facts${q}`);
  },

  // Consolidation
  async consolidate() { return this.request('POST', '/consolidate'); },

  // Import/Export
  async exportData() { return this.request('GET', '/export'); },
  async importData(data) { return this.request('POST', '/import', data); },

  // Keys (admin)
  async createKey(agentName, tenantId, scopes, rateLimit) {
    return this.request('POST', '/auth/keys', {
      agent_name: agentName,
      tenant_id: tenantId || 'default',
      scopes: scopes || ['read', 'write'],
      rate_limit_rpm: rateLimit || 120,
    });
  },
  async listKeys() { return this.request('GET', '/auth/keys'); },
  async revokeKey(keyId) { return this.request('DELETE', `/auth/keys/${keyId}`); },
  async rotateKey(keyId) { return this.request('POST', `/auth/keys/${keyId}/rotate`); },
};
