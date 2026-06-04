/* Ariadne Console — Main Alpine.js App */

function app() {
  return {
    // Auth
    authenticated: false,
    serverUrl: localStorage.getItem('ariadne_url') || '',
    apiKey: localStorage.getItem('ariadne_key') || '',
    authError: '',
    loading: false,

    // Navigation
    page: 'dashboard',
    sidebarOpen: true,

    // Health
    healthStatus: 'ok',

    // Dashboard
    stats: { memories: 0, entities: 0, edges: 0, avg_latency: 0 },
    activity: [],

    // Memories
    memories: [],
    memorySearch: '',
    memoryFilter: { type: '', tier: '' },
    selectedMemory: null,
    showNewMemory: false,
    newMemory: { content: '', topic: 'general', importance: 5 },

    // Search
    searchQuery: '',
    searchMode: 'hybrid',
    searchResults: [],
    searchSearched: false,

    // Graph
    graphPhysics: true,
    graphNodeCount: 0,
    graphEdgeCount: 0,

    // Lifecycle
    lifecycle: { hot: 0, warm: 0, cold: 0 },
    prunePreview: [],

    // Settings
    apiKeys: [],
    showNewKey: false,
    newKeyForm: { agent_name: '', tenant_id: 'default', scopes: ['read', 'write'] },
    newKeyResult: null,

    // Quick search
    quickSearch: '',

    // Init
    async init() {
      if (this.serverUrl) {
        AriadneAPI.init(this.serverUrl, this.apiKey);
        await this.verifyAuth();
      }
      this.$watch('page', (p) => this.onPageChange(p));
      // Init icons after DOM ready
      setTimeout(() => { if (window.lucide) lucide.createIcons(); }, 100);
    },

    async login() {
      this.loading = true;
      this.authError = '';
      AriadneAPI.init(this.serverUrl, this.apiKey);
      try {
        await this.verifyAuth();
        localStorage.setItem('ariadne_url', this.serverUrl);
        localStorage.setItem('ariadne_key', this.apiKey);
      } catch (e) {
        this.authError = e.message;
        this.authenticated = false;
      } finally {
        this.loading = false;
      }
    },

    async verifyAuth() {
      try {
        await AriadneAPI.health();
        this.authenticated = true;
        this.healthStatus = 'ok';
        await this.loadDashboard();
      } catch (e) {
        throw e;
      }
    },

    logout() {
      this.authenticated = false;
      localStorage.removeItem('ariadne_url');
      localStorage.removeItem('ariadne_key');
      this.serverUrl = '';
      this.apiKey = '';
    },

    async refresh() {
      this.loading = true;
      try {
        await this.loadDashboard();
        if (this.page === 'memories') await this.searchMemories();
        if (this.page === 'graph') await this.loadGraph();
        if (this.page === 'lifecycle') await this.loadLifecycle();
        if (this.page === 'settings') await this.loadKeys();
      } finally {
        this.loading = false;
      }
    },

    // Dashboard
    async loadDashboard() {
      try {
        const [health, stats, metrics] = await Promise.allSettled([
          AriadneAPI.health(),
          AriadneAPI.stats(),
          AriadneAPI.metrics(),
        ]);
        if (health.status === 'fulfilled') this.healthStatus = 'ok';
        else this.healthStatus = 'error';

        if (stats.status === 'fulfilled') {
          const s = stats.value;
          this.stats = {
            memories: s.total_memories || s.memories || 0,
            entities: s.total_entities || s.entities || 0,
            edges: s.total_edges || s.edges || 0,
            avg_latency: s.avg_latency || s.search_avg_latency || 0,
          };
        }
        if (metrics.status === 'fulfilled') {
          const m = metrics.value;
          if (m.lifecycle) this.lifecycle = m.lifecycle;
        }

        // Build activity from recent memories
        await this.loadActivity();
        this.renderCharts();
      } catch (e) {
        console.error('Dashboard load failed:', e);
      }
    },

    async loadActivity() {
      try {
        const res = await AriadneAPI.search('*', { limit: 5 });
        const items = res.results || res.memories || res || [];
        this.activity = (Array.isArray(items) ? items.slice(0, 8) : []).map((m, i) => ({
          id: m.id || i,
          icon: '🧠',
          text: (m.content || '').substring(0, 60) + (m.content?.length > 60 ? '...' : ''),
          time: m.created_at ? this.timeAgo(m.created_at) : '',
        }));
      } catch { this.activity = []; }
    },

    timeAgo(ts) {
      const diff = Date.now() / 1000 - ts;
      if (diff < 60) return 'just now';
      if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
      if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
      return Math.floor(diff / 86400) + 'd ago';
    },

    renderCharts() {
      setTimeout(() => {
        if (this.page === 'dashboard') AriadneCharts.renderLifecycle('lifecycleChart', this.stats);
        if (this.page === 'dashboard') AriadneCharts.renderTypes('typesChart', this.stats);
      }, 100);
    },

    // Memories
    async searchMemories() {
      this.loading = true;
      try {
        const q = this.memorySearch || '*';
        const opts = {};
        if (this.memoryFilter.type) opts.memory_type = this.memoryFilter.type;
        const res = await AriadneAPI.search(q, opts);
        this.memories = res.results || res.memories || res || [];
        if (Array.isArray(this.memories) && this.memories.length > 0 && this.memories[0].memory) {
          this.memories = this.memories.map(r => ({ ...r, ...(r.memory || {}) }));
        }
      } catch (e) {
        console.error('Search failed:', e);
        this.memories = [];
      } finally {
        this.loading = false;
      }
    },

    selectMemory(mem) {
      this.selectedMemory = mem;
    },

    async createMemory() {
      try {
        await AriadneAPI.store(this.newMemory.content, {
          topic: this.newMemory.topic,
          importance: this.newMemory.importance,
        });
        this.showNewMemory = false;
        this.newMemory = { content: '', topic: 'general', importance: 5 };
        await this.searchMemories();
      } catch (e) {
        alert('Failed: ' + e.message);
      }
    },

    async deleteMemory(id) {
      if (!confirm('Delete this memory?')) return;
      try {
        await AriadneAPI.deleteMemory(id);
        this.selectedMemory = null;
        await this.searchMemories();
      } catch (e) {
        alert('Failed: ' + e.message);
      }
    },

    // Search
    async performSearch() {
      if (!this.searchQuery) return;
      this.loading = true;
      this.searchSearched = true;
      try {
        const res = await AriadneAPI.search(this.searchQuery, {
          use_hybrid: this.searchMode === 'hybrid',
        });
        this.searchResults = res.results || res.memories || res || [];
      } catch (e) {
        console.error('Search failed:', e);
        this.searchResults = [];
      } finally {
        this.loading = false;
      }
    },

    async performQuickSearch() {
      if (!this.quickSearch) return;
      this.page = 'search';
      this.searchQuery = this.quickSearch;
      await this.$nextTick();
      await this.performSearch();
    },

    // Graph
    async loadGraph() {
      this.loading = true;
      try {
        const [entities, edges] = await Promise.allSettled([
          AriadneAPI.graphEntities(),
          AriadneAPI.graphEdges(),
        ]);
        const nodes = entities.status === 'fulfilled' ? (entities.value.entities || entities.value || []) : [];
        const edgeList = edges.status === 'fulfilled' ? (edges.value.edges || edges.value || []) : [];
        this.graphNodeCount = nodes.length;
        this.graphEdgeCount = edgeList.length;
        AriadneGraph.render('graphContainer', nodes, edgeList, this.graphPhysics);
      } catch (e) {
        console.error('Graph load failed:', e);
      } finally {
        this.loading = false;
        setTimeout(() => { if (window.lucide) lucide.createIcons(); }, 50);
      }
    },

    // Lifecycle
    async loadLifecycle() {
      try {
        const res = await AriadneAPI.lifecycle();
        this.lifecycle = res.lifecycle || res;
        setTimeout(() => {
          AriadneCharts.renderRetentionCurve('retentionChart', this.lifecycle);
        }, 100);
      } catch (e) {
        console.error('Lifecycle load failed:', e);
      }
    },

    async previewPrune() {
      try {
        const res = await AriadneAPI.request('GET', '/lifecycle/prune?dry_run=true&min_age_days=30');
        this.prunePreview = res.candidates || res.memories || [];
      } catch { this.prunePreview = []; }
    },

    async triggerLifecycle() {
      this.loading = true;
      try {
        await AriadneAPI.lifecycleRun();
        await this.loadLifecycle();
      } finally { this.loading = false; }
    },

    async triggerConsolidation() {
      this.loading = true;
      try {
        await AriadneAPI.consolidate();
        alert('Consolidation complete');
      } catch (e) {
        alert('Failed: ' + e.message);
      } finally { this.loading = false; }
    },

    // Settings
    async loadKeys() {
      try {
        const res = await AriadneAPI.listKeys();
        this.apiKeys = res.keys || res || [];
      } catch (e) {
        console.error('Load keys failed:', e);
      }
    },

    async createKey() {
      try {
        const res = await AriadneAPI.createKey(
          this.newKeyForm.agent_name,
          this.newKeyForm.tenant_id,
          this.newKeyForm.scopes,
        );
        this.newKeyResult = res;
        await this.loadKeys();
      } catch (e) {
        alert('Failed: ' + e.message);
      }
    },

    async revokeKey(id) {
      if (!confirm('Revoke this key?')) return;
      try {
        await AriadneAPI.revokeKey(id);
        await this.loadKeys();
      } catch (e) {
        alert('Failed: ' + e.message);
      }
    },

    async exportData() {
      try {
        const data = await AriadneAPI.exportData();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ariadne-export-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
      } catch (e) {
        alert('Export failed: ' + e.message);
      }
    },

    // Page change handler
    onPageChange(page) {
      setTimeout(() => {
        if (window.lucide) lucide.createIcons();
        if (page === 'dashboard') this.loadDashboard();
        if (page === 'memories') this.searchMemories();
        if (page === 'graph') this.loadGraph();
        if (page === 'lifecycle') this.loadLifecycle();
        if (page === 'settings') this.loadKeys();
      }, 50);
    },
  };
}
