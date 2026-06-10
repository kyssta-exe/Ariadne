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
    composition: [],
    linkTypes: [],
    activityTimeline: [],
    recentMemories: [],
    timeRange: '7d',

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
    graphFocusMode: false,
    graphFocusId: null,
    graphFocusDepth: 2,

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

    get pageTitle() {
      const titles = {
        dashboard: 'Dashboard',
        memories: 'Memories',
        graph: 'Knowledge Graph',
        search: 'Search',
        lifecycle: 'Lifecycle',
        settings: 'Settings',
      };
      return titles[this.page] || 'Dashboard';
    },

    // Init
    async init() {
      // Local mode: auto-authenticate, connect to self
      const selfUrl = window.location.origin;
      this.serverUrl = selfUrl;
      AriadneAPI.init(selfUrl, '');
      this.authenticated = true;
      await this.loadDashboard();
      this.$watch('page', (p) => this.onPageChange(p));
      setTimeout(() => { if (window.lucide) lucide.createIcons(); }, 100);
    },

    formatNumber(n) {
      if (!n) return '0';
      if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
      if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
      return String(n);
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
      await AriadneAPI.health();
      this.authenticated = true;
      this.healthStatus = 'ok';
      await this.loadDashboard();
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
        const [stats, comp, links, activity, recent] = await Promise.allSettled([
          AriadneAPI.stats(),
          AriadneAPI.request('GET', '/api/composition'),
          AriadneAPI.request('GET', '/api/link-types'),
          AriadneAPI.request('GET', `/api/activity?range=${this.timeRange}`),
          AriadneAPI.request('GET', '/api/recent?limit=8'),
        ]);

        if (stats.status === 'fulfilled') {
          const s = stats.value;
          this.stats = {
            memories: s.total_memories || s.active_memories || 0,
            entities: s.total_entities || 0,
            edges: s.total_edges || 0,
            avg_latency: s.avg_latency || 0,
            lifecycle: s.lifecycle || {},
            memory_types: s.by_type || {},
            faiss_vectors: s.faiss_vectors || 0,
            avg_importance: s.avg_importance || 0,
            db_size: s.db_size_bytes || 0,
            dedup_index: s.dedup_index_size || 0,
          };
          if (this.stats.lifecycle) this.lifecycle = this.stats.lifecycle;
        }

        if (comp.status === 'fulfilled') this.composition = comp.value.types || [];
        if (links.status === 'fulfilled') this.linkTypes = links.value.types || [];
        if (activity.status === 'fulfilled') this.activityTimeline = activity.value.timeline || [];
        if (recent.status === 'fulfilled') this.recentMemories = recent.value.memories || [];

        // Build activity feed from recent memories
        this.activity = this.recentMemories.map((m) => ({
          id: m.id,
          text: (m.content || '').substring(0, 80) + (m.content?.length > 80 ? '...' : ''),
          time: m.created_at ? this.timeAgo(m.created_at) : '',
          type: m.memory_type || 'semantic',
        }));

        this.renderCharts();
      } catch (e) {
        console.error('Dashboard load failed:', e);
      }
    },
    async loadActivity() {
      try {
        const res = await AriadneAPI.request('GET', `/api/activity?range=${this.timeRange}`);
        this.activityTimeline = res.timeline || [];
        this.renderCharts();
      } catch { this.activityTimeline = []; }
    },
    setTimeRange(range) {
      this.timeRange = range;
      this.loadActivity();
    },

    timeAgo(ts) {
      const diff = Date.now() / 1000 - ts;
      if (diff < 60) return 'just now';
      if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
      if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
      return Math.floor(diff / 86400) + 'd ago';
    },

    renderCharts() {
      this.$nextTick(() => {
        if (this.page === 'dashboard') {
          AriadneCharts.renderLifecycle('lifecycleChart', this.stats);
          AriadneCharts.renderTypeBars('compositionChart', this.composition);
          this.renderLinkTypesList();
          AriadneCharts.renderActivityTimeline('activityTimelineChart', this.activityTimeline);
        }
      });
    },

    renderLinkTypesList() {
      const el = document.getElementById('linkTypesList');
      if (!el || !this.linkTypes || this.linkTypes.length === 0) return;
      const palette = ['#a78bfa','#3b82f6','#f59e0b','#22c55e','#ec4899','#6366f1','#ef4444','#14b8a6','#f97316','#8b5cf6'];
      const total = this.linkTypes.reduce((s, i) => s + i.count, 0) || 1;
      const sorted = [...this.linkTypes].sort((a, b) => b.count - a.count);
      const showAll = el.dataset.showAll === 'true';
      const visible = showAll ? sorted : sorted.slice(0, 10);
      const remaining = sorted.length - 10;
      const app = this;
      el.innerHTML = visible.map((item, i) => {
        const pct = ((item.count / total) * 100).toFixed(0);
        return `<div class="link-type-row">
          <span class="link-type-dot" style="background:${palette[i % palette.length]}"></span>
          <span class="link-type-name">${item.type}</span>
          <span class="link-type-count">${item.count}</span>
          <span class="link-type-bar-wrap"><span class="link-type-bar" style="width:${pct}%"></span></span>
          <span class="link-type-pct">${pct}%</span>
        </div>`;
      }).join('');
      if (!showAll && remaining > 0) {
        const more = document.createElement('button');
        more.className = 'link-type-more';
        more.textContent = `+ ${remaining} more`;
        more.onclick = () => {
          const el2 = document.getElementById('linkTypesList');
          if (el2) el2.dataset.showAll = 'true';
          app.renderLinkTypesList();
        };
        el.appendChild(more);
      }
    },

    // Memories
    async searchMemories() {
      this.loading = true;
      try {
        const q = this.memorySearch || 'the';
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
        // Focus mode: load neighbors instead of full graph
        if (this.graphFocusMode && this.graphFocusId) {
          const resp = await AriadneAPI.request(
            'GET',
            '/api/graph/neighbors?memory_id=' + this.graphFocusId + '&depth=' + this.graphFocusDepth
          );
          if (resp && resp.nodes) {
            this.graphNodeCount = resp.nodes.length;
            this.graphEdgeCount = (resp.edges || []).length;
            AriadneGraph.render('graphContainer', resp.nodes, resp.edges || [], this.graphPhysics);
          }
          return;
        }

        // Fetch both entity graph and memory graph
        const [entityData, memData] = await Promise.allSettled([
          AriadneAPI.request('GET', '/api/graph/all'),
          AriadneAPI.request('GET', '/api/memory-graph'),
        ]);

        // Use memory graph as primary, entity graph as fallback
        const memGraph = memData.status === 'fulfilled' ? memData.value : null;
        const entityGraph = entityData.status === 'fulfilled' ? entityData.value : null;

        if (memGraph && memGraph.nodes && memGraph.nodes.length > 0) {
          this.graphNodeCount = memGraph.nodes.length;
          this.graphEdgeCount = (memGraph.edges || []).length;
          AriadneGraph.render('graphContainer', memGraph.nodes, memGraph.edges || [], this.graphPhysics, {
            onNodeClick: (nodeId) => this.focusGraph(nodeId),
          });
        } else if (entityGraph) {
          const nodes = entityGraph.nodes || [];
          const edgeList = entityGraph.edges || [];
          this.graphNodeCount = nodes.length;
          this.graphEdgeCount = edgeList.length;
          AriadneGraph.render('graphContainer', nodes, edgeList, this.graphPhysics, {
            onNodeClick: (nodeId) => this.focusGraph(nodeId),
          });
        }
      } catch (e) {
        console.error('Graph load failed:', e);
      } finally {
        this.loading = false;
        setTimeout(() => { if (window.lucide) lucide.createIcons(); }, 50);
      }
    },

    async focusGraph(nodeId) {
      if (!nodeId) return;
      this.graphFocusMode = true;
      this.graphFocusId = nodeId;
      await this.loadGraph();
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

    async backupDatabase() {
      this.loading = true;
      try {
        const result = await AriadneAPI.backup();
        alert('Backup saved: ' + result.filename);
      } catch (e) {
        alert('Backup failed: ' + e.message);
      } finally {
        this.loading = false;
      }
    },

    async restoreDatabase(event) {
      const file = event.target.files[0];
      if (!file) return;
      if (!confirm('Restore database from ' + file.name + '? This will overwrite current data.')) {
        event.target.value = '';
        return;
      }
      this.loading = true;
      try {
        await AriadneAPI.restore(file);
        alert('Database restored successfully.');
        await this.loadDashboard();
      } catch (e) {
        alert('Restore failed: ' + e.message);
      } finally {
        this.loading = false;
        event.target.value = '';
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
