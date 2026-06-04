import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/',
  title: 'Ariadne',
  description: 'Next-generation AI memory system — sub-millisecond hybrid search, cognitive retention, and knowledge graph traversal. Zero infrastructure.',

  head: [
    // Open Graph
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'Ariadne' }],
    ['meta', { property: 'og:title', content: 'Ariadne — Memory for AI Agents' }],
    ['meta', { property: 'og:description', content: '302us vector search. Hybrid retrieval. Knowledge graph. Zero infrastructure. One pip install.' }],
    ['meta', { property: 'og:url', content: 'https://ariadne.mantes.net' }],
    ['meta', { property: 'og:image', content: 'https://ariadne.mantes.net/og-image.png' }],
    ['meta', { property: 'og:image:width', content: '1200' }],
    ['meta', { property: 'og:image:height', content: '630' }],

    // Twitter Card
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:title', content: 'Ariadne — Memory for AI Agents' }],
    ['meta', { name: 'twitter:description', content: '302us vector search. Hybrid retrieval. Knowledge graph. Zero infrastructure.' }],
    ['meta', { name: 'twitter:image', content: 'https://ariadne.mantes.net/og-image.png' }],

    // Canonical
    ['link', { rel: 'canonical', href: 'https://ariadne.mantes.net' }],

    // Geo meta
    ['meta', { name: 'robots', content: 'index, follow, max-snippet:-1, max-image-preview:large, max-video-preview:-1' }],
    ['meta', { name: 'author', content: 'Mantes' }],
  ],

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'Ariadne',
    nav: [
      { text: 'Guide', link: '/guide/' },
      { text: 'API Reference', link: '/api/' },
      { text: 'Benchmarks', link: '/benchmarks' },
      { text: 'GitHub', link: 'https://github.com/kyssta-exe/Ariadne' },
    ],

    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Introduction', link: '/guide/' },
          { text: 'Installation', link: '/guide/installation' },
          { text: 'Quick Start', link: '/guide/quick-start' },
          { text: 'Setup with Hermes', link: '/guide/hermes' },
          { text: 'Configuration', link: '/guide/configuration' },
        ],
      },
      {
        text: 'Core Concepts',
        items: [
          { text: 'Memory Types', link: '/guide/memory-types' },
          { text: 'Search & Retrieval', link: '/guide/search' },
          { text: 'Knowledge Graph', link: '/guide/graph' },
          { text: 'Memory Lifecycle', link: '/guide/lifecycle' },
          { text: 'Deduplication', link: '/guide/deduplication' },
        ],
      },
      {
        text: 'Advanced',
        items: [
          { text: 'Architecture', link: '/guide/architecture' },
          { text: 'Embeddings', link: '/guide/embeddings' },
          { text: 'Migration', link: '/guide/migration' },
          { text: 'REST API', link: '/guide/rest-api' },
          { text: 'Observability', link: '/guide/observability' },
        ],
      },
      {
        text: 'Reference',
        items: [
          { text: 'Competitive Comparison', link: '/guide/comparison' },
          { text: 'Benchmarks', link: '/guide/benchmarks' },
        ],
      },
      {
        text: 'API Reference',
        items: [
          { text: 'AriadneMemory', link: '/api/' },
          { text: 'Storage Engine', link: '/api/storage' },
          { text: 'Dedup & Contradiction', link: '/api/dedup' },
          { text: 'CLI', link: '/api/cli' },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/kyssta-exe/Ariadne' },
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Powered by <a href="https://mantes.net" target="_blank">Mantes</a>',
    },

    search: {
      provider: 'local',
    },

    editLink: {
      pattern: 'https://github.com/kyssta-exe/Ariadne/edit/main/docs/:path',
      text: 'Edit this page on GitHub',
    },
  },

  sitemap: {
    hostname: 'https://ariadne.mantes.net',
  },

  markdown: {
    lineNumbers: true,
  },
})
