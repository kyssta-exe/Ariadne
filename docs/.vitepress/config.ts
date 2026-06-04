import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/',
  title: 'Ariadne',
  description: 'Next-generation AI memory system — thread through the labyrinth of memories.',
  head: [],
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

  markdown: {
    lineNumbers: true,
  },
})
