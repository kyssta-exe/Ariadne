<script setup>
import { useRoute, useData } from 'vitepress'
import { computed } from 'vue'

const route = useRoute()
const { frontmatter, site } = useData()

const structuredData = computed(() => {
  const base = 'https://ariadne.mantes.net'
  const path = route.path.replace(/\.html$/, '').replace(/\/index$/, '/')
  const url = base + path
  const schemas = []

  // SoftwareApplication schema (homepage only)
  if (path === '/' || path === '') {
    schemas.push({
      '@context': 'https://schema.org',
      '@type': 'SoftwareApplication',
      name: 'Ariadne',
      description: 'Next-generation AI memory system for agents. Sub-millisecond hybrid search, cognitive retention, knowledge graph traversal. Zero infrastructure.',
      url: base,
      applicationCategory: 'DeveloperApplication',
      operatingSystem: 'Linux, macOS, Windows',
      offers: {
        '@type': 'Offer',
        price: '0',
        priceCurrency: 'USD',
      },
      author: {
        '@type': 'Organization',
        name: 'Mantes',
        url: 'https://mantes.net',
      },
      softwareVersion: '0.1.2',
      downloadUrl: 'https://pypi.org/project/arriadne/',
      installUrl: 'https://pypi.org/project/arriadne/',
      codeRepository: 'https://github.com/kyssta-exe/Ariadne',
      programmingLanguage: 'Python',
      runtimePlatform: 'Python 3.10+',
      featureList: [
        'FAISS vector search (0.78ms)',
        'Hybrid search with Reciprocal Rank Fusion',
        'Knowledge graph with BFS traversal',
        'Ebbinghaus forgetting curve',
        'MinHash LSH deduplication',
        'SQLite + FAISS, zero infrastructure',
      ],
    })
  }

  // FAQPage schema (guide pages with FAQ sections)
  const faqData = {
    '/guide/': {
      '@type': 'FAQPage',
      mainEntity: [
        {
          '@type': 'Question',
          name: 'What is Ariadne?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: 'Ariadne is a local memory system for AI agents. It stores, searches, and manages memories using FAISS vector search, SQLite FTS5 keywords, a knowledge graph, and cognitive retention modeling. No cloud, no API keys, no daemon.',
          },
        },
        {
          '@type': 'Question',
          name: 'How fast is Ariadne?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: "Ariadne's FAISS vector search returns results in 0.78ms across 10,000 memories. Hybrid search (vector + keyword + graph) completes in 2.15ms. This is 196x faster than sqlite-vec.",
          },
        },
        {
          '@type': 'Question',
          name: 'Does Ariadne work with Hermes Agent?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: 'Yes. Ariadne ships as a drop-in plugin for Hermes Agent. Install the plugin, set memory.provider to ariadne, restart. All existing tool names and conversations work unchanged.',
          },
        },
        {
          '@type': 'Question',
          name: 'What makes Ariadne different from ChromaDB or LanceDB?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: 'Ariadne combines vector search, keyword search, a knowledge graph, and deduplication in a single library. ChromaDB and LanceDB are vector-only stores.',
          },
        },
        {
          '@type': 'Question',
          name: 'Can I migrate from Mnemosyne?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: 'Yes. Run ariadne migrate --from mnemosyne. All memories, graph edges, and metadata transfer automatically.',
          },
        },
      ],
    },
    '/guide/search': {
      '@type': 'FAQPage',
      mainEntity: [
        {
          '@type': 'Question',
          name: 'What is hybrid search?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: 'Hybrid search combines vector similarity (semantic understanding) and BM25 keyword matching (exact text match), then merges results using Reciprocal Rank Fusion. 92% recall@10.',
          },
        },
        {
          '@type': 'Question',
          name: 'When should I use vector search vs keyword search?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: "Use vector for semantic queries ('how do I deploy this'). Use keyword for exact matches ('kubectl apply'). Ariadne's hybrid search runs both automatically.",
          },
        },
        {
          '@type': 'Question',
          name: 'Does Ariadne support embeddings?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: 'Yes. Any embedding model works. Recommended: sentence-transformers with all-MiniLM-L6-v2 for 384-dim vectors. Without embeddings, Ariadne falls back to keyword-only search.',
          },
        },
        {
          '@type': 'Question',
          name: 'How fast is search at scale?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: '10K memories: 0.78ms (vector), 2.15ms (hybrid). 100K memories: 1.8ms (vector). FAISS auto-upgrades from exact to approximate search.',
          },
        },
      ],
    },
    '/guide/quick-start': {
      '@type': 'FAQPage',
      mainEntity: [
        {
          '@type': 'Question',
          name: 'How do I install Ariadne?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: 'pip install ariadne. No system packages, no Docker, no external services.',
          },
        },
        {
          '@type': 'Question',
          name: 'What Python version do I need?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: 'Python 3.10 or later.',
          },
        },
        {
          '@type': 'Question',
          name: 'Can I use Ariadne without an embedding model?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: 'Yes. Without embeddings, you get keyword-only search (FTS5), the knowledge graph, deduplication, and retention modeling.',
          },
        },
        {
          '@type': 'Question',
          name: 'How much disk space does Ariadne use?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: 'About 5MB per 1,000 memories. Both SQLite and FAISS are compact and auto-managed.',
          },
        },
      ],
    },
    '/guide/hermes': {
      '@type': 'FAQPage',
      mainEntity: [
        {
          '@type': 'Question',
          name: 'Will Ariadne break my existing Hermes setup?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: 'No. Ariadne uses the same tool names as Mnemosyne. All conversations, cron jobs, and memory references work unchanged.',
          },
        },
        {
          '@type': 'Question',
          name: 'Can I switch back to Mnemosyne?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: 'Yes. Run hermes config set memory.provider mnemosyne and restart.',
          },
        },
        {
          '@type': 'Question',
          name: 'Does Ariadne work with Hermes cron jobs?',
          acceptedAnswer: {
            '@type': 'Answer',
            text: "Yes. Cron jobs using mnemosyne_remember/mnemosyne_recall work identically.",
          },
        },
      ],
    },
  }

  // Match current path to FAQ data
  for (const [faqPath, faqSchema] of Object.entries(faqData)) {
    if (path.startsWith(faqPath) || path === faqPath) {
      schemas.push({
        '@context': 'https://schema.org',
        ...faqSchema,
      })
      break
    }
  }

  // BreadcrumbList schema (all pages)
  const parts = path.split('/').filter(Boolean)
  const breadcrumbItems = [{ name: 'Home', url: base }]

  if (parts.length > 0) {
    let currentPath = ''
    parts.forEach((part) => {
      currentPath += '/' + part
      const name = part
        .replace(/-/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase())
        .replace(/Api/g, 'API')
      breadcrumbItems.push({ name, url: base + currentPath })
    })
  }

  schemas.push({
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: breadcrumbItems.map((item, i) => ({
      '@type': 'ListItem',
      position: i + 1,
      name: item.name,
      item: item.url,
    })),
  })

  return schemas
})
</script>

<template>
  <div>
    <script
      v-for="(schema, i) in structuredData"
      :key="i"
      type="application/ld+json"
      v-html="JSON.stringify(schema)"
    />
  </div>
</template>
