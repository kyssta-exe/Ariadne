<script setup>
import { useRoute } from 'vitepress'
import { computed } from 'vue'

const route = useRoute()

const structuredData = computed(() => {
  const base = 'https://ariadne.mantes.net'
  const path = route.path.replace(/\.html$/, '').replace(/\/index$/, '/')
  const url = base + path

  // SoftwareApplication schema (always present)
  const softwareSchema = {
    '@context': 'https://schema.org',
    '@type': 'SoftwareApplication',
    name: 'Ariadne',
    description: 'Local-first AI memory system for agents. Hybrid search, cognitive retention, knowledge graph traversal. Zero infrastructure.',
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
      'FAISS vector search (auto Flat to IVF)',
      'Hybrid search with Reciprocal Rank Fusion',
      'Knowledge graph with multi-hop traversal',
      'Ebbinghaus forgetting curve',
      'MinHash LSH deduplication',
      'SQLite + FAISS, zero infrastructure',
    ],
  }

  // BreadcrumbList schema
  const parts = path.split('/').filter(Boolean)
  const breadcrumbItems = [
    { name: 'Home', url: base },
  ]

  if (parts.length > 0) {
    let currentPath = ''
    parts.forEach((part, i) => {
      currentPath += '/' + part
      const name = part
        .replace(/-/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase())
        .replace(/Api/, 'API')
        .replace(/Faq/, 'FAQ')
      breadcrumbItems.push({
        name,
        url: base + currentPath,
      })
    })
  }

  const breadcrumbSchema = {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: breadcrumbItems.map((item, i) => ({
      '@type': 'ListItem',
      position: i + 1,
      name: item.name,
      item: item.url,
    })),
  }

  return [softwareSchema, breadcrumbSchema]
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
