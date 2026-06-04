<script setup>
import DefaultTheme from 'vitepress/theme'
import { useRoute, useData } from 'vitepress'
import { computed, watch, onMounted } from 'vue'

const { Layout } = DefaultTheme
const route = useRoute()
const { frontmatter, site } = useData()

const pageTitle = computed(() => {
  const title = frontmatter.value.title || site.value.title
  if (title === 'Ariadne') return 'Ariadne — Memory for AI Agents'
  return `${title} — Ariadne`
})

const pageDescription = computed(() => {
  return frontmatter.value.description || site.value.description
})

const canonicalUrl = computed(() => {
  const base = 'https://ariadne.mantes.net'
  const path = route.path.replace(/\.html$/, '').replace(/\/index$/, '/')
  return base + (path === '/' ? '/' : path + '/')
})

function updateHead() {
  // Update title
  document.title = pageTitle.value

  // Update or create meta tags
  const setMeta = (name, content, isProperty = false) => {
    const attr = isProperty ? 'property' : 'name'
    let el = document.querySelector(`meta[${attr}="${name}"]`)
    if (!el) {
      el = document.createElement('meta')
      el.setAttribute(attr, name)
      document.head.appendChild(el)
    }
    el.setAttribute('content', content)
  }

  setMeta('og:title', pageTitle.value, true)
  setMeta('og:description', pageDescription.value, true)
  setMeta('og:url', canonicalUrl.value, true)
  setMeta('twitter:title', pageTitle.value)
  setMeta('twitter:description', pageDescription.value)

  // Update canonical
  let canonical = document.querySelector('link[rel="canonical"]')
  if (!canonical) {
    canonical = document.createElement('link')
    canonical.setAttribute('rel', 'canonical')
    document.head.appendChild(canonical)
  }
  canonical.setAttribute('href', canonicalUrl.value)
}

onMounted(updateHead)
watch(() => route.path, updateHead)
</script>

<template>
  <Layout>
    <template #layout-top>
      <SchemaOrg />
    </template>
  </Layout>
</template>
