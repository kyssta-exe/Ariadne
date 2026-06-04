import DefaultTheme from 'vitepress/theme'
import SeoLayout from './SeoLayout.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout: SeoLayout,
}
