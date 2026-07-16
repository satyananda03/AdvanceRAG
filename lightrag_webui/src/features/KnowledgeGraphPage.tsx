import { useEffect, useRef } from 'react'
import ThemeProvider from '@/components/ThemeProvider'
import TabVisibilityProvider from '@/contexts/TabVisibilityProvider'
import GraphViewer from '@/features/GraphViewer'
import { useBackendState, useAuthStore } from '@/stores/state'
import { getAuthStatus } from '@/api/lightrag'
import { useSettingsStore } from '@/stores/settings'
import { GraphReadonlyProvider } from '@/contexts/GraphReadonlyContext'
import i18n from '@/i18n'

/**
 * Full-page Knowledge Graph view at /webui/#/knowledge-graph
 * No tabs, no header — just the graph taking up the entire viewport.
 * Auto-applies: dark theme, English language, legend visible.
 */
function KnowledgeGraphPage() {
  const enableHealthCheck = useSettingsStore.use.enableHealthCheck()
  const versionCheckRef = useRef(false)

  // Force dark theme, English, and legend on mount
  useEffect(() => {
    useSettingsStore.getState().setTheme('dark')
    useSettingsStore.getState().setShowLegend(true)

    // Force English language
    if (i18n.language !== 'en') {
      i18n.changeLanguage('en')
    }
    useSettingsStore.getState().setLanguage('en')
  }, [])

  // Initialize auth/version (same as App.tsx but minimal)
  useEffect(() => {
    const checkVersion = async () => {
      if (versionCheckRef.current) return
      versionCheckRef.current = true

      try {
        const token = localStorage.getItem('LIGHTRAG-API-TOKEN')
        const status = await getAuthStatus()

        if (!status.auth_configured && status.access_token) {
          useAuthStore.getState().login(
            status.access_token,
            true,
            status.core_version,
            status.api_version,
            status.webui_title || null,
            status.webui_description || null
          )
        } else if (token && (status.core_version || status.api_version)) {
          const isGuestMode = status.auth_mode === 'disabled' || useAuthStore.getState().isGuestMode
          useAuthStore.getState().login(
            token,
            isGuestMode,
            status.core_version,
            status.api_version,
            status.webui_title || null,
            status.webui_description || null
          )
        }
      } catch (error) {
        console.error('Failed to get version info:', error)
      }
    }

    checkVersion()
  }, [])

  // Health check
  useEffect(() => {
    if (!enableHealthCheck) return

    const performHealthCheck = async () => {
      try {
        await useBackendState.getState().check()
      } catch (error) {
        console.error('Health check error:', error)
      }
    }

    useBackendState.getState().setHealthCheckFunction(performHealthCheck)
    useBackendState.getState().resetHealthCheckTimer()

    return () => {
      useBackendState.getState().clearHealthCheckTimer()
    }
  }, [enableHealthCheck])

  return (
    <ThemeProvider>
      <TabVisibilityProvider>
        <GraphReadonlyProvider value={true}>
          <div className="h-screen w-screen overflow-hidden dark relative">
            <GraphViewer />
            {/* Narasi branding pojok kanan atas */}
            <div className="absolute top-4 right-4 z-50 pointer-events-none select-none">
              <div className="bg-slate-800/80 backdrop-blur-md border border-slate-600/50 rounded-lg px-4 py-2 shadow-lg">
                <span className="text-sm font-semibold tracking-wide text-emerald-400 drop-shadow-sm">
                  Ekosistem Data &amp; Knowledge Base Ganusa
                </span>
              </div>
            </div>
          </div>
        </GraphReadonlyProvider>
      </TabVisibilityProvider>
    </ThemeProvider>
  )
}

export default KnowledgeGraphPage
