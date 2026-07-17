import { useEffect, useState } from 'react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'
import { getWorkspaces } from '@/api/lightrag'
import { Layers } from 'lucide-react'

export default function WorkspaceDropdown() {
  const [workspaces, setWorkspaces] = useState<string[]>([])
  const [loading, setLoading] = useState(true)

  const selectedWorkspace = useSettingsStore.use.selectedWorkspace()
  const setSelectedWorkspace = useSettingsStore.use.setSelectedWorkspace()
  const incrementGraphDataVersion = useGraphStore.use.incrementGraphDataVersion()

  useEffect(() => {
    const fetchWorkspaces = async () => {
      try {
        setLoading(true)
        const data = await getWorkspaces()
        setWorkspaces(data || [])
        
        // Auto-select the first workspace if nothing is selected yet
        if (data && data.length > 0 && !useSettingsStore.getState().selectedWorkspace) {
          useSettingsStore.getState().setSelectedWorkspace(data[0])
          useGraphStore.getState().incrementGraphDataVersion()
        }
      } catch (error) {
        console.error('Failed to fetch workspaces:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchWorkspaces()
  }, [])

  const handleValueChange = (value: string) => {
    if (value !== selectedWorkspace) {
      setSelectedWorkspace(value)
      // Trigger graph refetch with the new workspace scope
      incrementGraphDataVersion()
    }
  }

  const selectValue = selectedWorkspace || undefined

  return (
    <div className="absolute top-4 right-4 z-50 pointer-events-auto">
      <div className="bg-slate-800/80 backdrop-blur-md border border-slate-600/50 rounded-lg p-1.5 shadow-lg flex items-center gap-2">
        <div className="pl-2 text-slate-400">
          <Layers className="w-4 h-4" />
        </div>
        <Select value={selectValue} onValueChange={handleValueChange} disabled={loading}>
          <SelectTrigger className="w-[200px] h-8 bg-transparent border-0 focus:ring-0 focus:ring-offset-0 text-slate-200">
            <SelectValue placeholder="Select Workspace" />
          </SelectTrigger>
          <SelectContent className="bg-slate-800 border-slate-700 text-slate-200">
            {workspaces.map((ws) => (
              <SelectItem key={ws} value={ws}>
                {ws}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  )
}
