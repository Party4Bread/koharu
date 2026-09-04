'use client'

import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { TitleBar } from '@/components/app/TitleBar'
import { AgentPanel } from '@/components/editor/AgentPanel'
import { Editor } from '@/components/editor/Editor'
import { SettingsPage } from '@/components/preferences/SettingsPage'
import { StartView } from '@/components/start/StartView'
import { useProject } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@koharu/ui/components/dialog'
import { cn } from '@koharu/ui/lib/utils'

export function KoharuApp() {
  const { t } = useTranslation()
  const project = useProject().data
  const selectedPages = useKoharuStore((state) => state.selectedPages)
  const selectPages = useKoharuStore((state) => state.selectPages)
  const selectLayers = useKoharuStore((state) => state.selectLayers)
  const projectLoaded = project !== undefined
  const settingsOpen = useKoharuStore((state) => state.settingsOpen)
  const [startAgentOpen, setStartAgentOpen] = useState(false)
  const activePage = project?.active_page
  const editorOpen = project !== undefined && project !== null && !settingsOpen

  useEffect(() => {
    if (!projectLoaded || selectedPages.length > 0) return
    selectPages(activePage ? [activePage] : [])
    selectLayers([])
  }, [activePage, projectLoaded, selectLayers, selectPages, selectedPages.length])

  return (
    <div
      className={cn(
        'relative flex h-screen w-screen flex-col overflow-hidden text-foreground',
        editorOpen ? 'bg-transparent' : 'bg-[var(--surface-titlebar)]',
      )}
    >
      <TitleBar />
      {settingsOpen ? (
        <SettingsPage />
      ) : project === undefined ? (
        <main className='grid min-h-0 flex-1 place-items-center bg-[var(--surface-canvas)]'>
          <div className='flex items-center gap-3 text-[12px] text-muted-foreground'>
            <span className='size-2 rounded-full bg-primary' />
            {t('startup.opening')}
          </div>
        </main>
      ) : project === null ? (
        <StartView onOpenAgent={() => setStartAgentOpen(true)} />
      ) : (
        <Editor />
      )}
      <Dialog open={startAgentOpen} onOpenChange={setStartAgentOpen}>
        <DialogContent className='grid h-[min(720px,calc(100vh-4rem))] grid-rows-[auto_minmax(0,1fr)] gap-0 overflow-hidden p-0 sm:max-w-md'>
          <DialogHeader className='sr-only'>
            <DialogTitle>{t('agent.title')}</DialogTitle>
            <DialogDescription>{t('agent.emptyDescription')}</DialogDescription>
          </DialogHeader>
          <AgentPanel />
        </DialogContent>
      </Dialog>
    </div>
  )
}
