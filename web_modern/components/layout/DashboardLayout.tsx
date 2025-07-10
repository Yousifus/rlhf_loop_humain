'use client'

import React, { useState } from 'react'
import { 
  HomeIcon, 
  ChartBarIcon, 
  CogIcon, 
  DocumentChartBarIcon,
  CpuChipIcon,
  ArrowPathIcon,
  Bars3Icon,
  XMarkIcon,
  PencilIcon,
  ChatBubbleLeftRightIcon,
  EyeIcon
} from '@heroicons/react/24/outline'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

const navigation = [
    { name: 'Overview', href: '/', icon: HomeIcon },
    { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
    { name: 'Calibration', href: '/calibration', icon: DocumentChartBarIcon },
    { name: 'Model Evolution', href: '/evolution', icon: CpuChipIcon },
    { name: 'Drift Analysis', href: '/drift', icon: ArrowPathIcon },
    { name: 'Annotation', href: '/annotation', icon: PencilIcon },
    { name: 'Chat Interface', href: '/chat', icon: ChatBubbleLeftRightIcon },
    { name: 'Settings', href: '/settings', icon: CogIcon },
]

interface DashboardLayoutProps {
  children: React.ReactNode
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const pathname = usePathname()

  return (
    <div className="h-full flex">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="fixed inset-0 bg-gray-600 bg-opacity-75" onClick={() => setSidebarOpen(false)} />
          <div className="relative flex flex-col w-64 bg-white shadow-xl">
            <SidebarContent pathname={pathname} onItemClick={() => setSidebarOpen(false)} />
          </div>
        </div>
      )}

      {/* Desktop sidebar */}
      <div className="hidden lg:flex lg:flex-shrink-0">
        <div className="flex flex-col w-64 border-r border-gray-200 bg-white">
          <SidebarContent pathname={pathname} />
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Mobile header */}
        <div className="lg:hidden flex items-center justify-between bg-white border-b border-gray-200 px-4 py-3">
          <button
            type="button"
            className="text-gray-600 hover:text-gray-900"
            onClick={() => setSidebarOpen(true)}
          >
            <Bars3Icon className="h-6 w-6" />
          </button>
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-humain-400 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">H</span>
            </div>
            <span className="font-semibold text-gray-900">HUMAIN</span>
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1 bg-gray-50 overflow-auto">
          <div className="p-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}

function SidebarContent({ pathname, onItemClick }: { pathname: string, onItemClick?: () => void }) {
  return (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="flex items-center px-6 py-4 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-humain-400 rounded-xl flex items-center justify-center">
            <span className="text-white font-bold text-lg">H</span>
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900">HUMAIN</h1>
            <p className="text-xs text-gray-500">RLHF Dashboard</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-1">
        {navigation.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.name}
              href={item.href}
              onClick={onItemClick}
              className={`humain-nav-item ${
                isActive ? 'humain-nav-item-active' : 'humain-nav-item-inactive'
              }`}
            >
              <item.icon className="mr-3 h-5 w-5" />
              {item.name}
            </Link>
          )
        })}
      </nav>

      {/* System Status */}
      <div className="px-4 py-4 border-t border-gray-200">
        <div className="bg-gray-50 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-gray-600">System Status</span>
            <span className="humain-status-online text-xs">Online</span>
          </div>
          <div className="space-y-1 text-xs text-gray-500">
            <div className="flex justify-between">
              <span>API Response</span>
              <span className="text-green-600">145ms</span>
            </div>
            <div className="flex justify-between">
              <span>Uptime</span>
              <span>99.9%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 