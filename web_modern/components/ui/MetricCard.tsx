import React from 'react'
import { ArrowTrendingUpIcon, ArrowTrendingDownIcon } from '@heroicons/react/24/outline'

interface MetricCardProps {
  title: string
  value: string | number
  change?: string
  changeType?: 'positive' | 'negative' | 'neutral'
  icon?: React.ComponentType<React.SVGProps<SVGSVGElement>>
  description?: string
  loading?: boolean
}

export default function MetricCard({
  title,
  value,
  change,
  changeType = 'neutral',
  icon: Icon,
  description,
  loading = false
}: MetricCardProps) {
  if (loading) {
    return (
      <div className="humain-card animate-pulse">
        <div className="humain-card-content">
          <div className="flex items-center justify-between">
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
            <div className="h-8 w-8 bg-gray-200 rounded"></div>
          </div>
          <div className="mt-4">
            <div className="h-8 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-3 bg-gray-200 rounded w-1/4"></div>
          </div>
        </div>
      </div>
    )
  }

  const changeIcon = changeType === 'positive' ? ArrowTrendingUpIcon : ArrowTrendingDownIcon
  const changeColorClass = {
    positive: 'text-green-600',
    negative: 'text-red-600',
    neutral: 'text-gray-600'
  }[changeType]

  return (
    <div className="humain-card group hover:scale-105 transition-transform duration-200">
      <div className="humain-card-content">
        {/* Header */}
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium text-gray-600 group-hover:text-gray-900 transition-colors">
            {title}
          </h3>
          {Icon && (
            <div className="p-2 bg-humain-50 rounded-lg group-hover:bg-humain-100 transition-colors">
              <Icon className="h-5 w-5 text-humain-500" />
            </div>
          )}
        </div>

        {/* Value */}
        <div className="mt-4">
          <div className="text-2xl font-bold text-gray-900 group-hover:text-humain-600 transition-colors">
            {value}
          </div>
          
          {/* Change indicator */}
          {change && (
            <div className="mt-2 flex items-center space-x-1">
              {changeType !== 'neutral' && (
                <div className={`${changeColorClass}`}>
                  {React.createElement(changeIcon, { className: "h-3 w-3" })}
                </div>
              )}
              <span className={`text-xs font-medium ${changeColorClass}`}>
                {change}
              </span>
              <span className="text-xs text-gray-500">vs last period</span>
            </div>
          )}
          
          {/* Description */}
          {description && (
            <p className="mt-1 text-xs text-gray-500">
              {description}
            </p>
          )}
        </div>
      </div>
    </div>
  )
} 