export function formatDate(timestamp: number): string {
  const today = new Date()
  today.setHours(0, 0, 0, 0)
  const dateDay = new Date(timestamp)
  dateDay.setHours(0, 0, 0, 0)
  const diffDays = Math.round((today.getTime() - dateDay.getTime()) / (1000 * 60 * 60 * 24))

  if (diffDays === 0) {
    return 'Today'
  } else if (diffDays === 1) {
    return 'Yesterday'
  } else if (diffDays < 7) {
    return `${diffDays} days ago`
  } else {
    return new Date(timestamp).toLocaleDateString()
  }
}
