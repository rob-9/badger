/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'dark': '#14120b',
        'dark-surface': '#1a1812',
        'copper': '#d9955f',
        'copper-dark': '#cd7f47',
        'copper-light': '#e8a965',
        'light': '#f7f7f4',
        'light-dim': '#e8e6e0',
        'ink': '#1a1a1a',
        'clay': '#CC5833',
        'clay-light': '#E06B3F',
        'cream': '#f2f0e9',
        'charcoal': '#14120b',
        'moss': '#2E4036',
      },
      fontFamily: {
        'heading': ['"Cooper BT"', '"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        'drama': ['"Cormorant Garamond"', 'Georgia', 'serif'],
        'mono': ['"IBM Plex Mono"', 'monospace'],
        'sans': ['"Cooper BT"', '"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        '4xl': '2rem',
        '5xl': '3rem',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
