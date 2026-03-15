/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'paper': '#fefefe',
        'ink': '#1a1a1a',
        'accent': '#d9955f',
        'accent-hover': '#cd7f47',
        'surface': {
          warm: '#1a1812',
          deep: '#14120b',
          raised: '#211f17',
          muted: '#f7f7f4',
        },
      },
      fontFamily: {
        'serif': ['Georgia', 'Times New Roman', 'serif'],
        'sans': ['Inter', 'Helvetica', 'Arial', 'sans-serif'],
        'display': ['var(--font-display)', 'Georgia', 'serif'],
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
