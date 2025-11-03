/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config) => {
    // Disable canvas and encoding modules for pdfjs-dist compatibility
    // These modules are optional dependencies that aren't needed in browser environments
    config.resolve.alias.canvas = false;
    config.resolve.alias.encoding = false;
    return config;
  },
}

module.exports = nextConfig