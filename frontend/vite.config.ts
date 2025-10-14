import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 4200,
    open: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  optimizeDeps: {
    include: ['@angular/compiler', '@angular/common', '@angular/core', '@angular/platform-browser'],
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2020',
  },
  esbuild: {
    target: 'es2020',
  },
});
