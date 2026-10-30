import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5174,
    strictPort: true,
    watch: {
      // This workspace contains large Python environments and training assets;
      // none are frontend sources, so keep them out of Vite's file watcher.
      ignored: [
        '**/.venv/**',
        '**/.venv311/**',
        '**/datasets/**',
        '**/backend/**',
        '**/training/**',
        '**/runs/**',
        '**/artifacts/**',
        '**/output/**',
        '**/tmp/**',
        '**/dist/**',
        '**/.client-showcase-build/**',
        '**/qa_aadhaar_docx/**',
        '**/qa_aadhaar_docx_updated/**',
      ],
    },
    proxy: {
      '/api': 'http://127.0.0.1:8002',
    },
  },
});
