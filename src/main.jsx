import { createRoot } from 'react-dom/client';
import App from './BatchApp.jsx';
import { DisplayErrorBoundary } from './App.jsx';
import './style.css';
import './processing.css';

createRoot(document.getElementById('root')).render(
  <DisplayErrorBoundary title="Workspace unavailable" message="The workspace hit a display error. Refresh this page and upload the original file again.">
    <App />
  </DisplayErrorBoundary>,
);
