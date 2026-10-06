import { useEffect, useRef, useState } from 'react';
import { DocumentPreview, fileKind, humanSize, Icon } from './App.jsx';

const PROCESS_TIMEOUT_MS = 90_000;
const FILE_ACCEPT = 'image/*,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.gif,.heic,.heif,.pdf,.doc,.docx';

function fileIdentity(file) {
  return `${file.name}-${file.size}-${file.lastModified}`;
}

function createFileItem(file) {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    sourceKey: fileIdentity(file),
    file,
    status: 'queued',
    error: '',
    maskedFile: null,
    maskedUrl: '',
    maskedCount: 0,
    documentCount: 0,
    documentClasses: [],
    documentConfidence: 0,
    verificationConfidence: 0,
    showMasked: false,
  };
}

function statusLabel(item) {
  if (item.status === 'processing') return 'Processing locally';
  if (item.status === 'done') return 'Masked copy ready';
  if (item.status === 'error') return 'Needs attention';
  return 'Waiting in queue';
}

function DocumentCard({ item, index, total, onRemove, onRetry, onToggleResult }) {
  const previewFile = item.showMasked && item.maskedFile ? item.maskedFile : item.file;
  const kind = fileKind(previewFile);
  const hasResult = Boolean(item.maskedFile && item.maskedUrl);

  return (
    <article className="document-card" aria-label={`Uploaded document ${index + 1} of ${total}`}>
      <div className="document-card-header">
        <div className="document-file">
          <span className="document-index">{String(index + 1).padStart(2, '0')}</span>
          <span className="document-type-icon"><Icon name="file" size={18}/></span>
          <div>
            <strong title={item.file.name}>{item.file.name}</strong>
            <span>{fileKind(item.file).toUpperCase()} &middot; {humanSize(item.file.size)}</span>
          </div>
        </div>
        <button className="icon-button" type="button" onClick={() => onRemove(item.id)} aria-label={`Remove ${item.file.name}`}>
          <Icon name="close" size={17}/>
        </button>
      </div>

      <div className="document-card-preview">
        <DocumentPreview key={`${previewFile.name}-${previewFile.lastModified}-${item.showMasked}`} file={previewFile} kind={kind}/>
      </div>

      <div className="document-card-footer">
        <div className="document-card-meta">
          {hasResult && (
            <div className="result-tabs" role="tablist" aria-label={`Preview mode for ${item.file.name}`}>
              <button className={!item.showMasked ? 'selected' : ''} type="button" role="tab" aria-selected={!item.showMasked} onClick={() => onToggleResult(item.id, false)}>Original</button>
              <button className={item.showMasked ? 'selected' : ''} type="button" role="tab" aria-selected={item.showMasked} onClick={() => onToggleResult(item.id, true)}>Masked result</button>
            </div>
          )}
          <div className={`document-status status-${item.status}`}><span className="status-dot"/>{statusLabel(item)}</div>
        </div>

        {item.status === 'done' ? (
          <div className="document-card-result">
            <p>{item.documentCount} Aadhaar document{item.documentCount === 1 ? '' : 's'} isolated; {item.maskedCount} number line{item.maskedCount === 1 ? '' : 's'} redacted.{item.documentClasses.length ? ` Detected: ${item.documentClasses.map((name) => name.replaceAll('_', ' ')).join(', ')}.` : ''}</p>
            <a className="download-result-button" href={item.maskedUrl} download={item.maskedFile.name}>Download masked PDF <Icon name="arrow" size={14}/></a>
          </div>
        ) : item.status === 'error' ? (
          <div className="document-card-result document-card-error">
            <p>{item.error}</p>
            <button className="card-retry-button" type="button" onClick={() => onRetry(item.id)}>Try again <Icon name="arrow" size={14}/></button>
          </div>
        ) : (
          <div className="document-card-result document-card-progress">
            <p>{item.status === 'processing' ? 'The private pipeline is identifying, verifying, and masking this document.' : 'This file will be processed automatically when the local service is ready.'}</p>
          </div>
        )}
      </div>
    </article>
  );
}

function BatchApp() {
  const [files, setFiles] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [engineStatus, setEngineStatus] = useState('checking');
  const [engineMessage, setEngineMessage] = useState('');
  const [queueTick, setQueueTick] = useState(0);
  const fileInputRef = useRef(null);
  const filesRef = useRef([]);
  const processingIdRef = useRef(null);
  const abortControllersRef = useRef(new Map());
  const cancelledIdsRef = useRef(new Set());

  useEffect(() => {
    filesRef.current = files;
  }, [files]);

  useEffect(() => {
    let active = true;

    async function checkEngine() {
      try {
        const response = await fetch('/api/health');
        if (!response.ok) throw new Error('Local processing service is unavailable.');
        const status = await response.json();
        if (active) {
          setEngineStatus(status.ready ? 'ready' : (status.status || 'loading'));
          setEngineMessage(status.detail || '');
        }
      } catch {
        if (active) {
          setEngineStatus('offline');
          setEngineMessage('Start the local API service, then selected files will continue automatically.');
        }
      }
    }

    checkEngine();
    const timer = window.setInterval(checkEngine, 4000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => () => {
    abortControllersRef.current.forEach((controller) => controller.abort());
    filesRef.current.forEach((item) => {
      if (item.maskedUrl) URL.revokeObjectURL(item.maskedUrl);
    });
  }, []);

  function openFilePicker() {
    if (!fileInputRef.current) return;
    fileInputRef.current.value = '';
    fileInputRef.current.click();
  }

  function addFiles(candidates) {
    const incoming = Array.from(candidates || []).filter(Boolean);
    if (!incoming.length) return;

    setFiles((current) => {
      const seen = new Set(current.map((item) => item.sourceKey));
      const additions = incoming.filter((file) => {
        const key = fileIdentity(file);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      }).map(createFileItem);
      return additions.length ? [...current, ...additions] : current;
    });
  }

  function removeFile(id) {
    cancelledIdsRef.current.add(id);
    abortControllersRef.current.get(id)?.abort();
    abortControllersRef.current.delete(id);
    const item = filesRef.current.find((candidate) => candidate.id === id);
    if (item?.maskedUrl) URL.revokeObjectURL(item.maskedUrl);
    setFiles((current) => current.filter((candidate) => candidate.id !== id));
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  function clearFiles() {
    filesRef.current.forEach((item) => {
      cancelledIdsRef.current.add(item.id);
      if (item.maskedUrl) URL.revokeObjectURL(item.maskedUrl);
    });
    abortControllersRef.current.forEach((controller) => controller.abort());
    abortControllersRef.current.clear();
    setFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  function retryFile(id) {
    cancelledIdsRef.current.delete(id);
    setFiles((current) => current.map((item) => item.id === id ? {
      ...item,
      status: 'queued',
      error: '',
      showMasked: false,
    } : item));
  }

  function toggleResult(id, showMasked) {
    setFiles((current) => current.map((item) => item.id === id ? { ...item, showMasked } : item));
  }

  async function processFile(item) {
    if (!item || engineStatus !== 'ready' || processingIdRef.current) return;

    const controller = new AbortController();
    let timedOut = false;
    const timeoutId = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, PROCESS_TIMEOUT_MS);

    processingIdRef.current = item.id;
    abortControllersRef.current.set(item.id, controller);
    setFiles((current) => current.map((candidate) => candidate.id === item.id ? { ...candidate, status: 'processing', error: '' } : candidate));

    try {
      const body = new FormData();
      body.append('file', item.file);
      const response = await fetch('/api/process', { method: 'POST', body, signal: controller.signal });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const detail = payload.detail;
        const message = typeof detail === 'string' ? detail : detail?.message;
        const action = typeof detail === 'object' ? detail?.action : '';
        throw new Error([message, action].filter(Boolean).join(' ') || 'The document could not be processed.');
      }

      const blob = await response.blob();
      if (blob.type !== 'application/pdf') throw new Error('The processor returned an unexpected file.');
      const nextUrl = URL.createObjectURL(blob);

      if (cancelledIdsRef.current.has(item.id)) {
        URL.revokeObjectURL(nextUrl);
        return;
      }

      const safeName = item.file.name.replace(/\.[^.]+$/, '');
      const maskedFile = new File([blob], `${safeName}-masked.pdf`, { type: 'application/pdf' });
      setFiles((current) => current.map((candidate) => candidate.id === item.id ? {
        ...candidate,
        status: 'done',
        maskedFile,
        maskedUrl: nextUrl,
        maskedCount: Number(response.headers.get('X-Redacted-Regions') || 0),
        documentCount: Number(response.headers.get('X-Aadhaar-Documents') || 0),
        documentClasses: (response.headers.get('X-Document-Classes') || '').split(',').filter(Boolean),
        documentConfidence: Number(response.headers.get('X-Document-Confidence') || 0),
        verificationConfidence: Number(response.headers.get('X-Verification-Confidence') || 0),
        // Show the safe result as soon as processing finishes.  The Original
        // tab remains available for an explicit comparison.
        showMasked: true,
        error: '',
      } : candidate));
    } catch (error) {
      if (timedOut) {
        setFiles((current) => current.map((candidate) => candidate.id === item.id ? { ...candidate, status: 'error', error: 'Local processing exceeded 90 seconds. Try a clearer or more tightly framed document image.' } : candidate));
      } else if (error.name !== 'AbortError' && !cancelledIdsRef.current.has(item.id)) {
        setFiles((current) => current.map((candidate) => candidate.id === item.id ? { ...candidate, status: 'error', error: error.message || 'Could not reach the local processing service.' } : candidate));
      }
    } finally {
      window.clearTimeout(timeoutId);
      abortControllersRef.current.delete(item.id);
      if (processingIdRef.current === item.id) processingIdRef.current = null;
      setQueueTick((tick) => tick + 1);
    }
  }

  useEffect(() => {
    if (engineStatus !== 'ready' || processingIdRef.current) return;
    const nextItem = files.find((item) => item.status === 'queued');
    if (nextItem) void processFile(nextItem);
  }, [files, engineStatus, queueTick]);

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    addFiles(event.dataTransfer.files);
  }

  function handleFileInput(event) {
    addFiles(event.target.files);
    event.target.value = '';
  }

  const completedCount = files.filter((item) => item.status === 'done').length;
  const processingCount = files.filter((item) => item.status === 'processing').length;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Aadhaar Mask home">
          <span className="brand-mark"><Icon name="shield" size={21}/></span>
          <span className="brand-name">aadh<span>aar</span><i>.</i></span>
        </a>
        <div className="topbar-right">
          <span className="private-pill"><span className="pulse-dot"/><Icon name="lock" size={15}/> Local processing</span>
        </div>
      </header>

      <main id="top">
        <section className="workspace batch-page" aria-labelledby="workspace-title">
          <div className="workspace-heading">
            <div>
              <div className="section-kicker">THE MASKING DESK</div>
              <h1 id="workspace-title">Mask multiple Aadhaar files at once</h1>
            </div>
            <div className="step-indicator"><span className="step-active">01</span><span className="step-line"/><span>02</span><span className="step-line"/><span>03</span></div>
          </div>

          {!files.length ? (
            <div className="workspace-grid">
              <div className="upload-column">
                <div
                  className={`upload-card${dragging ? ' drag-active' : ''}`}
                  tabIndex={0}
                  role="button"
                  aria-label="Choose Aadhaar documents or drop files here"
                  onClick={openFilePicker}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      openFilePicker();
                    }
                  }}
                  onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
                  onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                  onDragLeave={(event) => { event.preventDefault(); setDragging(false); }}
                  onDrop={handleDrop}
                >
                  <input ref={fileInputRef} id="file-input" type="file" accept={FILE_ACCEPT} multiple hidden onClick={(event) => event.stopPropagation()} onChange={handleFileInput}/>
                  <div className="upload-icon"><Icon name="upload" size={23}/><span className="upload-icon-spark">*</span></div>
                  <div className="upload-card-copy"><h2>Drop your Aadhaar files here</h2><p>or <span className="browse-link">browse multiple files</span> from your device</p></div>
                  <div className="upload-format-row"><span>JPG</span><span>PNG</span><span>PDF</span><span>DOCX</span><span className="format-more">+ more</span></div>
                  <div className="upload-max">Select several files in one action &middot; up to 25 MB per file</div>
                </div>
                <div className="upload-note"><span className="note-check"><Icon name="shield" size={15}/></span><span><b>Processed by your private local service.</b> Every file gets its own preview and masked download.</span></div>
              </div>
            </div>
          ) : (
            <section className="multi-workspace" aria-live="polite">
              <div className="multi-toolbar">
                <div>
                  <div className="section-kicker">YOUR UPLOADS</div>
                  <h2>{files.length} file{files.length === 1 ? '' : 's'} selected</h2>
                  <p>{completedCount} ready{processingCount ? ` &middot; ${processingCount} processing` : ''} &middot; each file has its own preview and download.</p>
                </div>
                <div className="multi-actions">
                  <input ref={fileInputRef} type="file" accept={FILE_ACCEPT} multiple hidden onChange={handleFileInput}/>
                  <button className="multi-add-button" type="button" onClick={openFilePicker}><Icon name="upload" size={15}/> Add more files</button>
                  <button className="multi-clear-button" type="button" onClick={clearFiles}>Clear all</button>
                </div>
              </div>

              <div className="multi-progress" aria-label={`${completedCount} of ${files.length} files processed`}>
                <span><b>{completedCount}</b> of {files.length} masked</span>
                <div className="multi-progress-track"><span style={{ width: `${files.length ? (completedCount / files.length) * 100 : 0}%` }}/></div>
              </div>

              {engineStatus !== 'ready' && (
                <div className={`multi-engine-notice ${['error', 'model_error', 'models_missing', 'offline'].includes(engineStatus) ? 'is-error' : ''}`}>
                  <span className="engine-notice-dot"/>
                  <div>
                    <b>{engineStatus === 'checking' ? 'Connecting to local detection' : engineStatus === 'loading' ? 'Loading local models' : engineStatus === 'offline' ? 'Local processing service is unavailable' : engineStatus === 'models_missing' ? 'Trained models are required' : 'Detection pipeline could not start'}</b>
                    <span>{engineMessage || 'Selected files will stay in the queue until the local processor is ready.'}</span>
                  </div>
                </div>
              )}

              <div className="documents-grid">
                {files.map((item, index) => (
                  <DocumentCard
                    key={item.id}
                    item={item}
                    index={index}
                    total={files.length}
                    onRemove={removeFile}
                    onRetry={retryFile}
                    onToggleResult={toggleResult}
                  />
                ))}
              </div>
            </section>
          )}
        </section>
      </main>
    </div>
  );
}

export default BatchApp;
