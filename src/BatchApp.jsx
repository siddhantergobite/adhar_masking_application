import { useEffect, useRef, useState } from 'react';
import { DisplayErrorBoundary, DocumentPreview, fileKind, humanSize, Icon } from './App.jsx';

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
    confidenceScore: null,
    confidenceSource: 'none',
    confidenceReason: '',
    summary: '',
    summarySource: '',
    summaryRequestId: '',
    showMasked: false,
  };
}

function statusLabel(item) {
  if (item.status === 'processing') return 'Processing document';
  if (item.status === 'done') return 'Masked copy ready';
  if (item.status === 'error') return 'Needs attention';
  return 'Waiting in queue';
}

function fallbackSummary(payload) {
  if (payload.outcome === 'success') {
    const docs = `${payload.document_count} Aadhaar document${payload.document_count === 1 ? '' : 's'}`;
    const regions = `${payload.masked_regions} number line${payload.masked_regions === 1 ? '' : 's'}`;
    const score = Number.isFinite(payload.pipeline_score) ? ` Local pipeline evidence score: ${payload.pipeline_score}/100.` : '';
    return `The local checks accepted a masked copy for ${docs} and redacted ${regions}.${score}`;
  }
  if (payload.outcome === 'rejected') return 'No safe masked copy was produced, so this file needs review or a clearer upload.';
  return 'Processing did not finish, so confidence could not be measured.';
}

function ConfidenceSummary({ item }) {
  const settled = item.status === 'done' || item.status === 'error';
  const summary = item.summary || (settled ? 'Preparing a short result summary…' : 'A short summary will appear after processing.');
  const sourceLabel = item.summarySource === 'azure_openai'
    ? 'Azure AI'
    : item.summarySource === 'local'
      ? 'Local summary'
      : '';
  return (
    <section className="confidence-summary" aria-label="Result summary">
      <div className="confidence-summary-heading">
        <span>Result summary</span>
        {sourceLabel && <small>{sourceLabel}</small>}
      </div>
      <p>{summary}</p>
      {item.confidenceReason && <small className="confidence-reason">{item.confidenceReason}</small>}
    </section>
  );
}

function ConfidenceGauge({ item }) {
  const settled = item.status === 'done' || item.status === 'error';
  const score = Number.isFinite(item.confidenceScore) ? item.confidenceScore : null;
  const circumference = 2 * Math.PI * 34;
  const offset = score === null ? circumference : circumference * (1 - score / 100);
  const tone = score === null ? 'unknown' : score >= 80 ? 'green' : score >= 60 ? 'amber' : 'low';
  const scoringPending = item.status === 'done' && item.confidenceSource === 'pending';
  const reading = score === null ? (settled && !scoringPending ? '—' : '…') : score;
  const scoreDescription = score === null
    ? (settled && !scoringPending ? 'not scored' : 'pending')
    : `${score} out of 100`;
  const scoreSource = item.confidenceSource === 'azure_openai' ? 'Azure AI' : 'pipeline';
  const scoreMethod = item.confidenceSource === 'azure_openai'
    ? 'Azure AI assessment of per-document blur, boundary, OCR, verification, and masking evidence.'
    : 'Local evidence score from per-document detection, scan quality, OCR, verification, and masking checks.';

  return (
    <div
      className={`confidence-gauge confidence-gauge-${tone}`}
      role="img"
      aria-label={`End-to-end masking confidence score ${scoreDescription}`}
      title={`${scoreMethod} The local post-mask OCR audit must pass before a file is returned. This is an operational score, not a probability.`}
    >
      <svg viewBox="0 0 88 88" aria-hidden="true">
        <circle className="confidence-gauge-track" cx="44" cy="44" r="34" />
        <circle
          className="confidence-gauge-progress"
          cx="44"
          cy="44"
          r="34"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 44 44)"
        />
      </svg>
      <div className="confidence-gauge-reading">
        <strong>{reading}</strong>
        {score !== null && <span>/100</span>}
      </div>
      <small>{score === null ? (settled && !scoringPending ? 'not scored' : 'pending') : scoreSource}</small>
    </div>
  );
}

function DocumentCard({ item, index, total, onRemove, onRetry, onToggleResult }) {
  // Accept both property names so uploads survive Vite hot reloads across the
  // earlier state shape change.
  const originalFile = item.originalFile ?? item.file;
  const previewMode = item.showMasked && item.maskedFile ? 'masked' : 'original';
  const originalKind = originalFile ? fileKind(originalFile) : 'unsupported';
  const maskedKind = item.maskedFile ? fileKind(item.maskedFile) : null;
  const hasResult = Boolean(item.maskedFile && item.maskedUrl);

  return (
    <article className="document-card" aria-label={`Uploaded document ${index + 1} of ${total}`}>
      <div className="document-card-header">
        <div className="document-file">
          <span className="document-index">{String(index + 1).padStart(2, '0')}</span>
          <span className="document-type-icon"><Icon name="file" size={18}/></span>
          <div>
            <strong title={originalFile?.name || 'Uploaded file'}>{originalFile?.name || 'Uploaded file'}</strong>
            <span>{fileKind(originalFile).toUpperCase()} &middot; {humanSize(originalFile?.size || 0)}</span>
          </div>
        </div>
        <button className="icon-button" type="button" onClick={() => onRemove(item.id)} aria-label={`Remove ${originalFile?.name || 'uploaded file'}`}>
          <Icon name="close" size={17}/>
        </button>
      </div>

      <div className="document-card-preview">
        <div className="document-preview-slot" hidden={previewMode !== 'original'}>
          <DisplayErrorBoundary key={`${item.id}-original`}>
            <DocumentPreview file={originalFile} kind={originalKind}/>
          </DisplayErrorBoundary>
        </div>
        {item.maskedFile && (
          <div className="document-preview-slot" hidden={previewMode !== 'masked'}>
            <DisplayErrorBoundary key={`${item.id}-masked`}>
              <DocumentPreview file={item.maskedFile} kind={maskedKind}/>
            </DisplayErrorBoundary>
          </div>
        )}
        {(item.status !== 'done' || item.showMasked) && <ConfidenceGauge item={item}/>}
      </div>

      <div className="document-card-footer">
        <div className="document-card-meta">
          {hasResult && (
            <div className="result-tabs" role="tablist" aria-label={`Preview mode for ${originalFile?.name || 'uploaded file'}`}>
              <button className={!item.showMasked ? 'selected' : ''} type="button" role="tab" aria-selected={!item.showMasked} onClick={() => onToggleResult(item.id, false)}>Original</button>
              <button className={item.showMasked ? 'selected' : ''} type="button" role="tab" aria-selected={item.showMasked} onClick={() => onToggleResult(item.id, true)}>Masked result</button>
            </div>
          )}
          <div className={`document-status status-${item.status}`}><span className="status-dot"/>{statusLabel(item)}</div>
        </div>

        {item.status === 'done' ? (
          <div className="document-card-result">
            <ConfidenceSummary item={item}/>
            <a className="download-result-button" href={item.maskedUrl} download={item.maskedFile.name}>Download masked PDF <Icon name="arrow" size={14}/></a>
          </div>
        ) : item.status === 'error' ? (
          <div className="document-card-result document-card-error">
            <ConfidenceSummary item={item}/>
            <p>{item.error}</p>
            <button className="card-retry-button" type="button" onClick={() => onRetry(item.id)}>Try again <Icon name="arrow" size={14}/></button>
          </div>
        ) : (
          <div className="document-card-result document-card-progress">
            <ConfidenceSummary item={item}/>
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
  const [workersKvConfigured, setWorkersKvConfigured] = useState(false);
  const [kvStoredUpload, setKvStoredUpload] = useState(null);
  const [folderBatch, setFolderBatch] = useState(null);
  const [folderImportBusy, setFolderImportBusy] = useState(false);
  const [folderImportAction, setFolderImportAction] = useState('');
  const [folderImportError, setFolderImportError] = useState('');
  const [queueTick, setQueueTick] = useState(0);
  const fileInputRef = useRef(null);
  const filesRef = useRef([]);
  const processingIdRef = useRef(null);
  const abortControllersRef = useRef(new Map());
  const cancelledIdsRef = useRef(new Set());

  async function requestCardSummary(id, requestId, payload) {
    const localScore = Number.isFinite(payload.pipeline_score) ? payload.pipeline_score : null;
    let result = {
      summary: fallbackSummary(payload),
      source: 'local',
      confidenceScore: localScore,
      confidenceSource: localScore === null ? 'none' : 'pipeline',
      confidenceReason: '',
    };
    try {
      const response = await fetch('/api/card-summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (response.ok) {
        const candidate = await response.json();
        const hasScore = Number.isFinite(candidate.confidence_score);
        const validConfidenceSource = ['azure_openai', 'pipeline', 'none'].includes(candidate.confidence_source);
        result = {
          summary: typeof candidate.summary === 'string' && candidate.summary.trim()
            ? candidate.summary
            : result.summary,
          source: candidate.source === 'azure_openai' ? 'azure_openai' : 'local',
          confidenceScore: hasScore ? Math.max(0, Math.min(100, candidate.confidence_score)) : result.confidenceScore,
          confidenceSource: validConfidenceSource ? candidate.confidence_source : result.confidenceSource,
          confidenceReason: typeof candidate.confidence_reason === 'string' ? candidate.confidence_reason : '',
        };
      }
    } catch {
      // Keep the card useful when Azure credentials or network access are unavailable.
    }
    setFiles((current) => current.map((candidate) => candidate.id === id && candidate.summaryRequestId === requestId
      ? {
        ...candidate,
        summary: result.summary,
        summarySource: result.source,
        confidenceScore: result.confidenceScore,
        confidenceSource: result.confidenceSource,
        confidenceReason: result.confidenceReason,
      }
      : candidate));
  }

  function reportFailure(id, message, reasonCode) {
    const rejectionCodes = new Set(['no_aadhaar_detected', 'review_required', 'no_aadhaar_number']);
    const outcome = rejectionCodes.has(reasonCode) ? 'rejected' : 'unavailable';
    const requestId = `${id}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const allowedCodes = ['no_aadhaar_detected', 'review_required', 'no_aadhaar_number', 'models_not_ready', 'ocr_failed', 'processing_failed', 'processing_timeout', 'unsupported_file', 'file_too_large'];
    const payload = {
      outcome,
      document_count: 0,
      masked_regions: 0,
      document_classes: [],
      document_confidence: null,
      verification_confidence: null,
      pipeline_score: null,
      reason_code: allowedCodes.includes(reasonCode) ? reasonCode : 'processing_failed',
    };
    setFiles((current) => current.map((candidate) => candidate.id === id ? {
      ...candidate,
      status: 'error',
      error: message,
      confidenceScore: null,
      confidenceSource: 'none',
      confidenceReason: '',
      summary: 'Preparing a short result summary…',
      summarySource: '',
      summaryRequestId: requestId,
    } : candidate));
    void requestCardSummary(id, requestId, payload);
  }

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
          setWorkersKvConfigured(status.cloud_storage?.cloudflare_workers_kv_configured === true);
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

  async function startFolderBatch(mode) {
    const storeOnly = mode === 'kvStore';
    if (folderImportBusy || (!storeOnly && engineStatus !== 'ready')) return;
    const options = {
      local: {
        endpoint: '/api/batches/folder',
        label: 'local inbox',
      },
      kvStore: {
        endpoint: '/api/batches/kv/store-folder',
        label: 'local inbox stored in Workers KV',
      },
      kvImport: {
        endpoint: '/api/batches/kv',
        label: kvStoredUpload ? 'uploaded Workers KV files' : 'private Workers KV inbox',
      },
    };
    const selected = options[mode];
    if (!selected) return;

    setFolderImportBusy(true);
    setFolderImportAction(storeOnly ? 'Storing the local inbox in Workers KV' : `Starting the ${selected.label} import`);
    setFolderImportError('');
    setFolderBatch(null);
    try {
      const response = await fetch(selected.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mode === 'kvImport'
          ? { relative_prefix: kvStoredUpload?.relative_prefix || '' }
          : { relative_folder: '.', recursive: false }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof payload.detail === 'string' ? payload.detail : 'The folder could not be imported.';
        throw new Error(detail);
      }
      if (storeOnly) {
        setKvStoredUpload(payload);
        setFolderBatch(null);
      } else {
        setFolderBatch({ ...payload, import_label: selected.label });
        if (mode === 'kvImport' && kvStoredUpload) setKvStoredUpload(null);
      }
    } catch (error) {
      setFolderImportError(error.message || 'Could not contact the local batch service.');
    } finally {
      setFolderImportBusy(false);
    }
  }

  useEffect(() => {
    const batchId = folderBatch?.batch_id;
    if (!batchId || !['queued', 'processing'].includes(folderBatch.status)) return undefined;
    let active = true;

    async function refreshBatch() {
      try {
        const response = await fetch(`/api/batches/${batchId}`);
        const payload = await response.json().catch(() => ({}));
        if (active && response.ok) {
          setFolderBatch((current) => current?.batch_id === batchId
            ? { ...payload, import_label: current.import_label }
            : current);
        }
      } catch {
        // Keep the last known job state and retry on the next poll.
      }
    }

    const timer = window.setInterval(refreshBatch, 1800);
    void refreshBatch();
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [folderBatch?.batch_id, folderBatch?.status]);

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
      confidenceScore: null,
      confidenceSource: 'none',
      confidenceReason: '',
      summary: '',
      summarySource: '',
      summaryRequestId: '',
      showMasked: false,
    } : item));
  }

  function toggleResult(id, showMasked) {
    setFiles((current) => current.map((item) => item.id === id ? { ...item, showMasked } : item));
  }

  async function processFile(item) {
    if (!item || engineStatus !== 'ready' || processingIdRef.current) return;

    const originalFile = item.originalFile ?? item.file;
    if (!originalFile) {
      reportFailure(item.id, 'The original upload is no longer available. Please upload this file again.', 'processing_failed');
      return;
    }

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
      body.append('file', originalFile);
      const response = await fetch('/api/process', { method: 'POST', body, signal: controller.signal });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const detail = payload.detail;
        const message = typeof detail === 'string' ? detail : detail?.message;
        const action = typeof detail === 'object' ? detail?.action : '';
        const error = new Error([message, action].filter(Boolean).join(' ') || 'The document could not be processed.');
        error.code = typeof detail === 'object' ? detail?.code : response.status === 413 ? 'file_too_large' : response.status === 415 ? 'unsupported_file' : 'processing_failed';
        throw error;
      }

      const blob = await response.blob();
      if (blob.type !== 'application/pdf') throw new Error('The processor returned an unexpected file.');
      const nextUrl = URL.createObjectURL(blob);

      if (cancelledIdsRef.current.has(item.id)) {
        URL.revokeObjectURL(nextUrl);
        return;
      }

      const safeName = originalFile.name.replace(/\.[^.]+$/, '');
      const maskedFile = new File([blob], `${safeName}-masked.pdf`, { type: 'application/pdf' });
      const documentConfidence = Number(response.headers.get('X-Document-Confidence') || 0);
      const verificationConfidence = Number(response.headers.get('X-Verification-Confidence') || 0);
      const scoreHeader = response.headers.get('X-Masking-Confidence');
      const confidenceScore = scoreHeader === null
        ? null
        : Number(scoreHeader);
      const documentCount = Number(response.headers.get('X-Aadhaar-Documents') || 0);
      const maskedCount = Number(response.headers.get('X-Redacted-Regions') || 0);
      const documentClasses = (response.headers.get('X-Document-Classes') || '').split(',').filter(Boolean);
      let documentEvidence = [];
      try {
        const evidenceHeader = response.headers.get('X-Confidence-Evidence');
        const parsedEvidence = evidenceHeader ? JSON.parse(evidenceHeader) : [];
        if (Array.isArray(parsedEvidence)) documentEvidence = parsedEvidence;
      } catch {
        // Keep the local score if optional per-document evidence is unavailable.
      }
      const requestId = `${item.id}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const summaryPayload = {
        outcome: 'success',
        document_count: documentCount,
        masked_regions: maskedCount,
        document_classes: documentClasses,
        document_confidence: documentConfidence,
        verification_confidence: verificationConfidence,
        pipeline_score: Number.isFinite(confidenceScore) ? confidenceScore : null,
        document_evidence: documentEvidence,
      };
      setFiles((current) => current.map((candidate) => candidate.id === item.id ? {
        ...candidate,
        status: 'done',
        maskedFile,
        maskedUrl: nextUrl,
        maskedCount,
        documentCount,
        documentClasses,
        documentConfidence,
        verificationConfidence,
        confidenceScore: null,
        confidenceSource: 'pending',
        confidenceReason: '',
        summary: 'Preparing a short result summary…',
        summarySource: '',
        summaryRequestId: requestId,
        // Show the safe result as soon as processing finishes.  The Original
        // tab remains available for an explicit comparison.
        showMasked: true,
        error: '',
      } : candidate));
      void requestCardSummary(item.id, requestId, summaryPayload);
    } catch (error) {
      if (timedOut) {
        reportFailure(item.id, 'Local processing exceeded 90 seconds. Try a clearer or more tightly framed document image.', 'processing_timeout');
      } else if (error.name !== 'AbortError' && !cancelledIdsRef.current.has(item.id)) {
        reportFailure(item.id, error.message || 'Could not reach the local processing service.', error.code || 'processing_failed');
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

          <details className="batch-import-panel" open>
            <summary>Store inbox files in Cloudflare KV, then mask them locally</summary>
            <div className="batch-import-content">
              <p>Upload copies every supported file currently inside the configured <b>AadhaarInbox</b> folder to private Workers KV. This step does not start masking and does not include subfolders. Then import the stored copies to run masking on this computer's CPU.</p>
              <div className="batch-import-actions">
                <button type="button" onClick={() => startFolderBatch('local')} disabled={engineStatus !== 'ready' || folderImportBusy}>
                  Process local inbox
                </button>
                <button type="button" onClick={() => startFolderBatch('kvStore')} disabled={folderImportBusy || !workersKvConfigured}>
                  Upload inbox to Workers KV
                </button>
                <button type="button" onClick={() => startFolderBatch('kvImport')} disabled={engineStatus !== 'ready' || folderImportBusy || !workersKvConfigured}>
                  {kvStoredUpload ? 'Process uploaded KV files' : 'Import existing Workers KV inbox'}
                </button>
              </div>
              <div className="batch-import-note">
                {workersKvConfigured
                  ? 'Workers KV is configured. Upload stores originals without starting masking; importing starts local CPU processing. Originals remain in KV. Masked copies go to a separate prefix and follow the batch retention period.'
                  : 'Workers KV is not configured yet. Local inbox processing works offline. To enable cloud storage, configure a Workers KV namespace and a scoped API token in the backend .env file. Free-plan quotas apply; the model and OCR processing continue on this computer.'}
              </div>
              {kvStoredUpload && (
                <div className="batch-import-feedback" aria-live="polite">
                  Stored {kvStoredUpload.uploaded_count} files in Workers KV. Masking has not started. In Cloudflare, open the KV Pairs tab and search this key prefix: <code>{kvStoredUpload.storage_prefix}</code>. Return here and click "Process uploaded KV files" when ready.
                </div>
              )}
              {folderImportBusy && <div className="batch-import-feedback">{folderImportAction}...</div>}
              {folderImportError && <div className="batch-import-feedback is-error">{folderImportError}</div>}
              {folderBatch && (
                <div className="batch-import-result" aria-live="polite">
                  <div className="batch-import-result-heading">
                    <b>{folderBatch.import_label || 'Folder import'}</b>
                    <span>{folderBatch.status === 'completed' || folderBatch.status === 'completed_with_issues' ? 'Finished' : `${folderBatch.completed_files || 0} of ${folderBatch.total_files} masked`}</span>
                  </div>
                  <p>{folderBatch.total_files} files · {folderBatch.rejected_files || 0} rejected · {folderBatch.failed_files || 0} failed</p>
                  <ul>
                    {folderBatch.files?.map((item) => (
                      <li key={item.file_id}>
                        <div className="batch-file-info">
                          <span>{item.file_name}</span>
                          {Number.isFinite(item.masking_confidence) && (
                            <small className="batch-file-confidence">
                              {item.masking_confidence_source === 'azure_openai' ? 'Azure AI' : 'Pipeline'} confidence {item.masking_confidence}/100
                            </small>
                          )}
                          {(item.output_storage_uri || item.output_s3_uri) && <small className="batch-file-storage">Private cloud copy: {item.output_storage_uri || item.output_s3_uri}</small>}
                        </div>
                        <b className={`batch-file-state state-${item.status}`}>{item.status}</b>
                      </li>
                    ))}
                  </ul>
                  {folderBatch.download_zip_url && (
                    <a className="download-result-button batch-import-download" href={folderBatch.download_zip_url} download>
                      Download masked PDFs and manifest <Icon name="arrow" size={14}/>
                    </a>
                  )}
                </div>
              )}
            </div>
          </details>

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
