import { useEffect, useRef, useState } from 'react';
import DOMPurify from 'dompurify';

let pdfjsPromise;
async function getPdfJs() {
  pdfjsPromise ??= Promise.all([
    import('pdfjs-dist'),
    import('pdfjs-dist/build/pdf.worker.min.mjs?url'),
  ]).then(([pdfjsLib, worker]) => {
    pdfjsLib.GlobalWorkerOptions.workerSrc = worker.default;
    return pdfjsLib;
  });
  return pdfjsPromise;
}

const iconPaths = {
  shield: <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z"/><path d="m9 12 2 2 4-4"/></>,
  upload: <><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5M12 3v12"/></>,
  file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h8"/></>,
  arrow: <path d="M7 17 17 7M7 7h10v10"/>,
  lock: <><rect width="16" height="11" x="4" y="11" rx="2"/><path d="M8 11V7a4 4 0 1 1 8 0v4"/></>,
  spark: <><path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z"/><path d="m19 14 1.1 2.9L23 18l-2.9 1.1L19 22l-1.1-2.9L15 18l2.9-1.1L19 14Z"/></>,
  close: <path d="m18 6-12 12M6 6l12 12"/>,
  left: <path d="m15 18-6-6 6-6"/>,
  right: <path d="m9 18 6-6-6-6"/>,
};

const PROCESS_TIMEOUT_MS = 90_000;

function Icon({ name, size = 20 }) {
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{iconPaths[name]}</svg>;
}

function fileKind(file) {
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  if (file.type.startsWith('image/') || ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tif', 'tiff', 'gif', 'heic', 'heif'].includes(ext)) return 'image';
  if (file.type === 'application/pdf' || ext === 'pdf') return 'pdf';
  if (ext === 'docx' || file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') return 'docx';
  if (ext === 'doc') return 'doc';
  return 'unsupported';
}

function humanSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function LoadingPreview({ label = 'Preparing your local preview…' }) {
  return <div className="preview-empty"><div className="loader-ring"/><span>{label}</span></div>;
}

function UnsupportedPreview({ message }) {
  return (
    <div className="unsupported-preview">
      <span className="unsupported-icon"><Icon name="file" size={24}/></span>
      <h4>Preview unavailable</h4>
      <p>{message}</p>
    </div>
  );
}

function DocumentPreview({ file, kind }) {
  const [imageUrl, setImageUrl] = useState(null);
  const [pdf, setPdf] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [docxHtml, setDocxHtml] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (kind !== 'image') return undefined;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setLoading(true);
    setError('');
    return () => URL.revokeObjectURL(url);
  }, [file, kind]);

  useEffect(() => {
    if (kind !== 'pdf') return undefined;
    let cancelled = false;
    let loadedPdf;
    setPdf(null);
    setPageNumber(1);
    setLoading(true);
    setError('');

    async function loadPdf() {
      try {
        const pdfjsLib = await getPdfJs();
        const documentData = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
        loadedPdf = documentData;
        if (cancelled) {
          await documentData.destroy();
          return;
        }
        setPdf(documentData);
      } catch (cause) {
        if (!cancelled) setError(cause.message || 'We could not open this PDF. It may be damaged or password-protected.');
      }
    }

    loadPdf();
    return () => {
      cancelled = true;
      if (loadedPdf) void loadedPdf.destroy();
    };
  }, [file, kind]);

  useEffect(() => {
    if (kind !== 'docx') return undefined;
    let cancelled = false;
    setDocxHtml(null);
    setLoading(true);
    setError('');

    async function loadDocx() {
      try {
        const { default: mammoth } = await import('mammoth');
        const { value, messages } = await mammoth.convertToHtml({ arrayBuffer: await file.arrayBuffer() });
        if (cancelled) return;
        if (!value.trim()) {
          setError('This Word file has no previewable content. Save it as PDF and try again.');
          return;
        }
        setDocxHtml(DOMPurify.sanitize(value, { USE_PROFILES: { html: true } }));
        if (messages.length) console.info('DOCX preview notes:', messages);
      } catch (cause) {
        if (!cancelled) setError(cause.message || 'We could not open this Word document.');
      }
    }

    loadDocx();
    return () => { cancelled = true; };
  }, [file, kind]);

  useEffect(() => {
    if (kind !== 'pdf' || !pdf || !canvasRef.current) return undefined;
    let cancelled = false;
    let renderTask;
    setLoading(true);

    async function renderPage() {
      try {
        const page = await pdf.getPage(pageNumber);
        if (cancelled) return;
        const viewport = page.getViewport({ scale: 1.35 });
        const canvas = canvasRef.current;
        if (!canvas) return;
        canvas.width = Math.ceil(viewport.width);
        canvas.height = Math.ceil(viewport.height);
        renderTask = page.render({ canvasContext: canvas.getContext('2d', { alpha: false }), viewport });
        await renderTask.promise;
        if (!cancelled) setLoading(false);
      } catch (cause) {
        if (!cancelled && cause.name !== 'RenderingCancelledException') setError(cause.message || 'We could not render this PDF page.');
      }
    }

    renderPage();
    return () => {
      cancelled = true;
      renderTask?.cancel();
    };
  }, [kind, pageNumber, pdf]);

  if (kind === 'doc') {
    return <UnsupportedPreview message="Legacy .doc preview needs a document conversion service. Save it as .docx or PDF to preview in this browser."/>;
  }
  if (kind === 'unsupported') {
    return <UnsupportedPreview message="This file type is not supported for local preview. Try a JPEG, PNG, WebP, PDF, or DOCX file."/>;
  }
  if (error) {
    return <UnsupportedPreview message={error}/>;
  }
  if (kind === 'image') {
    return (
      <div className="image-frame">
        {imageUrl && <img className="image-preview" src={imageUrl} alt={`Local preview of ${file.name}`} onLoad={() => setLoading(false)} onError={() => setError('This image format cannot be previewed by the browser. Try a JPEG, PNG, or WebP file.')} />}
        {loading && <LoadingPreview/>}
      </div>
    );
  }
  if (kind === 'pdf') {
    return (
      <div className="pdf-frame">
        <canvas className="pdf-canvas" ref={canvasRef} aria-label={`PDF page ${pageNumber} preview`}/>
        {loading && <LoadingPreview/>}
        {pdf?.numPages > 1 && (
          <div className="pdf-controls">
            <button type="button" aria-label="Previous page" disabled={pageNumber <= 1} onClick={() => setPageNumber((page) => page - 1)}><Icon name="left" size={17}/></button>
            <span>Page <b>{pageNumber}</b> of {pdf.numPages}</span>
            <button type="button" aria-label="Next page" disabled={pageNumber >= pdf.numPages} onClick={() => setPageNumber((page) => page + 1)}><Icon name="right" size={17}/></button>
          </div>
        )}
      </div>
    );
  }
  if (kind === 'docx') {
    return docxHtml
      ? <article className="docx-preview" dangerouslySetInnerHTML={{ __html: docxHtml }}/>
      : <LoadingPreview label="Preparing your Word preview…"/>;
  }
  return <LoadingPreview/>;
}

function App() {
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [engineStatus, setEngineStatus] = useState('checking');
  const [engineMessage, setEngineMessage] = useState('');
  const [processing, setProcessing] = useState(false);
  const [processError, setProcessError] = useState('');
  const [maskedFile, setMaskedFile] = useState(null);
  const [maskedUrl, setMaskedUrl] = useState('');
  const [maskedCount, setMaskedCount] = useState(0);
  const [documentCount, setDocumentCount] = useState(0);
  const [documentClasses, setDocumentClasses] = useState([]);
  const [documentConfidence, setDocumentConfidence] = useState(0);
  const [verificationConfidence, setVerificationConfidence] = useState(0);
  const [showMasked, setShowMasked] = useState(false);
  const fileInputRef = useRef(null);
  const processAbortRef = useRef(null);
  const previewFile = showMasked && maskedFile ? maskedFile : file;
  const kind = previewFile ? fileKind(previewFile) : null;

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
        if (active) setEngineStatus('offline');
      }
    }
    checkEngine();
    const timer = window.setInterval(checkEngine, 4000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => () => {
    if (maskedUrl) URL.revokeObjectURL(maskedUrl);
  }, [maskedUrl]);

  function chooseFile(candidate) {
    if (!candidate) return;
    processAbortRef.current?.abort();
    setFile(candidate);
    setMaskedFile(null);
    setMaskedUrl('');
    setMaskedCount(0);
    setDocumentCount(0);
    setDocumentClasses([]);
    setDocumentConfidence(0);
    setVerificationConfidence(0);
    setShowMasked(false);
    setProcessError('');
    setProcessing(false);
  }

  function resetFile() {
    processAbortRef.current?.abort();
    setFile(null);
    setMaskedFile(null);
    setMaskedUrl('');
    setMaskedCount(0);
    setDocumentCount(0);
    setDocumentClasses([]);
    setDocumentConfidence(0);
    setVerificationConfidence(0);
    setShowMasked(false);
    setProcessError('');
    setProcessing(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  function openFilePicker() {
    if (!fileInputRef.current) return;
    fileInputRef.current.value = '';
    fileInputRef.current.click();
  }

  async function processDocument() {
    if (!file || engineStatus !== 'ready' || processing) return;
    const controller = new AbortController();
    let timedOut = false;
    const timeoutId = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, PROCESS_TIMEOUT_MS);
    processAbortRef.current = controller;
    setProcessing(true);
    setProcessError('');
    try {
      const body = new FormData();
      body.append('file', file);
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
      const safeName = file.name.replace(/\.[^.]+$/, '');
      setMaskedFile(new File([blob], `${safeName}-masked.pdf`, { type: 'application/pdf' }));
      setMaskedUrl(nextUrl);
      setMaskedCount(Number(response.headers.get('X-Redacted-Regions') || 0));
      setDocumentCount(Number(response.headers.get('X-Aadhaar-Documents') || 0));
      setDocumentClasses((response.headers.get('X-Document-Classes') || '').split(',').filter(Boolean));
      setDocumentConfidence(Number(response.headers.get('X-Document-Confidence') || 0));
      setVerificationConfidence(Number(response.headers.get('X-Verification-Confidence') || 0));
      setShowMasked(true);
    } catch (error) {
      if (timedOut) {
        setProcessError('Local processing exceeded 90 seconds. Try a clearer or more tightly framed document image.');
      } else if (error.name !== 'AbortError') {
        setProcessError(error.message || 'Could not reach the local processing service.');
      }
    } finally {
      window.clearTimeout(timeoutId);
      if (processAbortRef.current === controller) {
        processAbortRef.current = null;
        setProcessing(false);
      }
    }
  }

  useEffect(() => {
    if (file && engineStatus === 'ready' && !maskedFile && !processing && !processError) {
      void processDocument();
    }
  }, [file, engineStatus, maskedFile, processing, processError]);

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  }

  function handleFileInput(event) {
    chooseFile(event.target.files?.[0]);
    event.target.value = '';
  }

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
        <section className="intro">
          <div className="intro-copy">
            <div className="eyebrow"><span className="eyebrow-line"/> A LITTLE MORE PEACE OF MIND</div>
            <h1>Your identity.<br/><span>Only yours.</span></h1>
            <p className="intro-text">A simple, private place to prepare an Aadhaar document for safe sharing. Keep the last four digits visible. Cover the rest.</p>
            <div className="intro-proof">
              <div className="avatar-stack" aria-hidden="true"><span>A</span><span>R</span><span>✓</span></div>
              <span>Thoughtful by design</span><span className="proof-separator">·</span><span>Private file preview</span>
            </div>
          </div>
          <div className="intro-art" aria-hidden="true">
            <div className="art-orbit orbit-one"/><div className="art-orbit orbit-two"/>
            <div className="art-spark spark-one">✳</div><div className="art-spark spark-two">✦</div>
            <div className="art-card card-back"><span className="back-chip"/><span className="back-line line-a"/><span className="back-line line-b"/></div>
            <div className="art-card card-front">
              <div className="card-topline"><span className="mini-seal">✳</span><span>UNIQUE IDENTIFICATION AUTHORITY</span><span className="mini-flag">▰</span></div>
              <div className="card-content"><div className="face-placeholder"><span/></div><div className="card-data"><span className="data-rule"/><span className="data-rule short"/><span className="number-mask"><i/><i/><i/><i/><b>4826</b></span></div></div>
              <div className="card-bottom"><span>आधार</span><span>आम आदमी का अधिकार</span></div>
              <span className="scan-outline"><i/><i/><i/><i/></span>
            </div>
            <div className="art-caption">Your details, protected <span>✦</span></div>
          </div>
        </section>

        <section className="workspace" aria-labelledby="workspace-title">
          <div className="workspace-heading">
            <div><div className="section-kicker">THE MASKING DESK</div><h2 id="workspace-title">Let’s get your document ready</h2></div>
            <div className="step-indicator"><span className="step-active">01</span><span className="step-line"/><span>02</span><span className="step-line"/><span>03</span></div>
          </div>

          {!file ? (
            <div className="workspace-grid">
              <div className="upload-column">
                <div
                  className={`upload-card${dragging ? ' drag-active' : ''}`}
                  tabIndex={0}
                  role="button"
                  aria-label="Choose an Aadhaar document or drop a file here"
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
                  <input ref={fileInputRef} id="file-input" type="file" accept="image/*,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.gif,.heic,.heif,.pdf,.doc,.docx" hidden onClick={(event) => event.stopPropagation()} onChange={handleFileInput}/>
                  <div className="upload-icon"><Icon name="upload" size={23}/><span className="upload-icon-spark">✦</span></div>
                  <div className="upload-card-copy"><h3>Drop your Aadhaar here</h3><p>or <span className="browse-link">browse files</span> from your device</p></div>
                  <div className="upload-format-row"><span>JPG</span><span>PNG</span><span>PDF</span><span>DOCX</span><span className="format-more">+ more</span></div>
                  <div className="upload-max">JPG, PNG, WebP, HEIC, TIFF, PDF, DOC, or DOCX · up to 25 MB</div>
                </div>
              <div className="upload-note"><span className="note-check"><Icon name="shield" size={15}/></span><span><b>Processed by your private local service.</b> Files are not sent to Roboflow or another detection API.</span></div>
              </div>
            </div>
          ) : (
            <section className="document-workspace" aria-live="polite">
              <div className="document-toolbar">
                <div className="document-file"><span className="document-type-icon"><Icon name="file" size={18}/></span><div><strong title={previewFile.name}>{previewFile.name}</strong><span>{kind.toUpperCase()} · {humanSize(previewFile.size)}</span></div></div>
                <div className="document-toolbar-right">
                  {maskedFile && <div className="result-tabs"><button className={!showMasked ? 'selected' : ''} type="button" onClick={() => setShowMasked(false)}>Original</button><button className={showMasked ? 'selected' : ''} type="button" onClick={() => setShowMasked(true)}>Masked result</button></div>}
                  <span className="local-status"><span className="status-dot"/>{showMasked ? 'Redacted copy' : 'Local preview'}</span>
                  <button className="icon-button" type="button" onClick={resetFile} aria-label="Remove document"><Icon name="close" size={19}/></button>
                </div>
              </div>
              <div className="document-body">
                <div className="preview-panel"><DocumentPreview key={`${previewFile.name}-${previewFile.lastModified}`} file={previewFile} kind={kind}/></div>
                <aside className="process-panel">
                  <div className="process-kicker">YOUR DOCUMENT</div>
                  <h3>{maskedFile ? 'Your verified masked copy is ready' : 'Ready to detect Aadhaar only'}</h3>
                  <p className="process-copy">The private pipeline isolates an Aadhaar—not just any card—straightens its boundary, verifies it, and masks the first eight digits.</p>
                  <div className="process-steps">
                    <div className="process-step"><span className="process-step-icon">01</span><div><b>Identify Aadhaar</b><span>Segmentation ignores debit and other cards</span></div><span className="step-state">{maskedFile ? 'Done' : 'YOLO Seg'}</span></div>
                    <div className="process-step"><span className="process-step-icon">02</span><div><b>Straighten and verify</b><span>Four-corner correction and identity signals</span></div><span className="step-state">{maskedFile ? 'Done' : 'OpenCV'}</span></div>
                    <div className="process-step"><span className="process-step-icon">03</span><div><b>Locate and mask</b><span>Number detector, OCR, checksum, safety audit</span></div><span className="step-state">{maskedFile ? 'Done' : 'Fail closed'}</span></div>
                  </div>
                  {maskedFile ? (
                    <div className="engine-notice engine-success"><span className="engine-notice-dot"/><div><b>Detection and masking complete</b><span>{documentCount} Aadhaar document{documentCount === 1 ? '' : 's'} isolated; {maskedCount} number line{maskedCount === 1 ? '' : 's'} redacted.{documentClasses.length ? ` Detected: ${documentClasses.map((name) => name.replaceAll('_', ' ')).join(', ')}.` : ''} Minimum detector confidence {Math.round(documentConfidence * 100)}%, verification {Math.round(verificationConfidence * 100)}%. Review before sharing.</span></div></div>
                  ) : processing ? (
                    <div className="engine-notice"><span className="engine-notice-dot"/><div><b>Detecting and verifying locally…</b><span>CPU processing is optimized and usually completes in under a minute; difficult images may take longer.</span></div></div>
                  ) : processError ? (
                    <div className="engine-notice engine-error"><span className="engine-notice-dot"/><div><b>Could not safely finish</b><span>{processError}</span></div></div>
                  ) : engineStatus === 'ready' ? (
                    <div className="engine-notice engine-success"><span className="engine-notice-dot"/><div><b>Production pipeline is ready</b><span>Both private YOLO models and word-position OCR are loaded on this machine.</span></div></div>
                  ) : ['error', 'model_error', 'models_missing'].includes(engineStatus) ? (
                    <div className="engine-notice engine-error"><span className="engine-notice-dot"/><div><b>{engineStatus === 'models_missing' ? 'Trained models are required' : 'Detection pipeline could not start'}</b><span>{engineMessage || 'Restart the local service after checking its setup.'}</span></div></div>
                  ) : (
                    <div className="engine-notice"><span className="engine-notice-dot"/><div><b>{engineStatus === 'checking' ? 'Connecting to local detection…' : engineStatus === 'loading' ? 'Loading YOLO and OCR models…' : 'Start the local processing service'}</b><span>{engineStatus === 'loading' ? 'Model loading can take a minute the first time.' : 'In a second terminal, run: .venv\\Scripts\\python -m uvicorn backend.server:app --host 127.0.0.1 --port 8002'}</span></div></div>
                  )}
                  {maskedFile ? (
                    <a className="download-result-button" href={maskedUrl} download="masked-aadhaar.pdf">Download masked PDF <Icon name="arrow" size={14}/></a>
                  ) : (
                    <button className="process-button" type="button" onClick={processDocument} disabled={engineStatus !== 'ready' || processing}>
                      {processing ? <><span className="button-spinner"/> Processing document…</> : <>{processError ? 'Try detection again' : 'Detect & mask Aadhaar'} <Icon name="arrow" size={15}/></>}
                    </button>
                  )}
                  <button className="change-file-button" type="button" onClick={openFilePicker}>Choose another file</button>
                  <input ref={fileInputRef} type="file" accept="image/*,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.gif,.heic,.heif,.pdf,.doc,.docx" hidden onChange={handleFileInput}/>
                </aside>
              </div>
            </section>
          )}
        </section>

        <section className="how-section" id="how-section">
          <div className="how-heading"><div><div className="section-kicker">A CAREFUL THREE-STEP FLOW</div><h2>Made to be simple. Built to be careful.</h2></div><span className="how-note">No surprises. You review before anything is saved.</span></div>
          <div className="how-grid">
            <article className="how-card"><span className="how-number">01</span><span className="how-icon"><Icon name="upload" size={19}/></span><h3>Bring your document</h3><p>Choose a photo, scan, PDF, or Word document from your device.</p></article>
            <article className="how-card"><span className="how-number">02</span><span className="how-icon icon-green"><Icon name="spark" size={19}/></span><h3>Find Aadhaar only</h3><p>YOLO segmentation isolates the Aadhaar polygon; OpenCV corrects its four corners and removes surrounding objects.</p></article>
            <article className="how-card"><span className="how-number">03</span><span className="how-icon icon-peach"><Icon name="shield" size={19}/></span><h3>Verify, mask &amp; audit</h3><p>A second detector and OCR validate the number, cover the first eight digits, then check that no complete Aadhaar number remains readable.</p></article>
          </div>
        </section>
      </main>

      <footer className="footer"><a className="footer-brand" href="#top">aadh<span>aar</span><i>.</i></a><span>Designed for a little more peace of mind.</span><span className="footer-right">Your identity deserves care <span>✦</span></span></footer>
    </div>
  );
}

export default App;
export { DocumentPreview, fileKind, humanSize, Icon };
