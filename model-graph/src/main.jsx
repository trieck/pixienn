import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

function App() {
  const [model, setModel] = useState(null);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(null);
  const [zoom, setZoom] = useState(1);
  const fileInput = useRef(null);
  const graph = useRef(null);

  async function loadResponse(response) {
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || 'Unable to load model');
    setModel(data);
    setSelected(null);
    setZoom(1);
    setError('');
  }

  useEffect(() => {
    fetch('/api/model').then(loadResponse).catch((reason) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (!graph.current || !model) return undefined;
    const nodes = graph.current.querySelectorAll('g.node');
    const byId = new Map(model.layers.map((layer) => [layer.index < 0 ? 'input' : `n${layer.index}`, layer]));
    nodes.forEach((node) => node.addEventListener('click', () => setSelected(byId.get(node.id) || null)));
    return undefined;
  }, [model]);

  useEffect(() => {
    if (!graph.current || !model) return;
    const normalized = query.trim().toLowerCase();
    model.layers.forEach((layer) => {
      const node = graph.current.querySelector(`#${layer.index < 0 ? 'input' : `n${layer.index}`}`);
      const text = [layer.index, layer.type, layer.name, layer.shape, JSON.stringify(layer.params)].join(' ').toLowerCase();
      node?.classList.toggle('dim', Boolean(normalized) && !text.includes(normalized));
      node?.classList.toggle('selected', selected?.index === layer.index);
    });
  }, [model, query, selected]);

  async function loadFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      await loadResponse(await fetch('/api/model', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: file.name, text }),
      }));
    } catch (reason) { setError(reason.message); }
    event.target.value = '';
  }

  const layerCount = model?.layers.length - 1 || 0;
  return <>
    <header className="topbar">
      <div className="brand"><span className="brand-mark">P</span><div><h1>PixieNN Model Graph</h1><p>Netron-style topology inspector</p></div></div>
      <div className="actions">
        <input ref={fileInput} type="file" accept=".yml,.yaml,text/yaml" onChange={loadFile} hidden />
        <button className="primary" onClick={() => fileInput.current?.click()}>Load YML</button>
        <input className="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search layers…" />
        <button onClick={() => setQuery('')}>Clear</button>
        <button onClick={() => setZoom(Math.max(.35, zoom - .15))}>−</button>
        <button onClick={() => setZoom(1)}>{Math.round(zoom * 100)}%</button>
        <button onClick={() => setZoom(Math.min(2.5, zoom + .15))}>+</button>
      </div>
    </header>
    <main className="workspace">
      <section className="canvas-shell">
        {error && <div className="error">{error}</div>}
        {model && <div ref={graph} className="graph-canvas" style={{ '--graph-zoom': zoom }} dangerouslySetInnerHTML={{ __html: model.svg }} />}
        {!model && !error && <div className="loading">Loading model…</div>}
      </section>
      <aside className="inspector">
        <div className="inspector-heading"><span>INSPECTOR</span><b>{model?.name || 'No model loaded'}</b></div>
        {selected ? <><h2>#{selected.index} {selected.name}</h2><dl><dt>Type</dt><dd>{selected.type}</dd><dt>Input shapes</dt><dd>{(selected.input_shapes || [selected.input_shape]).map((shape, index) => <div key={index}>input: {shape}</div>)}</dd><dt>Output shape</dt><dd>output: {selected.shape}</dd><dt>Inputs</dt><dd>{selected.references.length ? selected.references.map((value) => `#${value}`).join(', ') : 'sequential input'}</dd></dl><pre>{JSON.stringify(selected.params, null, 2)}</pre></> : <div className="empty"><div className="empty-icon">⌁</div><p>Select a node to inspect its inferred input/output shapes, inputs, and YAML properties.</p></div>}
        <div className="legend">{model && [...new Map(model.layers.map((layer) => [layer.type, layer])).values()].map((layer) => <span key={layer.type}><i style={{ background: layer.color }} />{layer.name}</span>)}</div>
      </aside>
    </main>
  </>;
}

createRoot(document.getElementById('root')).render(<App />);
