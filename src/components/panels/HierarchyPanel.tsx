import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { useEditorStore } from '../../store/editorStore';
import type { GeometryType, SceneObject } from '../../store/types';
import { engineRef } from '../../engine/engineRef';
import { loadModelFile } from '../../engine/ModelLoader';
import { exportGlb, exportObj, exportFbxFromMeshy } from '../../engine/ExportUtils';

const primitives: { type: GeometryType; label: string }[] = [
  { type: 'box', label: 'Cube' },
  { type: 'sphere', label: 'Sphere' },
  { type: 'cylinder', label: 'Cylinder' },
  { type: 'cone', label: 'Cone' },
  { type: 'torus', label: 'Torus' },
  { type: 'plane', label: 'Plane' },
  { type: 'icosahedron', label: 'Icosahedron' },
];

function AddMenu() {
  const addObject = useEditorStore((s) => s.addObject);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    engineRef.current?.keyboardManager.setAddMenuCallback(() =>
      setOpen((o) => !o),
    );
  }, []);

  useEffect(() => {
    if (!open) return;
    function close(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [open]);

  function add(type: GeometryType) {
    addObject(type);
    setOpen(false);
  }

  function handleImportClick() {
    setOpen(false);
    fileRef.current?.click();
  }

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const { name, geometry, animations, scene, hasScene } = await loadModelFile(file);
      engineRef.current?.sceneManager.importModel(name, geometry, scene, animations, hasScene);
    } catch (err) {
      console.error('Import failed:', err);
    }
    e.target.value = '';
  }

  return (
    <div className="add-menu-wrap" ref={ref}>
      <button className="add-btn" onClick={() => setOpen((o) => !o)} title="Add primitive (Shift+A)">
        +
      </button>
      <input
        ref={fileRef}
        type="file"
        accept=".fbx,.obj,.glb,.gltf"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
      {open && (
        <div className="add-menu">
          {primitives.map((p) => (
            <button
              key={p.type}
              className="add-menu-item"
              onClick={() => add(p.type)}
            >
              {p.label}
            </button>
          ))}
          <div className="add-menu-divider" />
          <button className="add-menu-item" onClick={handleImportClick}>
            Import File...
          </button>
        </div>
      )}
    </div>
  );
}

function ExportMenu({ obj }: { obj: SceneObject }) {
  const [open, setOpen] = useState(false);
  const [menuPos, setMenuPos] = useState({ x: 0, y: 0 });
  const [exporting, setExporting] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function close(e: MouseEvent) {
      const clickedBtn = btnRef.current?.contains(e.target as Node);
      const clickedMenu = menuRef.current?.contains(e.target as Node);
      if (!clickedBtn && !clickedMenu) setOpen(false);
    }
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [open]);

  function handleButtonClick(e: React.MouseEvent) {
    e.stopPropagation();
    if (!open && btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect();
      setMenuPos({ x: rect.left, y: rect.bottom + 2 });
    }
    setOpen((o) => !o);
  }

  async function handleExport(format: 'glb' | 'obj' | 'fbx') {
    setOpen(false);
    setExporting(true);
    try {
      const sm = engineRef.current?.sceneManager;
      if (!sm) return;
      const mesh = sm.getMeshById(obj.id);
      if (!mesh) return;

      if (format === 'glb') {
        const clips = sm.animationManager.getClips(obj.id);
        await exportGlb(mesh, obj.name, clips);
      } else if (format === 'obj') {
        exportObj(mesh, obj.name);
      } else if (format === 'fbx') {
        if (!obj.meshyTaskId) return;
        await exportFbxFromMeshy(obj.meshyTaskId, obj.name);
      }
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      setExporting(false);
    }
  }

  const hasFbx = Boolean(obj.meshyTaskId);

  return (
    <div className="export-menu-wrap" onClick={(e) => e.stopPropagation()}>
      <button
        ref={btnRef}
        className={`export-btn${exporting ? ' exporting' : ''}`}
        onClick={handleButtonClick}
        disabled={exporting}
        title="Export mesh"
      >
        {exporting ? '…' : '⬇'}
      </button>
      {open && createPortal(
        <div ref={menuRef} className="export-menu" style={{ left: menuPos.x, top: menuPos.y }}>
          <div className="export-menu-label">Export as</div>
          <button className="export-menu-item" onClick={() => handleExport('glb')}>
            GLB <span className="export-fmt-note">+ animations</span>
          </button>
          <button className="export-menu-item" onClick={() => handleExport('obj')}>
            OBJ
          </button>
          <button
            className={`export-menu-item${!hasFbx ? ' disabled' : ''}`}
            onClick={() => hasFbx && handleExport('fbx')}
            title={!hasFbx ? 'FBX is available for Meshy-generated models' : 'Download FBX'}
          >
            FBX {!hasFbx && <span className="export-fmt-note">Meshy only</span>}
          </button>
        </div>,
        document.body,
      )}
    </div>
  );
}

export default function HierarchyPanel() {
  const objects = useEditorStore((s) => s.objects);
  const selectedIds = useEditorStore((s) => s.selectedIds);
  const setSelection = useEditorStore((s) => s.setSelection);
  const addToSelection = useEditorStore((s) => s.addToSelection);
  const removeFromSelection = useEditorStore((s) => s.removeFromSelection);
  const updateObject = useEditorStore((s) => s.updateObject);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');

  const sorted = Object.values(objects).sort((a, b) =>
    a.name.localeCompare(b.name),
  );

  function handleClick(id: string, e: React.MouseEvent) {
    if (e.shiftKey) {
      if (selectedIds.includes(id)) removeFromSelection(id);
      else addToSelection(id);
    } else {
      setSelection([id]);
    }
  }

  function handleDoubleClick(id: string) {
    setEditingId(id);
    setEditName(objects[id].name);
  }

  function commitRename() {
    if (editingId && editName.trim()) {
      updateObject(editingId, { name: editName.trim() });
    }
    setEditingId(null);
  }

  return (
    <div className="hierarchy-panel">
      <div className="hierarchy-header">
        <span className="hierarchy-title">Scene</span>
        <AddMenu />
      </div>
      <div className="hierarchy-list">
        {sorted.length === 0 && (
          <div className="hierarchy-empty">Empty scene</div>
        )}
        {sorted.map((obj) => (
          <div
            key={obj.id}
            className={`hierarchy-item ${selectedIds.includes(obj.id) ? 'selected' : ''}`}
            onClick={(e) => handleClick(obj.id, e)}
            onDoubleClick={() => handleDoubleClick(obj.id)}
          >
            <span className="icon">&#9670;</span>
            {editingId === obj.id ? (
              <input
                className="name-input"
                value={editName}
                autoFocus
                onChange={(e) => setEditName(e.target.value)}
                onBlur={commitRename}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commitRename();
                  if (e.key === 'Escape') setEditingId(null);
                }}
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <span className="name">{obj.name}</span>
            )}
            <ExportMenu obj={obj} />
          </div>
        ))}
      </div>
    </div>
  );
}
