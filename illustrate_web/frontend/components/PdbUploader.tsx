import { FC, ChangeEvent, useState, useEffect, useRef, FormEvent, useCallback } from 'react';
import { suggestPdb } from '../lib/api';

type PdbSuggestion = {
  pdb_id: string;
  title: string;
};

type PdbUploaderProps = {
  disabled?: boolean;
  onUploaded: (pdbId: string) => void | Promise<void>;
  compact?: boolean;
};

function normalizeQuery(value: string): string {
  return value.trim().toUpperCase();
}

function extractPdbId(value: string): string | null {
  const direct = normalizeQuery(value);
  if (/^[A-Z0-9]{4}$/.test(direct)) {
    return direct;
  }
  const token = direct
    .split(/\s+/)
    .find((part) => /^[A-Z0-9]{4}$/.test(part));
  return token ?? null;
}

function localSuggestionsFromQuery(value: string): PdbSuggestion[] {
  const pdbId = extractPdbId(value);
  if (!pdbId) {
    return [];
  }
  return [{ pdb_id: pdbId, title: '' }];
}

function mergeSuggestions(primary: PdbSuggestion[], secondary: PdbSuggestion[]): PdbSuggestion[] {
  const seen = new Set<string>();
  const merged: PdbSuggestion[] = [];
  for (const item of [...primary, ...secondary]) {
    const id = normalizeQuery(item.pdb_id);
    if (!id || seen.has(id)) {
      continue;
    }
    seen.add(id);
    merged.push({ pdb_id: id, title: item.title });
  }
  return merged;
}

export const PdbUploader: FC<PdbUploaderProps> = ({ disabled, onUploaded, compact }) => {
  const [query, setQuery] = useState('');
  const [uploading, setUploading] = useState(false);
  const [fetching, setFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<PdbSuggestion[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const suggestRequestRef = useRef(0);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const queryInputRef = useRef<HTMLInputElement>(null);

  const syncQueryFromDom = useCallback(() => {
    const domValue = queryInputRef.current?.value ?? '';
    const normalized = normalizeQuery(domValue);
    if (!normalized) {
      return;
    }
    setQuery((current) => (current === normalized ? current : normalized));
  }, []);

  useEffect(() => {
    const frame = window.requestAnimationFrame(syncQueryFromDom);
    return () => window.cancelAnimationFrame(frame);
  }, [syncQueryFromDom]);

  // Debounced search
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    const q = normalizeQuery(query);
    const localSuggestions = localSuggestionsFromQuery(q);
    if (q.length < 2) {
      setSuggestions([]);
      setShowDropdown(false);
      return;
    }
    setSuggestions(localSuggestions);
    setShowDropdown(localSuggestions.length > 0);
    const requestId = suggestRequestRef.current + 1;
    suggestRequestRef.current = requestId;
    debounceRef.current = setTimeout(async () => {
      const results = await suggestPdb(q);
      if (suggestRequestRef.current !== requestId) {
        return;
      }
      const merged = mergeSuggestions(results, localSuggestions);
      setSuggestions(merged);
      setShowDropdown(merged.length > 0);
    }, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query]);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const { uploadPdb } = await import('../lib/api');
      const id = await uploadPdb(file);
      await Promise.resolve(onUploaded(id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  const selectSuggestion = async (pdbId: string) => {
    const normalized = normalizeQuery(pdbId);
    if (!normalized) {
      return;
    }
    setQuery(normalized);
    setShowDropdown(false);
    setSuggestions([]);
    setFetching(true);
    setError(null);
    try {
      const { fetchPdbById } = await import('../lib/api');
      const id = await fetchPdbById(normalized);
      await Promise.resolve(onUploaded(id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setFetching(false);
    }
  };

  const handleFetch = async (e: FormEvent) => {
    e.preventDefault();
    const fromDom = queryInputRef.current?.value ?? query;
    const pdbId = extractPdbId(fromDom);
    if (!pdbId) {
      setError('Enter a 4-character PDB ID (e.g. 2HHB).');
      return;
    }
    setQuery(pdbId);
    await selectSuggestion(pdbId);
  };

  const isDisabled = Boolean(disabled) || fetching || uploading;

  return (
    <section className={compact ? 'toolbar-card' : 'utility-card'}>
      <h3 className="panel-title">{compact ? 'Structure' : 'Structure Input'}</h3>
      {!compact && <p className="subtle-text">Upload a local .pdb file or search RCSB.</p>}
      <input className="form-input" type="file" accept=".pdb,.ent" disabled={isDisabled} onChange={handleFileChange} />
      <div ref={wrapperRef} style={{ position: 'relative' }}>
        <form onSubmit={handleFetch} className={compact ? 'fetch-form compact' : 'fetch-form'}>
          <input
            ref={queryInputRef}
            className="form-input"
            type="text"
            placeholder="PDB ID or name (e.g. 2hhb, hemoglobin)"
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="characters"
            spellCheck={false}
            value={query}
            onChange={(e) => {
              setError(null);
              setQuery(e.target.value.toUpperCase());
            }}
            onInput={(e) => {
              setError(null);
              setQuery((e.target as HTMLInputElement).value.toUpperCase());
            }}
            onFocus={() => {
              syncQueryFromDom();
              if (suggestions.length > 0) setShowDropdown(true);
            }}
            disabled={isDisabled}
          />
          <button className="action-button secondary" type="submit" disabled={isDisabled}>
            {fetching ? 'Fetching…' : 'Fetch'}
          </button>
        </form>
        {showDropdown && (
          <ul className="pdb-suggestions">
            {suggestions.map((s) => (
              <li key={s.pdb_id}>
                <button type="button" className="pdb-suggestion-item" onClick={() => void selectSuggestion(s.pdb_id)}>
                  <span className="pdb-suggestion-id">{s.pdb_id}</span>
                  {s.title && <span className="pdb-suggestion-title">{s.title}</span>}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      {error && <span className="error-text">{error}</span>}
      {!compact && (fetching || uploading) && <span className="subtle-text">{uploading ? 'Uploading…' : 'Fetching structure…'}</span>}
    </section>
  );
};
