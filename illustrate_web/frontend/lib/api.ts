import type { PresetPayload, RenderPayload } from './types';

export async function uploadPdb(file: File): Promise<string> {
  const form = new FormData();
  form.append('file', file);
  const response = await fetch('/api/upload-pdb', {
    method: 'POST',
    body: form,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`upload failed: ${response.status} ${body}`);
  }
  const payload = (await response.json()) as { pdb_id: string };
  return payload.pdb_id;
}

export async function renderImage(payload: RenderPayload, options: { signal?: AbortSignal } = {}): Promise<Blob> {
  const response = await fetch('/api/render', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
    signal: options.signal,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`render failed: ${response.status} ${body}`);
  }
  return response.blob();
}

export async function fetchPdbById(id: string): Promise<string> {
  const response = await fetch('/api/fetch-pdb', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pdb_id: id }),
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`fetch failed: ${response.status} ${body}`);
  }
  const payload = (await response.json()) as { pdb_id: string };
  return payload.pdb_id;
}

export async function fetchPresets(): Promise<PresetPayload[]> {
  const response = await fetch('/api/presets', { method: 'GET' });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`presets failed: ${response.status} ${body}`);
  }
  return (await response.json()) as PresetPayload[];
}

export type PdbSuggestion = {
  pdb_id: string;
  title: string;
};

const RCSB_SUGGEST_URL = 'https://search.rcsb.org/rcsbsearch/v2/suggest';
const RCSB_QUERY_URL = 'https://search.rcsb.org/rcsbsearch/v2/query';
const RCSB_GRAPHQL_URL = 'https://data.rcsb.org/graphql';

function normalizePdbId(value: string): string {
  return value.replace(/<\/?em>/gi, '').trim().toUpperCase();
}

function dedupeSuggestions(items: PdbSuggestion[]): PdbSuggestion[] {
  const seen = new Set<string>();
  const merged: PdbSuggestion[] = [];
  for (const item of items) {
    const id = normalizePdbId(item.pdb_id);
    if (!id || seen.has(id)) {
      continue;
    }
    seen.add(id);
    merged.push({ pdb_id: id, title: item.title ?? '' });
  }
  return merged;
}

function fallbackSuggestionFromQuery(query: string): PdbSuggestion[] {
  const tokens = query
    .trim()
    .toUpperCase()
    .split(/\s+/)
    .filter((token) => token.length > 0);
  const directId = tokens.find((token) => /^[A-Z0-9]{4}$/.test(token));
  if (!directId) {
    return [];
  }
  return [{ pdb_id: directId, title: '' }];
}

async function fetchBackendSuggestions(query: string): Promise<PdbSuggestion[] | null> {
  let response: Response;
  try {
    response = await fetch(`/api/pdb-suggest?q=${encodeURIComponent(query)}`, { method: 'GET' });
  } catch {
    return null;
  }
  if (!response.ok) {
    return null;
  }
  const payload = (await response.json()) as Array<{ pdb_id?: string; title?: string }>;
  return payload
    .filter((entry) => typeof entry.pdb_id === 'string' && entry.pdb_id.length > 0)
    .map((entry) => ({ pdb_id: normalizePdbId(String(entry.pdb_id)), title: String(entry.title ?? '') }));
}

async function fetchRcsbTitles(ids: string[]): Promise<Record<string, string>> {
  if (ids.length === 0) {
    return {};
  }
  const idsLiteral = ids.map((id) => `"${id}"`).join(', ');
  const query = `{ entries(entry_ids: [${idsLiteral}]) { rcsb_id struct { title } } }`;
  let response: Response;
  try {
    response = await fetch(RCSB_GRAPHQL_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
  } catch {
    return {};
  }
  if (!response.ok) {
    return {};
  }
  type GraphQlEntry = { rcsb_id?: string; struct?: { title?: string } | null };
  const payload = (await response.json()) as { data?: { entries?: GraphQlEntry[] } };
  const entries = payload.data?.entries;
  if (!Array.isArray(entries)) {
    return {};
  }
  const titles: Record<string, string> = {};
  for (const entry of entries) {
    const id = normalizePdbId(String(entry?.rcsb_id ?? ''));
    if (!id) {
      continue;
    }
    titles[id] = String(entry?.struct?.title ?? '').trim();
  }
  return titles;
}

async function fetchRcsbSuggestions(query: string): Promise<PdbSuggestion[]> {
  const ids: string[] = [];
  const suggestPayload = {
    type: 'term',
    suggest: {
      text: query,
      completion: [{ attribute: 'rcsb_entry_container_identifiers.entry_id' }],
      size: 10,
    },
    results_content_type: ['experimental'],
  };

  try {
    const response = await fetch(`${RCSB_SUGGEST_URL}?json=${encodeURIComponent(JSON.stringify(suggestPayload))}`, {
      method: 'GET',
    });
    if (response.ok) {
      const payload = (await response.json()) as {
        suggestions?: { 'rcsb_entry_container_identifiers.entry_id'?: Array<{ text?: string }> };
      };
      const suggestEntries = payload.suggestions?.['rcsb_entry_container_identifiers.entry_id'];
      if (Array.isArray(suggestEntries)) {
        for (const item of suggestEntries) {
          const candidate = normalizePdbId(String(item?.text ?? ''));
          if (candidate && !ids.includes(candidate)) {
            ids.push(candidate);
          }
        }
      }
    }
  } catch {
    // fall through to full-text query
  }

  if (ids.length < 5) {
    const searchPayload = {
      query: {
        type: 'terminal',
        service: 'full_text',
        parameters: { value: query },
      },
      return_type: 'entry',
      request_options: {
        results_content_type: ['experimental'],
        paginate: { start: 0, rows: 10 },
      },
    };
    try {
      const response = await fetch(RCSB_QUERY_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(searchPayload),
      });
      if (response.ok) {
        const payload = (await response.json()) as { result_set?: Array<{ identifier?: string }> };
        const resultSet = payload.result_set;
        if (Array.isArray(resultSet)) {
          for (const item of resultSet) {
            const candidate = normalizePdbId(String(item?.identifier ?? ''));
            if (candidate && !ids.includes(candidate)) {
              ids.push(candidate);
            }
          }
        }
      }
    } catch {
      // keep whatever we already collected
    }
  }

  const limitedIds = ids.slice(0, 10);
  const titles = await fetchRcsbTitles(limitedIds);
  return limitedIds.map((id) => ({ pdb_id: id, title: titles[id] ?? '' }));
}

export async function suggestPdb(query: string): Promise<PdbSuggestion[]> {
  const trimmed = query.trim();
  if (trimmed.length < 2) {
    return [];
  }
  const fallback = fallbackSuggestionFromQuery(trimmed);
  const backendSuggestions = await fetchBackendSuggestions(trimmed);
  if (backendSuggestions && backendSuggestions.length > 0) {
    return dedupeSuggestions([...backendSuggestions, ...fallback]);
  }
  const directSuggestions = await fetchRcsbSuggestions(trimmed);
  if (directSuggestions.length > 0) {
    return dedupeSuggestions([...directSuggestions, ...fallback]);
  }
  return fallback;
}
