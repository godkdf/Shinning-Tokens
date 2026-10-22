// Sum disjoint language/split groups. Never average per-file or per-language ratios.
export function summarizeCodeRows(rows, tokenizer, filters = {}) {
  const selected = rows.filter(row => row.tokenizer === tokenizer &&
    ['dataset', 'language', 'split'].every(key => !filters[key] || row[key] === filters[key]));
  if (!selected.length) return null;
  const result = {tokenizer};
  for (const key of ['documents', 'utf8_bytes', 'tokens', 'exact', 'canonical_only', 'mismatches', 'errors', 'unknown_emissions', 'utf16le_exact', 'utf16be_exact']) {
    result[key] = selected.reduce((total, row) => total + row[key], 0);
  }
  result.byte_exact = result.exact === result.documents && result.errors === 0;
  result.bytes_per_token = result.tokens > 0 && result.errors === 0 ? result.utf8_bytes / result.tokens : null;
  return result;
}

export function codeStudyCSV(rows) {
  const fields = ['tokenizer', 'documents', 'utf8_bytes', 'tokens', 'bytes_per_token', 'exact', 'canonical_only', 'mismatches', 'errors', 'unknown_emissions', 'utf16le_exact', 'utf16be_exact', 'byte_exact'];
  const escape = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
  return [fields.join(','), ...rows.map(row => fields.map(key => escape(row[key])).join(','))].join('\r\n') + '\r\n';
}
