/* Regression engine: runs the whole analysis in the browser for any CSV.
 *
 * Pipeline mirrors run_boston_feature_selection.py:
 *   parse CSV -> pick target -> one-hot categoricals (first level is the reference)
 *   -> 80/20 split (seeded) -> five feature-selection methods, k = 1..K
 *   -> per k: OLS on the training set (SE, t, p, CI, VIF, F) and metrics on the test set.
 *
 * Output has the same shape as MODEL_DATA / BOSTON_DATA, so the dashboard can render it unchanged.
 */
(function (root) {
  'use strict';

  const MAX_FEATURES = 40;
  const MAX_LEVELS = 8;
  const EXHAUSTIVE_MAX = 15;
  const MIN_ROWS = 20;
  const MAX_CHARS = 25e6;
  const SEED = 0;

  // ------------------------------------------------------------------
  // CSV
  // ------------------------------------------------------------------
  function detectDelimiter(text) {
    const lines = text.split(/\r?\n/);
    const line = lines.find(l => l.trim() !== '') || '';
    let best = ',', bestCount = 0;
    [',', ';', '\t', '|'].forEach(d => {
      let inQ = false, count = 0;
      for (const ch of line) {
        if (ch === '"') inQ = !inQ;
        else if (ch === d && !inQ) count++;
      }
      if (count > bestCount) { best = d; bestCount = count; }
    });
    return best;
  }

  function parseCSV(text) {
    text = text.replace(/^﻿/, '');
    const delim = detectDelimiter(text);
    const rows = [];
    let row = [], field = '', inQ = false;
    const endRow = () => {
      row.push(field);
      field = '';
      if (row.length > 1 || row[0].trim() !== '') rows.push(row);
      row = [];
    };
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (inQ) {
        if (ch === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false;
        } else field += ch;
      } else if (ch === '"' && field === '') inQ = true;
      else if (ch === delim) { row.push(field); field = ''; }
      else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        endRow();
      } else field += ch;
    }
    if (field !== '' || row.length) endRow();
    return rows;
  }

  const MISSING = /^(|na|n\/a|nan|null|none|\?|-)$/i;

  function readTable(text) {
    if (typeof text !== 'string' || !text.trim()) throw new Error('The file is empty.');
    if (text.length > MAX_CHARS) throw new Error('The file is larger than 25 MB. Please use a smaller sample.');
    if (/^\s*<(!doctype|html)/i.test(text)) {
      throw new Error('That link returned a web page, not a CSV file. For GitHub files, use the "Raw" link.');
    }
    const rows = parseCSV(text);
    if (rows.length < 2) throw new Error('Could not find a header row plus data rows in this file.');

    const seen = {};
    const header = rows[0].map((h, i) => {
      let name = h.trim() || `column_${i + 1}`;
      if (seen[name]) name = `${name}_${seen[name]++ + 1}`; else seen[name] = 1;
      return name;
    });
    const width = header.length;
    const body = rows.slice(1);

    const columns = header.map((name, j) => {
      const values = new Array(body.length);
      for (let i = 0; i < body.length; i++) values[i] = (body[i][j] ?? '').trim();
      const nums = new Float64Array(body.length);
      let numeric = true, nonMissing = 0;
      for (let i = 0; i < body.length; i++) {
        const v = values[i];
        if (MISSING.test(v)) { nums[i] = NaN; continue; }
        nonMissing++;
        const x = Number(v);
        if (!isFinite(x)) { numeric = false; }
        nums[i] = x;
      }
      return { name, values, nums, numeric: numeric && nonMissing > 0, nonMissing };
    });
    return { columns, nRows: body.length, width };
  }

  // ------------------------------------------------------------------
  // Distributions (for p-values and confidence intervals)
  // ------------------------------------------------------------------
  function lgamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) ser += c[j] / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }

  function betacf(a, b, x) {
    const FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 3e-14) break;
    }
    return h;
  }

  // Regularised incomplete beta I_x(a, b)
  function betai(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2)
      ? bt * betacf(a, b, x) / a
      : 1 - bt * betacf(b, a, 1 - x) / b;
  }

  const tTwoSided = (t, df) => betai(df / 2, 0.5, df / (df + t * t));
  const fUpper = (F, d1, d2) => (F <= 0 ? 1 : betai(d2 / 2, d1 / 2, d2 / (d2 + d1 * F)));

  function tCritical(df) {
    let lo = 0, hi = 1e3;
    for (let i = 0; i < 100; i++) {
      const mid = (lo + hi) / 2;
      if (tTwoSided(mid, df) > 0.05) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  // ------------------------------------------------------------------
  // Small linear algebra on flat row-major arrays
  // ------------------------------------------------------------------
  function cholesky(A, q) {
    const L = new Float64Array(q * q);
    for (let i = 0; i < q; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i * q + j];
        for (let k = 0; k < j; k++) s -= L[i * q + k] * L[j * q + k];
        if (i === j) {
          if (!(s > 1e-12)) return null;
          L[i * q + i] = Math.sqrt(s);
        } else L[i * q + j] = s / L[j * q + j];
      }
    }
    return L;
  }

  // Solve A x = b for symmetric positive (semi)definite A; adds a tiny ridge only if A is singular.
  function solveSPD(A, b, q) {
    let trace = 0;
    for (let i = 0; i < q; i++) trace += A[i * q + i];
    let ridge = 0;
    for (let attempt = 0; attempt < 6; attempt++) {
      let M = A;
      if (ridge > 0) { M = Float64Array.from(A); for (let i = 0; i < q; i++) M[i * q + i] += ridge; }
      const L = cholesky(M, q);
      if (L) {
        const y = new Float64Array(q), x = new Float64Array(q);
        for (let i = 0; i < q; i++) {
          let s = b[i];
          for (let k = 0; k < i; k++) s -= L[i * q + k] * y[k];
          y[i] = s / L[i * q + i];
        }
        for (let i = q - 1; i >= 0; i--) {
          let s = y[i];
          for (let k = i + 1; k < q; k++) s -= L[k * q + i] * x[k];
          x[i] = s / L[i * q + i];
        }
        return x;
      }
      ridge = ridge === 0 ? 1e-10 * trace / q : ridge * 100;
    }
    return new Float64Array(q);
  }

  function invert(A, q) {
    const n = q;
    const M = new Float64Array(n * 2 * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) M[i * 2 * n + j] = A[i * n + j];
      M[i * 2 * n + n + i] = 1;
    }
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r * 2 * n + c]) > Math.abs(M[piv * 2 * n + c])) piv = r;
      if (Math.abs(M[piv * 2 * n + c]) < 1e-14) return null;
      if (piv !== c) for (let j = 0; j < 2 * n; j++) { const t = M[c * 2 * n + j]; M[c * 2 * n + j] = M[piv * 2 * n + j]; M[piv * 2 * n + j] = t; }
      const d = M[c * 2 * n + c];
      for (let j = 0; j < 2 * n; j++) M[c * 2 * n + j] /= d;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r * 2 * n + c];
        if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) M[r * 2 * n + j] -= f * M[c * 2 * n + j];
      }
    }
    const inv = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) inv[i * n + j] = M[i * 2 * n + n + j];
    return inv;
  }

  // ------------------------------------------------------------------
  // Utilities
  // ------------------------------------------------------------------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffled(n, seed) {
    const idx = Array.from({ length: n }, (_, i) => i);
    const rnd = mulberry32(seed);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx;
  }

  const tick = () => new Promise(r => setTimeout(r, 0));
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const median = (a) => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

  // ------------------------------------------------------------------
  // Prepare: choose predictors, encode categoricals, drop incomplete rows
  // ------------------------------------------------------------------
  function numericColumns(table) {
    return table.columns.filter(c => c.numeric).map(c => c.name);
  }

  function prepare(table, targetName) {
    const target = table.columns.find(c => c.name === targetName);
    if (!target) throw new Error(`Column "${targetName}" was not found.`);
    if (!target.numeric) throw new Error(`The target column "${targetName}" is not numeric.`);

    const notices = [];
    const feats = [];

    table.columns.forEach(c => {
      if (c === target) return;
      if (/^(id|index|unnamed.*|row_?id|row_?names?)$/i.test(c.name)) {
        notices.push(`Skipped "${c.name}" because it looks like an identifier.`);
        return;
      }
      if (c.numeric) {
        const uniq = new Set();
        for (let i = 0; i < table.nRows; i++) if (!isNaN(c.nums[i])) uniq.add(c.nums[i]);
        if (uniq.size <= 1) { notices.push(`Skipped "${c.name}" because it is constant.`); return; }
        feats.push({ name: c.name, source: c.name, kind: 'numeric', values: c.nums });
      } else {
        const levels = [...new Set(c.values.filter(v => !MISSING.test(v)))].sort();
        if (levels.length <= 1) { notices.push(`Skipped "${c.name}" because it has a single value.`); return; }
        if (levels.length > MAX_LEVELS) {
          notices.push(`Skipped "${c.name}" because it has ${levels.length} distinct text values (the limit is ${MAX_LEVELS}).`);
          return;
        }
        const ref = levels[0];
        levels.slice(1).forEach(level => {
          const values = new Float64Array(table.nRows);
          for (let i = 0; i < table.nRows; i++) values[i] = MISSING.test(c.values[i]) ? NaN : (c.values[i] === level ? 1 : 0);
          feats.push({ name: `${c.name}_${level}`, source: c.name, kind: 'dummy', level, ref, values });
        });
      }
    });

    if (!feats.length) throw new Error('No usable predictor columns were found.');

    const keep = [];
    for (let i = 0; i < table.nRows; i++) {
      if (!isFinite(target.nums[i])) continue;
      let ok = true;
      for (const f of feats) if (!isFinite(f.values[i])) { ok = false; break; }
      if (ok) keep.push(i);
    }
    const dropped = table.nRows - keep.length;
    if (dropped > 0) notices.push(`Dropped ${dropped} row${dropped > 1 ? 's' : ''} with missing values.`);
    if (keep.length < MIN_ROWS) throw new Error(`Only ${keep.length} complete rows are left; at least ${MIN_ROWS} are needed.`);

    return {
      target: targetName,
      rowIds: keep,
      y: Float64Array.from(keep, i => target.nums[i]),
      feats: feats.map(f => ({ ...f, values: Float64Array.from(keep, i => f.values[i]) })),
      notices
    };
  }

  // ------------------------------------------------------------------
  // Analyse
  // ------------------------------------------------------------------
  async function analyze(prep, opts = {}) {
    const onProgress = opts.onProgress || (() => {});
    const m = prep.y.length;

    // ---- split
    let trainIdx, testIdx;
    if (opts.split) { trainIdx = opts.split.train; testIdx = opts.split.test; }
    else {
      const perm = shuffled(m, SEED);
      const nTest = Math.max(4, Math.ceil(0.2 * m));
      testIdx = perm.slice(0, nTest);
      trainIdx = perm.slice(nTest);
    }
    const ntr = trainIdx.length, nte = testIdx.length;

    // ---- drop predictors that are constant in the training rows
    let feats = prep.feats.filter(f => {
      let mn = Infinity, mx = -Infinity;
      for (const i of trainIdx) { if (f.values[i] < mn) mn = f.values[i]; if (f.values[i] > mx) mx = f.values[i]; }
      return mx > mn;
    });
    const notices = [...prep.notices];
    if (feats.length < prep.feats.length) notices.push('Some predictors were constant in the training rows and were removed.');
    if (!feats.length) throw new Error('No usable predictor columns were found.');
    if (feats.length > MAX_FEATURES) {
      throw new Error(`This file has ${feats.length} predictors after encoding; the limit is ${MAX_FEATURES}. Remove some columns and try again.`);
    }
    const p = feats.length;
    const kMax = Math.min(p, ntr - 6);
    if (kMax < 1) throw new Error('There are too few rows for the number of predictors.');
    if (kMax < p) notices.push(`Models are limited to ${kMax} predictors because the training set has only ${ntr} rows.`);

    const names = feats.map(f => f.name);
    const P = p + 1;

    // ---- train statistics and standardised design
    const xbar = new Float64Array(p), sd = new Float64Array(p);
    feats.forEach((f, j) => {
      let s = 0; for (const i of trainIdx) s += f.values[i];
      xbar[j] = s / ntr;
      let v = 0; for (const i of trainIdx) v += (f.values[i] - xbar[j]) ** 2;
      sd[j] = Math.sqrt(v / ntr);
    });
    let ybar = 0; for (const i of trainIdx) ybar += prep.y[i];
    ybar /= ntr;

    // ---- folds and Gram matrices of [1, Z] (Z = standardised predictors), y centred
    const K = Math.min(5, Math.max(2, Math.floor(ntr / 4)));
    const order = shuffled(ntr, SEED + 1);
    const foldOf = new Int32Array(ntr);
    order.forEach((pos, r) => { foldOf[pos] = r % K; });

    const Gf = Array.from({ length: K }, () => new Float64Array(P * P));
    const XYf = Array.from({ length: K }, () => new Float64Array(P));
    const YYf = new Float64Array(K), Nf = new Float64Array(K);
    const v = new Float64Array(P);
    v[0] = 1;
    const Z = feats.map(() => new Float64Array(ntr));
    const yc = new Float64Array(ntr);
    for (let t = 0; t < ntr; t++) {
      const i = trainIdx[t];
      for (let j = 0; j < p; j++) { Z[j][t] = (feats[j].values[i] - xbar[j]) / sd[j]; v[j + 1] = Z[j][t]; }
      yc[t] = prep.y[i] - ybar;
      const f = foldOf[t], G = Gf[f], XY = XYf[f];
      for (let a = 0; a < P; a++) {
        const va = v[a];
        XY[a] += va * yc[t];
        for (let b = 0; b <= a; b++) G[a * P + b] += va * v[b];
      }
      YYf[f] += yc[t] * yc[t];
      Nf[f]++;
    }
    const Gt = new Float64Array(P * P), XYt = new Float64Array(P);
    let YYt = 0;
    for (let f = 0; f < K; f++) {
      for (let a = 0; a < P; a++) {
        XYt[a] += XYf[f][a];
        for (let b = 0; b <= a; b++) Gt[a * P + b] += Gf[f][a * P + b];
      }
      YYt += YYf[f];
    }
    for (let f = 0; f < K; f++) for (let a = 0; a < P; a++) for (let b = 0; b < a; b++) Gf[f][b * P + a] = Gf[f][a * P + b];
    for (let a = 0; a < P; a++) for (let b = 0; b < a; b++) Gt[b * P + a] = Gt[a * P + b];

    // ---- cross-validated RMSE for a feature subset (used by Forward SFS and Exhaustive)
    function cvRmse(sub) {
      const idx = [0, ...sub.map(j => j + 1)];
      const q = idx.length;
      let total = 0;
      const A = new Float64Array(q * q), b = new Float64Array(q);
      for (let f = 0; f < K; f++) {
        const G = Gf[f], XY = XYf[f];
        for (let a = 0; a < q; a++) {
          b[a] = XYt[idx[a]] - XY[idx[a]];
          for (let c = 0; c <= a; c++) {
            const val = Gt[idx[a] * P + idx[c]] - G[idx[a] * P + idx[c]];
            A[a * q + c] = val; A[c * q + a] = val;
          }
        }
        const beta = solveSPD(A, b, q);
        let sse = YYf[f];
        for (let a = 0; a < q; a++) {
          sse -= 2 * beta[a] * XY[idx[a]];
          for (let c = 0; c < q; c++) sse += beta[a] * beta[c] * G[idx[a] * P + idx[c]];
        }
        total += Math.sqrt(Math.max(sse, 0) / Nf[f]);
      }
      return total / K;
    }

    // ---- coordinate descent (elastic net) on the correlation matrix
    const Cz = new Float64Array(p * p), rz = new Float64Array(p);
    for (let a = 0; a < p; a++) {
      rz[a] = XYt[a + 1] / ntr;
      for (let b = 0; b < p; b++) Cz[a * p + b] = Gt[(a + 1) * P + (b + 1)] / ntr;
    }

    function elasticPath(C, r, rho, alphas, ntrain) {
      const w = new Float64Array(p), out = [];
      for (const a of alphas) {
        const l1 = a * rho, l2 = a * (1 - rho);
        for (let it = 0; it < 500; it++) {
          let maxd = 0;
          for (let j = 0; j < p; j++) {
            let s = r[j];
            for (let k = 0; k < p; k++) if (k !== j) s -= C[j * p + k] * w[k];
            const wj = (Math.abs(s) <= l1 ? 0 : s - Math.sign(s) * l1) / (C[j * p + j] + l2);
            const d = Math.abs(wj - w[j]);
            if (d > maxd) maxd = d;
            w[j] = wj;
          }
          if (maxd < 1e-7) break;
        }
        out.push(Float64Array.from(w));
      }
      return out;
    }

    const alphaGrid = (r, rho, n) => {
      let amax = 0;
      for (let j = 0; j < p; j++) amax = Math.max(amax, Math.abs(r[j]));
      amax = amax / Math.max(rho, 1e-3) || 1;
      return Array.from({ length: n }, (_, i) => amax * Math.pow(10, -3 * i / (n - 1)));
    };

    function entryOrder(path) {
      const entry = new Array(p).fill(path.length);
      for (let i = path.length - 1; i >= 0; i--) for (let j = 0; j < p; j++) if (Math.abs(path[i][j]) > 1e-10) entry[j] = i;
      return entry;
    }

    // ---- ranking methods
    function lassoRanking() {
      const path = elasticPath(Cz, rz, 1, alphaGrid(rz, 1, 100));
      const entry = entryOrder(path), last = path[path.length - 1];
      return [...Array(p).keys()].sort((a, b) => entry[a] - entry[b] || Math.abs(last[b]) - Math.abs(last[a]) || Math.abs(rz[b]) - Math.abs(rz[a]));
    }

    function elasticNetRanking() {
      let best = { mse: Infinity, rho: 0.5, ai: 0 };
      const NA = 50;
      for (const rho of [0.1, 0.5, 0.9]) {
        const alphas = alphaGrid(rz, rho, NA);
        const mse = new Float64Array(NA);
        for (let f = 0; f < K; f++) {
          const nf = ntr - Nf[f];
          const C = new Float64Array(p * p), r = new Float64Array(p);
          for (let a = 0; a < p; a++) {
            r[a] = (XYt[a + 1] - XYf[f][a + 1]) / nf;
            for (let b = 0; b < p; b++) C[a * p + b] = (Gt[(a + 1) * P + (b + 1)] - Gf[f][(a + 1) * P + (b + 1)]) / nf;
          }
          const path = elasticPath(C, r, rho, alphas);
          path.forEach((w, ai) => {
            let sse = YYf[f];
            for (let a = 0; a < p; a++) {
              sse -= 2 * w[a] * XYf[f][a + 1];
              for (let b = 0; b < p; b++) sse += w[a] * w[b] * Gf[f][(a + 1) * P + (b + 1)];
            }
            mse[ai] += Math.max(sse, 0) / Nf[f] / K;
          });
        }
        mse.forEach((val, ai) => { if (val < best.mse) best = { mse: val, rho, ai }; });
      }
      const path = elasticPath(Cz, rz, best.rho, alphaGrid(rz, best.rho, 50));
      const w = path[best.ai], entry = entryOrder(path);
      return [...Array(p).keys()].sort((a, b) => Math.abs(w[b]) - Math.abs(w[a]) || entry[a] - entry[b]);
    }

    function forwardRanking() {
      const chosen = [], remaining = [...Array(p).keys()];
      while (remaining.length && chosen.length < kMax) {
        let bestJ = remaining[0], bestV = Infinity;
        for (const j of remaining) {
          const val = cvRmse([...chosen, j].sort((a, b) => a - b));
          if (val < bestV) { bestV = val; bestJ = j; }
        }
        chosen.push(bestJ);
        remaining.splice(remaining.indexOf(bestJ), 1);
      }
      return chosen;
    }

    function rfeRanking() {
      let remaining = [...Array(p).keys()];
      const removed = [];
      while (remaining.length > 1) {
        const idx = [0, ...remaining.map(j => j + 1)], q = idx.length;
        const A = new Float64Array(q * q), b = new Float64Array(q);
        for (let a = 0; a < q; a++) { b[a] = XYt[idx[a]]; for (let c = 0; c < q; c++) A[a * q + c] = Gt[idx[a] * P + idx[c]]; }
        const beta = solveSPD(A, b, q);
        let worst = 0;
        for (let a = 1; a < q; a++) if (Math.abs(beta[a]) < Math.abs(beta[worst + 1])) worst = a - 1;
        removed.push(remaining[worst]);
        remaining = remaining.filter((_, i) => i !== worst);
      }
      return [...remaining, ...removed.reverse()];
    }

    const nested = (ranking) => Array.from({ length: kMax }, (_, k) => ranking.slice(0, k + 1));

    // ---- evaluate one subset: statistics on the training set, metrics on the test set
    function evaluate(sub) {
      const subset = [...sub];
      const idx = [0, ...subset.map(j => j + 1)], q = idx.length, k = subset.length;
      const A = new Float64Array(q * q), b = new Float64Array(q);
      for (let a = 0; a < q; a++) { b[a] = XYt[idx[a]]; for (let c = 0; c < q; c++) A[a * q + c] = Gt[idx[a] * P + idx[c]]; }
      const beta = solveSPD(A, b, q);

      let sse = 0;
      for (let t = 0; t < ntr; t++) {
        let pred = beta[0];
        for (let a = 1; a < q; a++) pred += beta[a] * Z[idx[a] - 1][t];
        sse += (yc[t] - pred) ** 2;
      }
      const dof = ntr - q;
      const sigma2 = sse / dof;
      const inv = invert(A, q);
      const cov = (i, j) => (inv ? sigma2 * inv[i * q + j] : NaN);

      const coefOrig = subset.map((j, a) => beta[a + 1] / sd[j]);
      const intercept = ybar + beta[0] - subset.reduce((s, j, a) => s + beta[a + 1] * xbar[j] / sd[j], 0);

      const avec = [1, ...subset.map(j => -xbar[j] / sd[j])];
      let varInt = 0;
      for (let a = 0; a < q; a++) for (let c = 0; c < q; c++) varInt += avec[a] * avec[c] * cov(a, c);

      const tcrit = tCritical(dof);
      const mkRow = (feature, est, se, vif) => {
        const t = est / se;
        const pv = isFinite(t) ? tTwoSided(Math.abs(t), dof) : NaN;
        return { feature, coef: est, stderr: se, tstat: t, pvalue: pv, ci_lower: est - tcrit * se, ci_upper: est + tcrit * se, vif };
      };

      // VIF from the inverse correlation matrix of the chosen predictors
      let vifs = subset.map(() => 1);
      if (k > 1) {
        const R = new Float64Array(k * k);
        subset.forEach((ja, a) => subset.forEach((jb, c) => { R[a * k + c] = Cz[ja * p + jb]; }));
        const Ri = invert(R, k);
        vifs = subset.map((_, a) => (Ri ? Ri[a * k + a] : Infinity));
      }

      const coefficients = [mkRow('Intercept (β₀)', intercept, Math.sqrt(Math.max(varInt, 0)), null)];
      subset.forEach((j, a) => coefficients.push(mkRow(names[j], coefOrig[a], Math.sqrt(Math.max(cov(a + 1, a + 1), 0)) / sd[j], vifs[a])));

      const ssr = YYt - sse;
      const fStat = (ssr / k) / sigma2;
      const fP = fUpper(fStat, k, dof);

      // test-set metrics
      const preds = new Float64Array(nte), actual = new Float64Array(nte);
      let se2 = 0, ae = 0, ys = 0;
      testIdx.forEach((i, r) => {
        let pr = intercept;
        subset.forEach((j, a) => { pr += coefOrig[a] * feats[j].values[i]; });
        preds[r] = pr; actual[r] = prep.y[i];
        const e = actual[r] - pr;
        se2 += e * e; ae += Math.abs(e); ys += actual[r];
      });
      const ymu = ys / nte;
      let sst = 0; for (let r = 0; r < nte; r++) sst += (actual[r] - ymu) ** 2;
      const r2 = 1 - se2 / sst;
      const adj = nte - k - 1 > 0 ? 1 - (1 - r2) * (nte - 1) / (nte - k - 1) : null;

      const outliers = [...Array(nte).keys()]
        .sort((a, b) => Math.abs(actual[b] - preds[b]) - Math.abs(actual[a] - preds[a]))
        .slice(0, 10)
        .map((r, rank) => {
          const res = actual[r] - preds[r];
          return { rank: rank + 1, index: prep.rowIds[testIdx[r]], actual: actual[r], predicted: preds[r], residual: res, abs_error: Math.abs(res), pct_error: actual[r] !== 0 ? Math.abs(res / actual[r]) * 100 : 0 };
        });

      return {
        k, features: subset.map(j => names[j]), rmse: Math.sqrt(se2 / nte), mae: ae / nte, r2, adj_r2: adj,
        f_stat: fStat, f_pvalue: fP, coefficients, outliers
      };
    }

    // ---- run everything
    const models = {};
    onProgress('Ranking predictors with Lasso…'); await tick();
    models['Lasso'] = nested(lassoRanking()).map(evaluate);
    onProgress('Running forward selection…'); await tick();
    models['Forward SFS'] = nested(forwardRanking()).map(evaluate);
    onProgress('Running recursive feature elimination…'); await tick();
    models['RFE'] = nested(rfeRanking()).map(evaluate);
    onProgress('Fitting ElasticNet…'); await tick();
    models['ElasticNet'] = nested(elasticNetRanking()).map(evaluate);

    if (p <= EXHAUSTIVE_MAX) {
      onProgress('Searching every subset…'); await tick();
      const bestByK = new Array(p + 1).fill(null);
      const total = (1 << p) - 1;
      for (let mask = 1; mask <= total; mask++) {
        let kk = 0; for (let j = 0; j < p; j++) if (mask & (1 << j)) kk++;
        if (kk > kMax) continue;
        const sub = []; for (let j = 0; j < p; j++) if (mask & (1 << j)) sub.push(j);
        const val = cvRmse(sub);
        if (!bestByK[kk] || val < bestByK[kk].val) bestByK[kk] = { val, sub };
        if (mask % 800 === 0) await tick();
      }
      models['Exhaustive'] = bestByK.slice(1, kMax + 1).map(b => evaluate(b.sub));
    } else {
      notices.push(`Exhaustive search was skipped: ${p} predictors means ${(2 ** p).toLocaleString('en-US')} subsets (the limit is ${EXHAUSTIVE_MAX} predictors).`);
    }

    // ---- overview
    const yAll = Array.from(prep.y);
    const ym = mean(yAll);
    let syy = 0; yAll.forEach(y => { syy += (y - ym) ** 2; });
    const overview = feats.map(f => {
      const xs = Array.from(f.values), xm = mean(xs);
      let sxx = 0, sxy = 0;
      xs.forEach((x, i) => { sxx += (x - xm) ** 2; sxy += (x - xm) * (yAll[i] - ym); });
      return {
        feature: f.name, source: f.source, kind: f.kind, level: f.level ?? null, ref: f.ref ?? null,
        desc: f.kind === 'numeric' ? 'Numeric column' : `1 if ${f.source} = ${f.level} (reference: ${f.ref})`,
        mean: xm, std: Math.sqrt(sxx / Math.max(xs.length - 1, 1)),
        min: Math.min(...xs), median: median(xs), max: Math.max(...xs),
        corr: sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0
      };
    });

    return {
      models, overview, notices,
      target: { name: prep.target, mean: ym, min: Math.min(...yAll), max: Math.max(...yAll) },
      meta: { n: m, train_n: ntr, test_n: nte, features: p, kMax, folds: K },
      _internals: { evaluate, cvRmse, names }
    };
  }

  root.RegEngine = { parseCSV, readTable, numericColumns, prepare, analyze, _stats: { betai, tTwoSided, fUpper, tCritical } };
})(typeof window !== 'undefined' ? window : globalThis);
