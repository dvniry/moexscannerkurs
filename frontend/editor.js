// ── Шаблоны ───────────────────────────────────────────────
const TEMPLATES = {
    ema_diff:    { name: 'EMA Diff',      pane: 'sub',  formula: 'RESULT = EMA(9) - EMA(21)' },
    rsi:         { name: 'RSI Zone',      pane: 'sub',  formula: 'RESULT = IF(RSI(14) > 70, 1, IF(RSI(14) < 30, -1, 0))' },
    typical:     { name: 'Typical Price', pane: 'main', formula: 'RESULT = (HIGH + LOW + CLOSE) / 3' },
    volatility:  { name: 'Volatility %',  pane: 'sub',  formula: 'RANGE = HIGH - LOW\nAVG = SMA(CLOSE, 20)\nRESULT = RANGE / AVG * 100' },
    vol_pressure:{ name: 'Vol Pressure',  pane: 'sub',  formula: 'HL = HIGH - LOW\nRESULT = (CLOSE - LOW) / HL * VOLUME' },
    cross:       { name: 'EMA Cross',     pane: 'sub',  formula: 'RESULT = CROSS_UP(EMA(9), EMA(21)) - CROSS_DOWN(EMA(9), EMA(21))' },
    bb:          { name: 'BB Width',      pane: 'sub',  formula: 'RESULT = (BB_UPPER(20, 2) - BB_LOWER(20, 2)) / SMA(20) * 100' },
};

function tpl(key) {
    const t = TEMPLATES[key];
    document.getElementById('f-name').value  = t.name;
    document.getElementById('f-code').value  = t.formula;
    document.getElementById('f-pane').value  = t.pane;
}

// ── Применить формулу ─────────────────────────────────────
async function applyFormula() {
    const name    = document.getElementById('f-name').value.trim() || 'Custom';
    const formula = document.getElementById('f-code').value.trim();
    const pane    = document.getElementById('f-pane').value;

    const errEl = document.getElementById('f-error');
    const resEl = document.getElementById('f-result');
    errEl.style.display = 'none';
    resEl.style.display = 'none';

    if (!formula) { showErr('Введите формулу'); return; }

    setStatus(`⏳ Расчёт: ${name}...`);

    try {
        const r = await fetch('/api/formula', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: state.ticker, interval: state.interval,
                days: Number(state.days), formula, name, params: {}
            })
        });

        const data = await r.json();

        if (data.error) { showErr(data.error); return; }

        const color  = nextColor();
        const series = addLine(name, data.points, pane, color);

        // Сохраняем для удаления
        if (pane === 'sub') state.subLines.push(series);
        else                state.mainLines.push(series);

        addToList(name, color, pane, series);

        resEl.textContent = `✅ ${name} = ${data.last}`;
        resEl.style.display = 'block';
        setStatus(`✅ ${name}: ${data.points.length} точек, последнее = ${data.last}`);

    } catch(e) { showErr(`Ошибка: ${e.message}`); }
}

function showErr(msg) {
    const el = document.getElementById('f-error');
    el.textContent = msg;
    el.style.display = 'block';
    setStatus(`❌ ${msg.slice(0, 80)}`);
}

// ── Список индикаторов ────────────────────────────────────
function addToList(name, color, pane, series) {
    const list = document.getElementById('ind-list');
    const ph   = list.querySelector('.muted');
    if (ph) ph.remove();

    const item = document.createElement('div');
    item.className = 'ind-item';
    item.innerHTML = `
      <span>
        <span class="ind-dot" style="background:${color}"></span>
        ${name}
        <span class="muted">(${pane})</span>
      </span>
      <span class="ind-remove" title="Удалить">✕</span>
    `;

    item.querySelector('.ind-remove').onclick = () => {
        const chart = pane === 'sub' ? subChart : mainChart;
        try { chart.removeSeries(series); } catch(_) {}
        item.remove();
        if (!list.children.length)
            list.innerHTML = '<span class="muted">Пока нет</span>';
    };

    list.appendChild(item);
}
