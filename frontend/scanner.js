// ── Состояние сканера ─────────────────────────────────────
const scanner = {
    tickers:  [],
    running:  false,
    autoTimer: null,
};

// ── Шаблоны формул для сканера ────────────────────────────
const SCAN_TEMPLATES = {
    oversold:    { name: 'Перепроданность',  formula: 'RESULT = RSI(14) < 30',                              },
    overbought:  { name: 'Перекупленность',  formula: 'RESULT = RSI(14) > 70',                              },
    ema_cross_up:{ name: 'EMA пересечение ▲',formula: 'RESULT = CROSS_UP(EMA(9), EMA(21))',                 },
    vol_spike:   { name: 'Объём x2',         formula: 'RESULT = VOLUME > SMA(VOLUME, 20) * 2',              },
    bb_squeeze:  { name: 'Сжатие BB',        formula: 'RESULT = (BB_UPPER(20,2) - BB_LOWER(20,2)) / SMA(20) * 100 < 2', },
    trend_up:    { name: 'Тренд вверх',      formula: 'RESULT = EMA(9) > EMA(21)',                          },
    trend_down:  { name: 'Тренд вниз',       formula: 'RESULT = EMA(9) < EMA(21)',                          },
    new_high:    { name: 'Новый максимум',    formula: 'RESULT = CLOSE >= MAX(HIGH, 20)',                    },
};

// ── Инициализация ─────────────────────────────────────────
async function initScanner() {
    // Загружаем список тикеров с сервера
    try {
        const r    = await fetch('/api/scanner/tickers');
        const data = await r.json();
        scanner.tickers = [...data.tickers];
        renderTickerList();
    } catch (e) {
        console.error('initScanner:', e);
    }

    // Рендерим шаблоны
    renderScanTemplates();
}

// ── Рендер списка тикеров ─────────────────────────────────
function renderTickerList() {
    const el = document.getElementById('scan-ticker-list');
    if (!el) return;

    el.innerHTML = scanner.tickers.map(t => `
        <span class="ticker-tag" id="tag-${t}">
            ${t}
            <span class="tag-remove" onclick="removeTicker('${t}')">✕</span>
        </span>
    `).join('');

    document.getElementById('scan-count').textContent =
        `${scanner.tickers.length} тикеров`;
}

function removeTicker(ticker) {
    scanner.tickers = scanner.tickers.filter(t => t !== ticker);
    renderTickerList();
}

function addTicker() {
    const input  = document.getElementById('scan-add-input');
    const ticker = input.value.trim().toUpperCase();
    if (!ticker) return;
    if (scanner.tickers.includes(ticker)) { input.value = ''; return; }
    scanner.tickers.push(ticker);
    input.value = '';
    renderTickerList();
}

// Enter в поле добавления
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('scan-add-input');
    if (input) {
        input.addEventListener('keydown', e => {
            if (e.key === 'Enter') addTicker();
        });
    }
    initScanner();
});

// ── Шаблоны сканера ───────────────────────────────────────
function renderScanTemplates() {
    const el = document.getElementById('scan-templates');
    if (!el) return;

    el.innerHTML = Object.entries(SCAN_TEMPLATES).map(([key, t]) => `
        <button class="btn btn-tpl" onclick="setScanTemplate('${key}')">${t.name}</button>
    `).join('');
}

function setScanTemplate(key) {
    const t = SCAN_TEMPLATES[key];
    document.getElementById('scan-formula').value = t.formula;
    document.getElementById('scan-formula-name').value = t.name;
}

// ── Запуск сканера ────────────────────────────────────────
async function runScanner() {
    if (scanner.running) return;
    if (!scanner.tickers.length) {
        setScanStatus('❌ Добавьте тикеры');
        return;
    }

    const formula  = document.getElementById('scan-formula').value.trim();
    const interval = document.getElementById('scan-interval').value;

    if (!formula) { setScanStatus('❌ Введите формулу'); return; }

    scanner.running = true;
    document.getElementById('btn-scan').disabled = true;
    document.getElementById('btn-scan').textContent = '⏳ Сканирование...';
    setScanStatus(`⏳ Проверяем ${scanner.tickers.length} тикеров...`);

    const t0 = Date.now();

    try {
        const r = await fetch('/api/scanner', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tickers:  scanner.tickers,
                formula:  formula,
                interval: interval,
                params:   {}
            })
        });

        const data = await r.json();

        if (!r.ok) { setScanStatus(`❌ ${data.detail}`); return; }

        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
        renderScanResults(data.results);
        setScanStatus(
            `✅ Готово за ${elapsed}с | ` +
            `Сигналов: ${data.signals} из ${data.total}`
        );

    } catch (e) {
        setScanStatus(`❌ ${e.message}`);
    } finally {
        scanner.running = false;
        document.getElementById('btn-scan').disabled = false;
        document.getElementById('btn-scan').textContent = '▶ Сканировать';
    }
}

// ── Рендер результатов ────────────────────────────────────
function renderScanResults(results) {
    const el = document.getElementById('scan-results');
    if (!el) return;

    if (!results.length) {
        el.innerHTML = '<div class="muted" style="padding:12px">Нет результатов</div>';
        return;
    }

    // Сортируем: сначала сигналы, потом по изменению цены
    const sorted = [...results].sort((a, b) => {
        if (a.signal !== b.signal) return b.signal - a.signal;
        return Math.abs(b.change) - Math.abs(a.change);
    });

    el.innerHTML = sorted.map(r => {
        if (r.error) return `
            <div class="scan-row error-row">
                <span class="scan-ticker">${r.ticker}</span>
                <span class="scan-error">${r.error}</span>
            </div>
        `;

        const signalClass = r.signal ? 'signal-yes' : 'signal-no';
        const signalIcon  = r.signal ? '🔴' : '⚪';
        const changeClass = r.change > 0 ? 'change-up' : r.change < 0 ? 'change-down' : '';
        const changeSign  = r.change > 0 ? '+' : '';

        return `
            <div class="scan-row ${signalClass}" onclick="openTickerChart('${r.ticker}')">
                <span class="scan-signal">${signalIcon}</span>
                <span class="scan-ticker">${r.ticker}</span>
                <span class="scan-price">${r.price} ₽</span>
                <span class="scan-change ${changeClass}">${changeSign}${r.change}%</span>
                <span class="scan-value">= ${r.value}</span>
            </div>
        `;
    }).join('');
}

// Клик по тикеру — открываем его график
function openTickerChart(ticker) {
    // Переключаемся на вкладку графика
    switchTab('chart');

    // Загружаем тикер
    document.getElementById('ticker').value = ticker;
    loadChart();
}

// ── Авто-обновление ───────────────────────────────────────
function toggleAutoScan() {
    const btn = document.getElementById('btn-auto-scan');

    if (scanner.autoTimer) {
        clearInterval(scanner.autoTimer);
        scanner.autoTimer = null;
        btn.textContent = '🔄 Авто';
        btn.classList.remove('btn-active');
        setScanStatus('⏸ Авто-сканирование остановлено');
    } else {
        runScanner();
        scanner.autoTimer = setInterval(runScanner, 60_000); // каждую минуту
        btn.textContent = '⏹ Стоп';
        btn.classList.add('btn-active');
        setScanStatus('🔄 Авто-сканирование каждые 60с');
    }
}

function setScanStatus(msg) {
    const el = document.getElementById('scan-status');
    if (el) el.textContent = msg;
}
