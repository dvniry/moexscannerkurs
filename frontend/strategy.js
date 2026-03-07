// ── Состояние стратегий ───────────────────────────────────
const strategyState = {
    signals:     [],
    inPosition:  false,
    markers:     [],    // LightweightCharts markers
};

// ── Шаблоны стратегий ─────────────────────────────────────
const STRATEGY_TEMPLATES = {
    ema_cross: {
        name:  'EMA Cross',
        entry: 'RESULT = CROSS_UP(EMA(9), EMA(21))',
        exit:  'RESULT = CROSS_DOWN(EMA(9), EMA(21))',
        stop:  '',
        size:  10,
    },
    rsi_reversal: {
        name:  'RSI Reversal',
        entry: 'RESULT = CROSS_UP(RSI(14), 30)',
        exit:  'RESULT = CROSS_DOWN(RSI(14), 70)',
        stop:  'RESULT = RSI(14) < 20',
        size:  10,
    },
    trend_follow: {
        name:  'Trend Follow',
        entry: 'RESULT = EMA(9) > EMA(21) AND CROSS_UP(CLOSE, EMA(9))',
        exit:  'RESULT = CROSS_DOWN(CLOSE, EMA(21))',
        stop:  '',
        size:  15,
    },
    bb_bounce: {
        name:  'BB Bounce',
        entry: 'RESULT = CLOSE < BB_LOWER(20, 2)',
        exit:  'RESULT = CLOSE > BB_UPPER(20, 2)',
        stop:  '',
        size:  10,
    },
};

// ── Применить шаблон ──────────────────────────────────────
function setStrategyTemplate(key) {
    const t = STRATEGY_TEMPLATES[key];
    document.getElementById('strat-name').value  = t.name;
    document.getElementById('strat-entry').value = t.entry;
    document.getElementById('strat-exit').value  = t.exit;
    document.getElementById('strat-stop').value  = t.stop;
    document.getElementById('strat-size').value  = t.size;
}

// ── Инициализация вкладки ─────────────────────────────────
function initStrategy() {
    const el = document.getElementById('strat-templates');
    if (!el) return;
    el.innerHTML = Object.entries(STRATEGY_TEMPLATES).map(([key, t]) =>
        `<button class="btn btn-tpl" onclick="setStrategyTemplate('${key}')">${t.name}</button>`
    ).join('');

    refreshSandboxPortfolio();  // ← загружаем портфель при открытии
}


// ── Запуск стратегии ──────────────────────────────────────
async function runStrategy() {
    const ticker = document.getElementById('ticker').value.trim().toUpperCase();
    const name   = document.getElementById('strat-name').value.trim()  || 'My Strategy';
    const entry  = document.getElementById('strat-entry').value.trim();
    const exit_  = document.getElementById('strat-exit').value.trim();
    const stop   = document.getElementById('strat-stop').value.trim();
    const size   = parseFloat(document.getElementById('strat-size').value) || 10;
    const interval = document.getElementById('interval').value;

    if (!entry) { setStratStatus('❌ Введите Entry формулу'); return; }
    if (!exit_) { setStratStatus('❌ Введите Exit формулу');  return; }

    setStratStatus('⏳ Считаем сигналы...');
    document.getElementById('btn-run-strat').disabled = true;

    try {
        const r = await fetch('/api/strategy/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker,
                name,
                entry_formula: entry,
                exit_formula:  exit_,
                stop_formula:  stop || null,
                size,
                interval,
                params: {},
            })
        });

        const data = await r.json();

        if (data.error) {
            setStratStatus(`❌ ${data.error}`);
            return;
        }

        strategyState.signals    = data.signals;
        strategyState.inPosition = data.in_position;

        renderStrategySignals(data);
        drawSignalsOnChart(data.signals);

        const s = data.stats;
        setStratStatus(
            `✅ Сигналов: ${s.total_signals} | ` +
            `Сделок: ${s.total_trades} | ` +
            `Winrate: ${s.winrate}% | ` +
            `${data.in_position ? '🟢 В позиции' : '⚪ Вне позиции'}`
        );

    } catch (e) {
        setStratStatus(`❌ ${e.message}`);
    } finally {
        document.getElementById('btn-run-strat').disabled = false;
    }
}

// ── Рендер таблицы сигналов ───────────────────────────────
function renderStrategySignals(data) {
    const el = document.getElementById('strat-signals');
    if (!el) return;

    if (!data.signals.length) {
        el.innerHTML = '<div class="muted" style="padding:12px">Сигналов нет</div>';
        return;
    }

    const rows = [...data.signals].reverse().map(s => {
        const dt      = new Date(s.time * 1000).toLocaleString('ru');
        const icon    = s.action === 'BUY' ? '🟢' : s.action === 'SELL' ? '🔴' : '🛑';
        const cls     = s.action === 'BUY' ? 'signal-buy' : s.action === 'SELL' ? 'signal-sell' : 'signal-stop';
        return `
            <div class="strat-row ${cls}">
                <span class="strat-icon">${icon}</span>
                <span class="strat-action">${s.action}</span>
                <span class="strat-price">${s.price} ₽</span>
                <span class="strat-time">${dt}</span>
                <button class="btn btn-sm btn-sandbox"
                    onclick="sendSandboxOrder('${s.action}', ${s.price})">
                    📤 Sandbox
                </button>
            </div>
        `;
    }).join('');

    el.innerHTML = rows;
}

// ── Стрелки на графике (LightweightCharts markers) ────────
function drawSignalsOnChart(signals) {
    // было: if (!window.candleSeries)
    // state.candleSeries — так хранится в chart.js
    const series = window.state?.candleSeries || window.candleSeries;
    if (!series) {
        console.warn('candleSeries не найден');
        return;
    }

    series.setMarkers([]);

    const markers = signals.map(s => ({
        time:     s.time,
        position: s.action === 'BUY' ? 'belowBar' : 'aboveBar',
        color:    s.action === 'BUY'  ? '#4CAF50'
                : s.action === 'SELL' ? '#f44336' : '#FF9800',
        shape:    s.action === 'BUY'  ? 'arrowUp' : 'arrowDown',
        text:     s.action === 'BUY'  ? `▲ ${s.price}`
                : s.action === 'SELL' ? `▼ ${s.price}` : `✕ STOP`,
        size: 1,
    }));

    markers.sort((a, b) => a.time - b.time);
    series.setMarkers(markers);
}


// ── Очистить сигналы с графика ────────────────────────────
function clearStrategySignals() {
    if (window.candleSeries) candleSeries.setMarkers([]);
    strategyState.signals = [];
    const el = document.getElementById('strat-signals');
    if (el) el.innerHTML = '<div class="muted" style="padding:12px">Сигналов нет</div>';
    setStratStatus('Очищено');
}

// ── Отправить ордер в Sandbox ─────────────────────────────
async function sendSandboxOrder(direction, price) {
    if (!window._lastFigi) {
        setStratStatus('❌ Сначала загрузите график на вкладке График');
        return;
    }

    const btn = event?.target;
    if (btn) { btn.disabled = true; btn.textContent = '⏳'; }

    try {
        const r = await fetch('/api/strategy/order', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                figi:      window._lastFigi,
                direction: direction,
                lots:      1,
                price:     0,   // market order
            })
        });

        const data = await r.json();

        if (!r.ok) {
            setStratStatus(`❌ ${data.detail}`);
            return;
        }

        // Показываем результат
        setStratStatus(
            `✅ ${direction} исполнен | ` +
            `Цена: ${data.price} ₽ | ` +
            `Лотов: ${data.lots_exec} | ` +
            `ID: ${data.order_id.slice(0, 8)}...`
        );

        // Подгружаем портфель
        await refreshSandboxPortfolio();

    } catch (e) {
        setStratStatus(`❌ ${e.message}`);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '📤 Sandbox'; }
    }
}

// ── Портфель sandbox ──────────────────────────────────────
async function refreshSandboxPortfolio() {
    try {
        const r    = await fetch('/api/strategy/sandbox/portfolio');
        const data = await r.json();

        const el = document.getElementById('sandbox-portfolio');
        if (!el) return;

        if (!data.positions?.length) {
            el.innerHTML = `
                <div class="portfolio-balance">
                    💰 Баланс: <b>${data.total.toLocaleString('ru')} ₽</b>
                </div>
                <div class="muted" style="padding:8px">Позиций нет</div>
            `;
            return;
        }

        const rows = data.positions
            .filter(p => p.figi !== 'RUB000UTSTOM')  // скрываем рубли
            .map(p => `
                <div class="portfolio-row">
                    <span class="p-figi">${p.figi}</span>
                    <span class="p-qty">${p.quantity} лот</span>
                    <span class="p-price">${p.avg_price.toFixed(2)} ₽</span>
                </div>
            `).join('');

        el.innerHTML = `
            <div class="portfolio-balance">
                💰 <b>${data.total.toLocaleString('ru')} ₽</b>
            </div>
            ${rows || '<div class="muted" style="padding:8px">Только рубли</div>'}
        `;

    } catch (e) {
        console.warn('portfolio error:', e);
    }
}

// ── Статус стратегии ──────────────────────────────────────
function setStratStatus(msg) {
    const el = document.getElementById('strat-status');
    if (el) el.textContent = msg;
    console.log('[Strategy]', msg);
}
