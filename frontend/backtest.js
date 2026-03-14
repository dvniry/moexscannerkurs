// ── Состояние бэктеста ────────────────────────────────────
const btState = {
    result:     null,
    equityChart: null,
};

// ── Шаблоны (те же что в стратегиях) ─────────────────────
const BT_TEMPLATES = {
    ema_cross:    { name: 'EMA Cross',    entry: 'RESULT = CROSS_UP(EMA(9), EMA(21))',    exit: 'RESULT = CROSS_DOWN(EMA(9), EMA(21))',  stop: '' },
    rsi_reversal: { name: 'RSI Reversal', entry: 'RESULT = CROSS_UP(RSI(14), 30)',        exit: 'RESULT = CROSS_DOWN(RSI(14), 70)',      stop: 'RESULT = RSI(14) < 20' },
    trend_follow: { name: 'Trend Follow', entry: 'RESULT = EMA(9) > EMA(21) AND CROSS_UP(CLOSE, EMA(9))', exit: 'RESULT = CROSS_DOWN(CLOSE, EMA(21))', stop: '' },
    bb_bounce:    { name: 'BB Bounce',    entry: 'RESULT = CLOSE < BB_LOWER(20, 2)',      exit: 'RESULT = CLOSE > BB_UPPER(20, 2)',      stop: '' },
};

function setBtTemplate(key) {
    const t = BT_TEMPLATES[key];
    document.getElementById('bt-name').value  = t.name;
    document.getElementById('bt-entry').value = t.entry;
    document.getElementById('bt-exit').value  = t.exit;
    document.getElementById('bt-stop').value  = t.stop;
}

function initBacktest() {
    const el = document.getElementById('bt-templates');
    if (!el) return;
    el.innerHTML = Object.entries(BT_TEMPLATES).map(([key, t]) =>
        `<button class="btn btn-tpl" onclick="setBtTemplate('${key}')">${t.name}</button>`
    ).join('');
}

// ── Запуск бэктеста ───────────────────────────────────────
async function runBacktest() {
    const ticker   = document.getElementById('ticker').value.trim().toUpperCase();
    const name     = document.getElementById('bt-name').value.trim()   || 'Backtest';
    const entry    = document.getElementById('bt-entry').value.trim();
    const exit_    = document.getElementById('bt-exit').value.trim();
    const stop     = document.getElementById('bt-stop').value.trim();
    const capital  = parseFloat(document.getElementById('bt-capital').value)  || 100000;
    const interval = document.getElementById('bt-interval').value;
    const days     = parseInt(document.getElementById('bt-days').value) || 365;

    if (!entry) { setBtStatus('❌ Введите Entry формулу'); return; }
    if (!exit_) { setBtStatus('❌ Введите Exit формулу');  return; }

    setBtStatus(`⏳ Тестируем ${ticker} за ${days} дней...`);
    document.getElementById('btn-run-bt').disabled = true;

    try {
        const r = await fetch('/api/backtest', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker,
                name,
                entry_formula: entry,
                exit_formula:  exit_,
                stop_formula:  stop || null,
                size:          parseFloat(document.getElementById('bt-size').value) || 10,
                interval,
                days,
                capital,
                params: {},
            })
        });

        const data = await r.json();

        if (!r.ok) {
            console.error('[Backtest] Server error:', data);   // ← увидишь detail
            setBtStatus(`❌ ${data.detail || 'Ошибка сервера ' + r.status}`);
            return;
        }

        if (data.error) { setBtStatus(`❌ ${data.error}`); return; }

        btState.result = data;

        renderBtStats(data);
        renderBtTrades(data.trades);
        renderEquityChart(data.equity_curve, data.initial);

        const sign = data.total_return >= 0 ? '+' : '';
        setBtStatus(
            `✅ ${data.total_trades} сделок | ` +
            `${sign}${data.total_return}% | ` +
            `Winrate: ${data.winrate}% | ` +
            `Sharpe: ${data.sharpe} | ` +
            `Просадка: -${data.max_drawdown}%`
        );

    } catch (e) {
        setBtStatus(`❌ ${e.message}`);
    } finally {
        document.getElementById('btn-run-bt').disabled = false;
    }
}

// ── Метрики ───────────────────────────────────────────────
function renderBtStats(d) {
    const el = document.getElementById('bt-stats');
    if (!el) return;

    const returnColor = d.total_return >= 0 ? 'var(--green)' : 'var(--red)';
    const sign        = d.total_return >= 0 ? '+' : '';

    el.innerHTML = `
        <div class="bt-metrics">
            <div class="bt-metric">
                <div class="bt-metric-val" style="color:${returnColor}">${sign}${d.total_return}%</div>
                <div class="bt-metric-lbl">Доходность</div>
            </div>
            <div class="bt-metric">
                <div class="bt-metric-val">${d.final.toLocaleString('ru')} ₽</div>
                <div class="bt-metric-lbl">Итоговый капитал</div>
            </div>
            <div class="bt-metric">
                <div class="bt-metric-val" style="color:var(--green)">${d.winrate}%</div>
                <div class="bt-metric-lbl">Winrate</div>
            </div>
            <div class="bt-metric">
                <div class="bt-metric-val">${d.total_trades}</div>
                <div class="bt-metric-lbl">Сделок</div>
            </div>
            <div class="bt-metric">
                <div class="bt-metric-val" style="color:var(--red)">-${d.max_drawdown}%</div>
                <div class="bt-metric-lbl">Макс просадка</div>
            </div>
            <div class="bt-metric">
                <div class="bt-metric-val">${d.sharpe}</div>
                <div class="bt-metric-lbl">Sharpe</div>
            </div>
            <div class="bt-metric">
                <div class="bt-metric-val" style="color:var(--green)">+${d.avg_profit}%</div>
                <div class="bt-metric-lbl">Avg прибыль</div>
            </div>
            <div class="bt-metric">
                <div class="bt-metric-val" style="color:var(--red)">${d.avg_loss}%</div>
                <div class="bt-metric-lbl">Avg убыток</div>
            </div>
            <div class="bt-metric">
                <div class="bt-metric-val" style="color:var(--green)">+${d.best_trade}%</div>
                <div class="bt-metric-lbl">Лучшая сделка</div>
            </div>
            <div class="bt-metric">
                <div class="bt-metric-val" style="color:var(--red)">${d.worst_trade}%</div>
                <div class="bt-metric-lbl">Худшая сделка</div>
            </div>
            <div class="bt-metric">
                <div class="bt-metric-val">${d.wins} / ${d.losses}</div>
                <div class="bt-metric-lbl">Прибыльных / убыточных</div>
            </div>
            <div class="bt-metric">
                <div class="bt-metric-val">${d.commission.toLocaleString('ru')} ₽</div>
                <div class="bt-metric-lbl">Комиссия (0.05%)</div>
            </div>
        </div>
    `;
}

// ── Таблица сделок ────────────────────────────────────────
function renderBtTrades(trades) {
    const el = document.getElementById('bt-trades');
    if (!el) return;

    if (!trades.length) {
        el.innerHTML = '<div class="muted" style="padding:12px">Сделок нет</div>';
        return;
    }

    const header = `
        <div class="bt-trade-row bt-trade-header">
            <span>Вход</span>
            <span>Выход</span>
            <span>Цена входа</span>
            <span>Цена выхода</span>
            <span>P&L ₽</span>
            <span>P&L %</span>
            <span>Тип</span>
        </div>
    `;

    const rows = [...trades].reverse().map(t => {
        const entryDt  = new Date(t.entry_time * 1000).toLocaleString('ru');
        const exitDt   = new Date(t.exit_time  * 1000).toLocaleString('ru');
        const pnlColor = t.pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const sign     = t.pnl >= 0 ? '+' : '';
        const icon     = t.action === 'STOP' ? '🛑' : t.pnl >= 0 ? '✅' : '❌';

        return `
            <div class="bt-trade-row">
                <span>${entryDt}</span>
                <span>${exitDt}</span>
                <span>${t.entry_price} ₽</span>
                <span>${t.exit_price} ₽</span>
                <span style="color:${pnlColor}">${sign}${t.pnl} ₽</span>
                <span style="color:${pnlColor}">${sign}${t.pnl_pct}%</span>
                <span>${icon} ${t.action}</span>
            </div>
        `;
    }).join('');

    el.innerHTML = header + rows;
}

// ── График кривой капитала (LightweightCharts) ────────────
function renderEquityChart(equityCurve, initial) {
    const container = document.getElementById('bt-equity-chart');
    if (!container || !equityCurve.length) return;

    container.innerHTML = '';

    if (btState.equityChart) {
        try { btState.equityChart.remove(); } catch(_) {}
        btState.equityChart = null;
    }

    const chart = LightweightCharts.createChart(container, {
        width:  container.clientWidth,
        height: 220,
        layout: { background: { color: '#131722' }, textColor: '#d1d4dc' },
        grid:   { vertLines: { color: '#1e222d' }, horzLines: { color: '#1e222d' } },
        timeScale:       { borderColor: '#363a45', timeVisible: true },
        rightPriceScale: { borderColor: '#363a45' },
        crosshair: { mode: 1 },
    });

    btState.equityChart = chart;

    // Линия капитала
    const line = chart.addLineSeries({
        color:     '#2196F3',
        lineWidth: 2,
        title:     'Капитал',
    });

    // Базовая линия (initial)
    const baseline = chart.addLineSeries({
        color:     '#363a45',
        lineWidth: 1,
        lineStyle: 2,   // dashed
        title:     'Начало',
    });

    const points = equityCurve.map(e => ({ time: e.time, value: e.equity }));
    points.sort((a, b) => a.time - b.time);

    line.setData(points);
    baseline.setData([
        { time: points[0].time,  value: initial },
        { time: points.at(-1).time, value: initial },
    ]);

    chart.timeScale().fitContent();

    // Resize
    new ResizeObserver(() => {
        chart.applyOptions({ width: container.clientWidth });
    }).observe(container);
}

function setBtStatus(msg) {
    const el = document.getElementById('bt-status');
    if (el) el.textContent = msg;
}
