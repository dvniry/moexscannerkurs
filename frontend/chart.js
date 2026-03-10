// ── Лимиты дней по интервалу (Tinkoff API) ───────────────
const MAX_DAYS = {
    '1m': 1, '5m': 1, '15m': 3, '1h': 7, '1d': 365
};

// ── Состояние ─────────────────────────────────────────────
const state = {
    ticker: 'SBER', interval: '1h',
    candleSeries: null,
    subLines: [], mainLines: [],
    colors: ['#2196F3','#FF9800','#4CAF50','#9C27B0',
             '#F44336','#00BCD4','#FF5722','#E91E63'],
    colorIdx: 0,
    ws: null,
};

// ── Проверка зависимостей ─────────────────────────────────
if (typeof LightweightCharts === 'undefined') {
    document.getElementById('status').textContent =
        '❌ LightweightCharts не загружен — обнови страницу (Ctrl+Shift+R)';
    throw new Error('LightweightCharts not loaded');
}

const CHART_OPTS = {
    layout:    { background: { color: '#131722' }, textColor: '#d1d4dc' },
    grid:      { vertLines: { color: '#1e222d' }, horzLines: { color: '#1e222d' } },
    crosshair: { mode: 1 },
    timeScale: { borderColor: '#363a45', timeVisible: true, secondsVisible: false },
    rightPriceScale: { borderColor: '#363a45' },
};

// ── Инициализация графиков ────────────────────────────────
const mainEl = document.getElementById('chart-main');
const subEl  = document.getElementById('chart-sub');

if (!mainEl.clientHeight) {
    mainEl.style.height = '65vh';
    subEl.style.height  = '20vh';
}

const mainChart = LightweightCharts.createChart(mainEl, {
    ...CHART_OPTS,
    width:  mainEl.clientWidth  || window.innerWidth * 0.75,
    height: mainEl.clientHeight || window.innerHeight * 0.65,
});

const subChart = LightweightCharts.createChart(subEl, {
    ...CHART_OPTS,
    width:  subEl.clientWidth  || window.innerWidth * 0.75,
    height: subEl.clientHeight || window.innerHeight * 0.2,
});

// Синхронизация скролла
mainChart.timeScale().subscribeVisibleTimeRangeChange(r => {
    if (!r) return;
    try { subChart.timeScale().setVisibleRange(r); } catch (_) {}
});


// Свечной ряд
state.candleSeries = mainChart.addCandlestickSeries({
    upColor:         '#4CAF50',
    downColor:       '#f44336',
    borderUpColor:   '#4CAF50',
    borderDownColor: '#f44336',
    wickUpColor:     '#4CAF50',
    wickDownColor:   '#f44336',
});

// Resize
new ResizeObserver(() => {
    mainChart.applyOptions({ width: mainEl.clientWidth, height: mainEl.clientHeight });
    subChart.applyOptions({  width: subEl.clientWidth,  height: subEl.clientHeight  });
}).observe(document.querySelector('.chart-area'));

// ── Загрузка свечей ───────────────────────────────────────
async function loadChart() {
    if (!state.candleSeries) {
        setStatus('❌ График не инициализирован — обнови страницу');
        return;
    }

    const ticker   = document.getElementById('ticker').value.trim().toUpperCase();
    const interval = document.getElementById('interval').value;
    const days     = MAX_DAYS[interval] || 7;

    setStatus(`⏳ ${ticker} ${interval}...`);

    try {
        const r    = await fetch(`/api/candles?ticker=${ticker}&interval=${interval}&days=${days}`);
        const data = await r.json();

        if (!r.ok) { setStatus(`❌ ${data.detail}`); return; }

        state.candleSeries.setData(data.candles.map(c => ({
            time: c.time, open: c.open, high: c.high, low: c.low, close: c.close
        })));
        
        if (window.state?.candleSeries) {
            try { state.candleSeries.setMarkers([]); } catch(_) {}
        }

        state.candleCount = data.candles.length;

        mainChart.timeScale().fitContent();

        const last = data.candles.at(-1);
        setStatus(`✅ ${ticker} (${interval}): ${data.candles.length} свечей | ${last.close} ₽`);

        state.ticker   = ticker;
        state.interval = interval;

        // ← ФИКС: сохраняем figi и ticker глобально для стратегии/sandbox
        window._lastFigi   = data.figi;
        window._lastTicker = data.ticker;
        // ────────────────────────────────────────────────────────────────

        document.getElementById('rt-ticker').textContent = ticker;

        connectWS(ticker, interval);

    } catch (e) {
        console.error('loadChart error:', e);
        setStatus(`❌ ${e.message}`);
    }
}


// ── Кнопка "К текущей свече" ──────────────────────────────
function scrollToNow() {
    const total = state.candleCount || 100;
    const range = {
        from: total - 60,   // показываем последние 60 свечей
        to:   total + 3     // +3 отступ справа
    };
    mainChart.timeScale().setVisibleLogicalRange(range);
    subChart.timeScale().setVisibleLogicalRange(range);
}



// ── WebSocket real-time ───────────────────────────────────
function connectWS(ticker, interval) {
    if (state.ws) {
        state.ws.onclose = null;  // не триггерим реконнект старого
        state.ws.close();
        state.ws = null;
    }

    setDot(false);  // серый пока не подключились

    const ws = new WebSocket(`ws://localhost:8050/ws/price`);

    ws.onopen = () => {
        ws.send(JSON.stringify({ ticker, interval }));
        setDot(true);  // зелёный — подключились
    };

    ws.onmessage = ({ data }) => {
        const msg = JSON.parse(data);
        if (msg.error) return;

        // Шапка
        document.getElementById('rt-price').textContent = `${msg.price} ₽`;
        const changeEl = document.getElementById('rt-change');
        if      (msg.change > 0) { changeEl.textContent = `▲ +${msg.change}`; changeEl.className = 'change-up';      }
        else if (msg.change < 0) { changeEl.textContent = `▼ ${msg.change}`;  changeEl.className = 'change-down';    }
        else                     { changeEl.textContent = `— ${msg.change}`;  changeEl.className = 'change-neutral'; }

        // Обновляем последнюю свечу на графике
        if (msg.candle) {
            state.candleSeries.update(msg.candle);
        }
    };

    ws.onclose = () => {
        setDot(false);
        // Переподключаемся через 5 сек
        setTimeout(() => {
            if (state.ticker === ticker) connectWS(ticker, interval);
        }, 5000);
    };

    ws.onerror = () => {
        setStatus('⚠️ WebSocket: ошибка подключения');
        setDot(false);
    };

    state.ws = ws;
}

// Индикатор подключения
function setDot(live) {
    const dot = document.getElementById('rt-dot');
    if (!dot) return;
    dot.className = live ? 'rt-dot live' : 'rt-dot';
    dot.title     = live ? 'Real-time подключён' : 'Нет подключения';
}

// ── Добавить линию на график ──────────────────────────────
function addLine(name, points, pane, color) {
    const chart  = pane === 'main' ? mainChart : subChart;
    const series = chart.addLineSeries({ color, lineWidth: 2, title: name });
    series.setData(points.map(p => ({ time: p.time, value: p.value })));
    return series;
}

function nextColor() {
    return state.colors[(state.colorIdx++) % state.colors.length];
}

// ── Очистить индикаторы ───────────────────────────────────
function clearIndicators() {
    state.subLines.forEach(s  => { try { subChart.removeSeries(s);  } catch (_) {} });
    state.mainLines.forEach(s => { try { mainChart.removeSeries(s); } catch (_) {} });
    state.subLines  = [];
    state.mainLines = [];
    state.colorIdx  = 0;
    document.getElementById('ind-list').innerHTML = '<span class="muted">Пока нет</span>';
    setStatus('✅ Все индикаторы удалены');
}

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

// ── Переключение вкладок ──────────────────────────────────
function switchTab(name) {
    ['chart', 'scanner', 'strategy', 'backtest'].forEach(n => {
        document.getElementById(`page-${n}`).style.display = 'none';
        document.getElementById(`tab-${n}`).classList.remove('active');
    });

    document.getElementById(`page-${name}`).style.display = 'flex';
    document.getElementById(`tab-${name}`).classList.add('active');

    if (name === 'chart') {
        setTimeout(() => {
            mainChart.applyOptions({ width: mainEl.clientWidth, height: mainEl.clientHeight });
            subChart.applyOptions({  width: subEl.clientWidth,  height: subEl.clientHeight  });
        }, 50);
    }

    if (name === 'strategy') initStrategy();
    if (name === 'backtest') initBacktest();
}



// ── Запуск ────────────────────────────────────────────────
loadChart();
