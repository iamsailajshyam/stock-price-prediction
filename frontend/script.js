document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const tickerSelect = document.getElementById('ticker-select');
    const timeframeSelect = document.getElementById('timeframe-select');
    const predictBtn = document.getElementById('predict-btn');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');
    
    const displayTicker = document.getElementById('current-ticker-display');
    const currentPrice = document.getElementById('current-price');
    const targetPriceDisplay = document.getElementById('target-price-display');
    const sentimentValue = document.getElementById('sentiment-value');
    const sentimentScore = document.getElementById('sentiment-score');
    
    // Toggles
    const toggleSma = document.getElementById('toggle-sma');
    const toggleRsi = document.getElementById('toggle-rsi');
    const toggleArima = document.getElementById('toggle-arima');
    
    // Initialize Chart
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    const historicalGradient = ctx.createLinearGradient(0, 0, 0, 400);
    historicalGradient.addColorStop(0, 'rgba(148, 163, 184, 0.2)');
    historicalGradient.addColorStop(1, 'rgba(148, 163, 184, 0)');
    
    const predictedGradient = ctx.createLinearGradient(0, 0, 0, 400);
    predictedGradient.addColorStop(0, 'rgba(59, 130, 246, 0.4)');
    predictedGradient.addColorStop(1, 'rgba(59, 130, 246, 0)');

    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = 'Inter';

    let myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Historical', data: [], borderColor: '#94a3b8', backgroundColor: historicalGradient, borderWidth: 2, tension: 0.4, pointRadius: 0, pointHitRadius: 10, fill: true, yAxisID: 'y'
                },
                {
                    label: 'LSTM Predicted', data: [], borderColor: '#3b82f6', backgroundColor: predictedGradient, borderWidth: 3, borderDash: [5, 5], tension: 0.4, pointRadius: 4, pointBackgroundColor: '#0b0f19', pointBorderColor: '#3b82f6', pointBorderWidth: 2, fill: true, yAxisID: 'y'
                },
                {
                    label: 'ARIMA Predicted', data: [], borderColor: '#f59e0b', borderWidth: 2, borderDash: [3, 3], tension: 0.4, pointRadius: 0, fill: false, hidden: true, yAxisID: 'y'
                },
                {
                    label: 'SMA Overlay', data: [], borderColor: '#8b5cf6', borderWidth: 2, tension: 0.4, pointRadius: 0, fill: false, hidden: true, yAxisID: 'y'
                },
                {
                    label: 'RSI Indicator', data: [], borderColor: '#10b981', borderWidth: 1.5, tension: 0.4, pointRadius: 0, fill: false, hidden: true, yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.9)', titleColor: '#fff', bodyColor: '#cbd5e1', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, padding: 12, boxPadding: 4, usePointStyle: true }
            },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false }, ticks: { maxTicksLimit: 10 } },
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false }, border: { dash: [4, 4] }, ticks: { callback: v => '$' + v } },
                y1: { position: 'right', display: false, min: 0, max: 100, grid: { display: false } }
            }
        }
    });

    // Technical Indicators Generators
    const calculateSMA = (data, window) => {
        let sma = [];
        for (let i = 0; i < data.length; i++) {
            if (i < window - 1 || data[i] === null) {
                sma.push(null);
            } else {
                let sum = 0;
                for (let j = 0; j < window; j++) sum += parseFloat(data[i - j]);
                sma.push((sum / window).toFixed(2));
            }
        }
        return sma;
    };

    const calculateRSI = (data, window) => {
        let rsi = [];
        let gains = 0, losses = 0;
        for (let i = 0; i < data.length; i++) {
            if (i === 0 || data[i] === null || data[i-1] === null) {
                rsi.push(null);
                continue;
            }
            let diff = parseFloat(data[i]) - parseFloat(data[i-1]);
            if (i <= window) {
                if (diff > 0) gains += diff; else losses -= diff;
                if (i === window) {
                    let rs = (gains/window) / (losses/window || 1);
                    rsi.push((100 - (100 / (1 + rs))).toFixed(2));
                } else rsi.push(null);
            } else {
                let gain = diff > 0 ? diff : 0;
                let loss = diff < 0 ? -diff : 0;
                gains = (gains * (window - 1) + gain) / window;
                losses = (losses * (window - 1) + loss) / window;
                let rs = gains / (losses || 1);
                rsi.push((100 - (100 / (1 + rs))).toFixed(2));
            }
        }
        return rsi;
    };

    const generateMockData = (ticker, days) => {
        const basePrices = { 'AAPL': 228.00, 'MSFT': 506.00, 'AMZN': 225.00, 'TSLA': 324.00 };
        const basePrice = basePrices[ticker] || 100;
        const volatility = ticker === 'TSLA' ? 0.03 : 0.015;
        
        const labels = [], histData = [], predData = [], arimaData = [];
        let currentVal = basePrice * 0.95; 
        const today = new Date();
        
        // 60 days history for SMA/RSI calculation
        for(let i = 60; i > 0; i--) {
            const d = new Date(today); d.setDate(d.getDate() - i);
            labels.push(d.toLocaleDateString('en-US', {month: 'short', day: 'numeric'}));
            currentVal = currentVal * (1 + (Math.random() - 0.48) * volatility);
            histData.push(currentVal.toFixed(2));
            predData.push(null); arimaData.push(null);
        }
        
        labels.push('Today');
        histData.push(currentVal.toFixed(2));
        predData.push(currentVal.toFixed(2));
        arimaData.push(currentVal.toFixed(2));
        
        let predVal = currentVal, arimaVal = currentVal;
        for(let i = 1; i <= parseInt(days); i++) {
            const d = new Date(today); d.setDate(d.getDate() + i);
            labels.push(d.toLocaleDateString('en-US', {month: 'short', day: 'numeric'}));
            predVal = predVal * (1 + (Math.random() - 0.45) * volatility); // LSTM trend
            arimaVal = arimaVal * (1 + (Math.random() - 0.5) * volatility); // ARIMA random walk
            histData.push(null);
            predData.push(predVal.toFixed(2));
            arimaData.push(arimaVal.toFixed(2));
        }
        
        const smaData = calculateSMA(histData, 10); // Use 10-day for mock scaling
        const rsiData = calculateRSI(histData, 14);

        // Sentiment Logic
        const sentimentScoreNum = (Math.random() * 8 - 4).toFixed(1);
        const sentimentTxt = sentimentScoreNum > 2 ? 'Bullish' : (sentimentScoreNum < -2 ? 'Bearish' : 'Neutral');
        
        return {
            labels, historical: histData, predicted: predData, arima: arimaData, sma: smaData, rsi: rsiData,
            current: currentVal.toFixed(2), target: predVal.toFixed(2),
            sentiment: sentimentTxt, sentimentScore: `${sentimentScoreNum > 0 ? '+' : ''}${Math.abs(sentimentScoreNum)} NLP`
        };
    };

    const renderDashboardData = (ticker, data) => {
        displayTicker.textContent = `${ticker} Forecast`;
        currentPrice.textContent = `$${data.current}`;
        targetPriceDisplay.textContent = `$${data.target}`;
        
        // Sentiment
        if (sentimentValue && sentimentScore) {
            sentimentValue.textContent = data.sentiment;
            sentimentScore.textContent = data.sentimentScore;
            sentimentScore.className = `sub ${data.sentiment === 'Bearish' ? 'negative' : 'positive'}`;
        }
        
        myChart.data.labels = data.labels;
        myChart.data.datasets[0].data = data.historical;
        myChart.data.datasets[1].data = data.predicted;
        myChart.data.datasets[2].data = data.arima;
        myChart.data.datasets[3].data = data.sma;
        myChart.data.datasets[4].data = data.rsi;
        myChart.update();
        
        spinner.classList.add('hidden'); btnText.style.display = 'block'; predictBtn.disabled = false;
    };

    const updateDashboard = async (ticker, days) => {
        btnText.style.display = 'none'; spinner.classList.remove('hidden'); predictBtn.disabled = true;
        
        try {
            // Attempt to fetch from real Local Python FastAPI Backend
            const [predictRes, sentimentRes] = await Promise.all([
                fetch(`/api/predict?ticker=${ticker}&days=${days}`),
                fetch(`/api/sentiment?ticker=${ticker}`)
            ]);
            
            if (!predictRes.ok) throw new Error("Backend not available");
            
            const pData = await predictRes.json();
            const sData = await sentimentRes.json();
            
            // Reconstruct ARIMA data placeholder for real backend
            let arimaData = [];
            for (let i = 0; i < pData.labels.length; i++) arimaData.push(null);
            
            const data = {
                labels: pData.labels,
                historical: pData.historical,
                predicted: pData.predicted,
                arima: arimaData,
                sma: calculateSMA(pData.historical, 10),
                rsi: calculateRSI(pData.historical, 14),
                current: pData.current,
                target: pData.predicted[pData.predicted.length - 1],
                sentiment: sData.status,
                sentimentScore: `${sData.score > 0 ? '+' : ''}${Math.abs(sData.score)} NLP`
            };
            
            renderDashboardData(ticker, data);
        } catch (e) {
            console.warn("Backend unavailable. Falling back to internal Mock Data pipeline.", e);
            setTimeout(() => {
                const data = generateMockData(ticker, days);
                renderDashboardData(ticker, data);
            }, 800);
        }
    };

    predictBtn.addEventListener('click', () => updateDashboard(tickerSelect.value, timeframeSelect.value));
    
    // Toggle Overlays
    toggleSma.addEventListener('change', (e) => {
        myChart.data.datasets[3].hidden = !e.target.checked;
        myChart.update();
    });
    toggleRsi.addEventListener('change', (e) => {
        myChart.data.datasets[4].hidden = !e.target.checked;
        myChart.options.scales.y1.display = e.target.checked;
        myChart.update();
    });
    toggleArima.addEventListener('change', (e) => {
        myChart.data.datasets[2].hidden = !e.target.checked;
        myChart.update();
    });

    updateDashboard('AAPL', 7);

    // --- Authentication UI Logic ---
    const authModal = document.getElementById('auth-modal');
    const loginBtn = document.getElementById('login-btn');
    const registerBtn = document.getElementById('register-btn');
    const closeModal = document.getElementById('close-modal');
    const authForm = document.getElementById('auth-form');
    const modalTitle = document.getElementById('modal-title');
    const submitAuth = document.getElementById('submit-auth');
    const switchAuthMode = document.getElementById('switch-auth-mode');
    const authArea = document.getElementById('auth-area');
    
    const welcomeScreen = document.getElementById('welcome-screen');
    const appContainer = document.getElementById('app-container');
    const welcomeLoginBtn = document.getElementById('welcome-login-btn');
    const welcomeRegisterLink = document.getElementById('welcome-register-link');
    
    // Initial Auth Check
    const token = localStorage.getItem('neuralstock_token');
    const savedEmail = localStorage.getItem('neuralstock_email') || 'User';
    if (token) {
        if (welcomeScreen) welcomeScreen.style.display = 'none';
        if (appContainer) appContainer.style.display = 'grid';
        
        const name = savedEmail.split('@')[0];
        authArea.innerHTML = `
            <div class="user-profile">
                <div class="avatar">${name.charAt(0).toUpperCase()}</div>
                <span class="username">${name}</span>
                <button class="logout-btn" id="logout-btn">Logout</button>
            </div>
        `;
        setTimeout(() => {
            const logout = document.getElementById('logout-btn');
            if(logout) logout.addEventListener('click', () => {
                localStorage.removeItem('neuralstock_token');
                localStorage.removeItem('neuralstock_email');
                location.reload(); 
            });
        }, 100);
    }
    
    let isLoginMode = true;
    const openModal = (mode) => {
        isLoginMode = (mode === 'login');
        modalTitle.textContent = isLoginMode ? 'Welcome Back' : 'Create Account';
        submitAuth.textContent = isLoginMode ? 'Sign In' : 'Sign Up';
        switchAuthMode.textContent = isLoginMode ? 'Sign up' : 'Log in';
        switchAuthMode.parentElement.firstChild.textContent = isLoginMode ? "Don't have an account? " : "Already have an account? ";
        authModal.classList.remove('hidden');
    };
    
    if (loginBtn) loginBtn.addEventListener('click', () => openModal('login'));
    if (welcomeLoginBtn) welcomeLoginBtn.addEventListener('click', () => openModal('login'));
    if (registerBtn) registerBtn.addEventListener('click', () => openModal('register'));
    if (welcomeRegisterLink) welcomeRegisterLink.addEventListener('click', (e) => { e.preventDefault(); openModal('register'); });
    if (closeModal) closeModal.addEventListener('click', () => authModal.classList.add('hidden'));
    if (authModal) authModal.addEventListener('click', (e) => { if (e.target === authModal) authModal.classList.add('hidden'); });
    if (switchAuthMode) switchAuthMode.addEventListener('click', (e) => { e.preventDefault(); openModal(isLoginMode ? 'register' : 'login'); });
    
    if (authForm) {
        authForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('auth-email').value;
            const password = document.getElementById('auth-password').value;
            const name = email.split('@')[0];
            const endpoint = isLoginMode ? '/auth/login' : '/auth/register';
            
            try {
                // Call actual backend authentication
                const res = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password })
                });
                
                if (!res.ok) throw new Error("Auth Server Offline");
                
                const data = await res.json();
                if(data.access_token) {
                    localStorage.setItem('neuralstock_token', data.access_token);
                    localStorage.setItem('neuralstock_email', email);
                }
                
                authModal.classList.add('hidden');
                if (welcomeScreen) welcomeScreen.style.display = 'none';
                if (appContainer) appContainer.style.display = 'grid';
                
                authArea.innerHTML = `
                    <div class="user-profile">
                        <div class="avatar">${name.charAt(0).toUpperCase()}</div>
                        <span class="username">${name}</span>
                        <button class="logout-btn" id="logout-btn">Logout</button>
                    </div>
                `;
                document.getElementById('logout-btn').addEventListener('click', () => {
                    localStorage.removeItem('neuralstock_token');
                    localStorage.removeItem('neuralstock_email');
                    location.reload(); 
                });
            } catch (err) {
                console.warn("Auth Backend Unavailable, executing Mock Auth flow instead.", err);
                setTimeout(() => {
                    localStorage.setItem('neuralstock_token', 'mock_token');
                    localStorage.setItem('neuralstock_email', email);
                    
                    authModal.classList.add('hidden');
                    if (welcomeScreen) welcomeScreen.style.display = 'none';
                    if (appContainer) appContainer.style.display = 'grid';
                    
                    authArea.innerHTML = `
                        <div class="user-profile">
                            <div class="avatar">${name.charAt(0).toUpperCase()}</div>
                            <span class="username">${name}</span>
                            <button class="logout-btn" id="logout-btn">Logout</button>
                        </div>
                    `;
                    document.getElementById('logout-btn').addEventListener('click', () => {
                        localStorage.removeItem('neuralstock_token');
                        localStorage.removeItem('neuralstock_email');
                        location.reload();
                    });
                }, 800);
            }
        });
    }
});
