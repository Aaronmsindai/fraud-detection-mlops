import { useState, useEffect } from 'react';
import './App.css';

function App() {
  // --- Auth State ---
  const [token, setToken] = useState(''); 
  const [role, setRole] = useState('teller');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [authError, setAuthError] = useState('');

  // --- App State ---
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // --- Prediction State ---
  const [cardNetwork, setCardNetwork] = useState('Unknown');
  const [cardNumber, setCardNumber] = useState('');
  const [amount, setAmount] = useState('');
  const [hour, setHour] = useState('');
  const [age, setAge] = useState('');
  const [isInternational, setIsInternational] = useState(false);
  const [ipAddress, setIpAddress] = useState('192.168.1.1'); 
  const [deviceId, setDeviceId] = useState('iPhone-12');

  // --- Real-Time Validation Errors ---
  const [formErrors, setFormErrors] = useState({
    cardNumber: '', amount: '', hour: '', age: '', ipAddress: '', deviceId: ''
  });
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [driftData, setDriftData] = useState(null);
  const [auditLogs, setAuditLogs] = useState([]);
  const [retraining, setRetraining] = useState(false);

  const API_URL = 'https://fraud-detection-api-71aw.onrender.com';

  // --- Fetch Functions ---
  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_URL}/history`, { headers: { 'Authorization': `Bearer ${token}` } });
      if (res.ok) setHistory(await res.json());
    } catch (error) { console.error(error); }
  };

  const fetchDrift = async () => {
    try {
      const res = await fetch(`${API_URL}/admin/drift`, { headers: { 'Authorization': `Bearer ${token}` } });
      if (res.ok) setDriftData(await res.json());
    } catch (error) { console.error(error); }
  };

  const fetchAuditLogs = async () => {
    try {
      const res = await fetch(`${API_URL}/admin/audit`, { headers: { 'Authorization': `Bearer ${token}` } });
      if (res.ok) setAuditLogs(await res.json());
    } catch (error) { console.error(error); }
  };

  const triggerRetrain = async () => {
    setRetraining(true);
    try {
      const res = await fetch(`${API_URL}/admin/retrain`, { method: 'POST', headers: { 'Authorization': `Bearer ${token}` } });
      const data = await res.json();
      alert(data.message);
      fetchAuditLogs(); 
    } catch (error) { console.error(error); } finally { setRetraining(false); }
  };

  useEffect(() => {
    if (token) {
      fetchHistory();
      if (role === 'admin') { fetchDrift(); fetchAuditLogs(); }
    }
  }, [token, role]);

  // --- Auth Handlers ---
  const handleAuth = async (e) => {
    e.preventDefault();
    setAuthError('');
    
    try {
      if (isRegistering) {
        if (!/^[a-zA-Z]/.test(username)) { setAuthError("Username must start with a letter."); return; }
        if (!/^[a-zA-Z][a-zA-Z0-9_]*$/.test(username)) { setAuthError("Username can only contain letters, numbers, and underscores."); return; }

        const res = await fetch(`${API_URL}/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password }),
        });
        
        if (res.ok) {
          alert("Registration successful! Please log in.");
          setIsRegistering(false);
        } else {
          const data = await res.json();
          if (Array.isArray(data.detail)) { setAuthError(data.detail[0].msg); }
          else if (typeof data.detail === 'object') { setAuthError("Validation error. Please check your inputs."); }
          else { setAuthError(data.detail || "Registration failed"); }
        }
      } else {
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);
        const res = await fetch(`${API_URL}/login`, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: formData });
        
        if (res.ok) {
          const data = await res.json();
          setToken(data.access_token);
          setRole(data.role);
          setActiveTab('dashboard');
        } else {
          setAuthError("Invalid username or password");
        }
      }
    } catch (error) {
      setAuthError("Failed to connect to the server.");
    }
  };

  const handleLogout = () => {
    setToken(''); setRole('teller'); setResult(null); setHistory([]); setDriftData(null); setAuditLogs([]); setActiveTab('dashboard');
  };

  // --- Helpers ---
  const detectCardNetwork = (number) => {
    const clean = number.replace(/[\s-]/g, '');
    if (/^4/.test(clean)) return "Visa";
    if (/^(5[1-5]|2[2-7])/.test(clean)) return "Mastercard";
    if (/^3[47]/.test(clean)) return "Amex";
    if (/^62/.test(clean)) return "UnionPay";
    return "Unknown";
  };

  const isValidLuhn = (cardNumber) => {
    let cleanNumber = cardNumber.replace(/[\s-]/g, '');
    if (!/^\d+$/.test(cleanNumber)) return false;
    let sum = 0, shouldDouble = false;
    for (let i = cleanNumber.length - 1; i >= 0; i--) {
      let digit = parseInt(cleanNumber.charAt(i));
      if (shouldDouble) { digit *= 2; if (digit > 9) digit -= 9; }
      sum += digit; shouldDouble = !shouldDouble;
    }
    return sum % 10 === 0;
  };

  // --- Comprehensive Validation Function ---
  const validateForm = () => {
    let errors = { cardNumber: '', amount: '', hour: '', age: '', ipAddress: '', deviceId: '' };
    let isValid = true;

    // 1. Card Number
    const cleanCard = cardNumber.replace(/\D/g, '');
    if (cleanCard.length < 16 || cleanCard.length > 19) {
      errors.cardNumber = "Card number must be 16-19 digits.";
      isValid = false;
    } else if (!isValidLuhn(cleanCard)) {
      errors.cardNumber = "Invalid card number (Luhn check failed).";
      isValid = false;
    }

    // 2. Amount (Must be > 0 and < 1M)
    if (isNaN(parseFloat(amount)) || parseFloat(amount) <= 0) {
      errors.amount = "Amount must be greater than $0.";
      isValid = false;
    } else if (parseFloat(amount) >= 1000000) {
      errors.amount = "Amount exceeds maximum limit of $1,000,000.";
      isValid = false;
    }

    // 3. Hour (0-23)
    const hourNum = parseInt(hour);
    if (isNaN(hourNum) || hourNum < 0 || hourNum > 23) {
      errors.hour = "Hour must be between 0 and 23.";
      isValid = false;
    }

    // 4. Age (18-120)
    const ageNum = parseInt(age);
    if (isNaN(ageNum) || ageNum < 18 || ageNum > 120) {
      errors.age = "Age must be between 18 and 120.";
      isValid = false;
    }

    // 5. IP Address (Valid IPv4)
    const ipRegex = /^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
    if (!ipRegex.test(ipAddress)) {
      errors.ipAddress = "Invalid IPv4 address (e.g. 192.168.1.1).";
      isValid = false;
    }

    // 6. Device ID (Min 3 chars)
    if (deviceId.trim().length < 3) {
      errors.deviceId = "Device ID must be at least 3 characters.";
      isValid = false;
    }

    setFormErrors(errors);
    return isValid;
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    
    // Run validation before sending to backend
    if (!validateForm()) return;

    setLoading(true); setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({
          card_network: cardNetwork, 
          card_number: cardNumber.replace(/\s/g, ''),
          amount: parseFloat(amount),
          hour_of_day: parseInt(hour), 
          user_age: parseInt(age), 
          is_international: isInternational ? 1 : 0,
          ip_address: ipAddress,
          device_id: deviceId
        }),
      });

      if (response.status === 401) { handleLogout(); return; }
      if (!response.ok) throw new Error('API Error');
      
      const data = await response.json();
      setResult(data);
      
      if (data.prediction === "STEP-UP AUTH") {
        alert("🛑 STEP-UP AUTHENTICATION REQUIRED: An OTP has been sent to the customer's phone.");
      }

      fetchHistory(); 
      if (role === 'admin') { fetchDrift(); fetchAuditLogs(); }
    } catch (error) {
      alert("Failed to connect to the API.");
    } finally {
      setLoading(false);
    }
  };

  // --- Render Login/Register ---
  if (!token) {
    return (
      <div className="app-container">
        <div className="header">
          <h1>🏦 Secure Bank Portal</h1>
          <p>Employee Authentication</p>
        </div>
        <div className="card" style={{ maxWidth: '400px', margin: '0 auto' }}>
          <form onSubmit={handleAuth}>
            <h2 style={{ marginBottom: '20px', textAlign: 'center', color: '#4DB8FF' }}>
              {isRegistering ? 'Create Account' : 'Sign In'}
            </h2>
            {authError && <div style={{ color: '#FF4B4B', marginBottom: '15px', padding: '10px', background: 'rgba(255,75,75,0.1)', borderRadius: '5px', fontSize: '0.9rem' }}>{authError}</div>}
            
            <div className="form-group">
              <label>Username</label>
              <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} required minLength="3" />
            </div>
            <div className="form-group">
              <label>Password</label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
            </div>
            
            <button type="submit" style={{ marginTop: '20px' }}>{isRegistering ? 'Register' : 'Login'}</button>
          </form>
          
          <p style={{ marginTop: '20px', textAlign: 'center', cursor: 'pointer', color: '#4DB8FF' }} onClick={() => setIsRegistering(!isRegistering)}>
            {isRegistering ? 'Already have an account? Login' : 'Need an account? Register'}
          </p>
        </div>
      </div>
    );
  }

  // --- Calculate KPIs ---
  const totalAnalyzed = history.length;
  const fraudCount = history.filter(tx => tx.prediction === 'BLOCK').length;
  const safeCount = history.filter(tx => tx.prediction === 'APPROVE').length;
  const fraudRate = totalAnalyzed > 0 ? ((fraudCount / totalAnalyzed) * 100).toFixed(1) : 0;

  // --- Render Dashboard ---
  return (
    <div className="dashboard-wrapper">
      {/* SIDEBAR */}
      <div className="sidebar">
        <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#4DB8FF', marginBottom: '20px' }}>⚡ FraudShield AI</div>
        <div className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>Dashboard</div>
        <div className={`nav-item ${activeTab === 'transactions' ? 'active' : ''}`} onClick={() => setActiveTab('transactions')}>Transactions</div>
        <div className={`nav-item ${activeTab === 'analytics' ? 'active' : ''}`} onClick={() => setActiveTab('analytics')}>Risk Analytics</div>
        {role === 'admin' && <div className={`nav-item ${activeTab === 'admin' ? 'active' : ''}`} onClick={() => setActiveTab('admin')}>Admin Settings</div>}
        <div style={{ marginTop: 'auto' }}><div className="nav-item" onClick={handleLogout} style={{ color: '#FF4B4B' }}>Logout</div></div>
      </div>

      {/* MAIN CONTENT */}
      <div className="main-content">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ margin: 0, color: '#FAFAFA' }}>
              {activeTab === 'dashboard' && 'Dashboard Overview'}
              {activeTab === 'transactions' && 'All Transactions'}
              {activeTab === 'analytics' && 'Risk Analytics'}
              {activeTab === 'admin' && 'Admin Settings'}
            </h1>
            <p style={{ color: '#A0AABF', margin: '5px 0 0 0' }}>Real-time transaction analysis</p>
          </div>
          <div style={{ color: '#4DB8FF', fontWeight: 'bold' }}>Role: {role.toUpperCase()}</div>
        </div>

        {/* --- TAB: DASHBOARD --- */}
        {activeTab === 'dashboard' && (
          <>
            <div className="kpi-grid">
              <div className="kpi-card">
                <span className="kpi-title">Total Analyzed</span>
                <span className="kpi-value">{totalAnalyzed}</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-title">Frauds Blocked</span>
                <span className="kpi-value danger">{fraudCount}</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-title">Approved</span>
                <span className="kpi-value success">{safeCount}</span>
              </div>
            </div>

            <div className="workspace-grid">
              {/* Form */}
              <div className="dashboard-card">
                <h3 style={{ marginBottom: '20px', color: '#FAFAFA' }}>Analyze New Transaction</h3>
                <form onSubmit={handlePredict} noValidate>
                  
                  {/* Card Number */}
                  <div className="form-group">
                    <label style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Card Number</span>
                      {cardNumber.length >= 1 && <span style={{ color: '#4DB8FF', fontWeight: 'bold', fontSize: '0.8rem' }}>{detectCardNetwork(cardNumber)}</span>}
                    </label>
                    <input 
                      type="text" 
                      value={cardNumber} 
                      onChange={(e) => {
                        // Sanitize: Only allow numbers and spaces
                        const sanitized = e.target.value.replace(/[^0-9 ]/g, '');
                        setCardNumber(sanitized); 
                        setCardNetwork(detectCardNetwork(sanitized));
                        if (formErrors.cardNumber) setFormErrors({...formErrors, cardNumber: ''});
                      }} 
                      placeholder="4242 4242 4242 4242" 
                      style={{ borderColor: formErrors.cardNumber ? '#FF4B4B' : '' }}
                    />
                    {formErrors.cardNumber && <span style={{ color: '#FF4B4B', fontSize: '0.8rem', marginTop: '5px' }}>{formErrors.cardNumber}</span>}
                  </div>

                  {/* Amount */}
                  <div className="form-group">
                    <label>Amount ($)</label>
                    <input 
                      type="number" 
                      value={amount} 
                      onChange={(e) => {
                        setAmount(e.target.value);
                        if (formErrors.amount) setFormErrors({...formErrors, amount: ''});
                      }} 
                      placeholder="e.g. 4500" 
                      style={{ borderColor: formErrors.amount ? '#FF4B4B' : '' }}
                    />
                    {formErrors.amount && <span style={{ color: '#FF4B4B', fontSize: '0.8rem', marginTop: '5px' }}>{formErrors.amount}</span>}
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    {/* Hour */}
                    <div className="form-group">
                      <label>Hour (0-23)</label>
                      <input 
                        type="number" 
                        value={hour} 
                        onChange={(e) => {
                          setHour(e.target.value);
                          if (formErrors.hour) setFormErrors({...formErrors, hour: ''});
                        }} 
                        placeholder="e.g. 2" 
                        style={{ borderColor: formErrors.hour ? '#FF4B4B' : '' }}
                      />
                      {formErrors.hour && <span style={{ color: '#FF4B4B', fontSize: '0.8rem', marginTop: '5px' }}>{formErrors.hour}</span>}
                    </div>
                    {/* Age */}
                    <div className="form-group">
                      <label>User Age</label>
                      <input 
                        type="number" 
                        value={age} 
                        onChange={(e) => {
                          setAge(e.target.value);
                          if (formErrors.age) setFormErrors({...formErrors, age: ''});
                        }} 
                        placeholder="e.g. 25" 
                        style={{ borderColor: formErrors.age ? '#FF4B4B' : '' }}
                      />
                      {formErrors.age && <span style={{ color: '#FF4B4B', fontSize: '0.8rem', marginTop: '5px' }}>{formErrors.age}</span>}
                    </div>
                  </div>
                  
                  {/* IP and Device */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    <div className="form-group">
                      <label>IP Address</label>
                      <input 
                        type="text" 
                        value={ipAddress} 
                        onChange={(e) => {
                          setIpAddress(e.target.value);
                          if (formErrors.ipAddress) setFormErrors({...formErrors, ipAddress: ''});
                        }} 
                        style={{ borderColor: formErrors.ipAddress ? '#FF4B4B' : '' }}
                      />
                      {formErrors.ipAddress && <span style={{ color: '#FF4B4B', fontSize: '0.8rem', marginTop: '5px' }}>{formErrors.ipAddress}</span>}
                    </div>
                    <div className="form-group">
                      <label>Device ID</label>
                      <input 
                        type="text" 
                        value={deviceId} 
                        onChange={(e) => {
                          setDeviceId(e.target.value);
                          if (formErrors.deviceId) setFormErrors({...formErrors, deviceId: ''});
                        }} 
                        style={{ borderColor: formErrors.deviceId ? '#FF4B4B' : '' }}
                      />
                      {formErrors.deviceId && <span style={{ color: '#FF4B4B', fontSize: '0.8rem', marginTop: '5px' }}>{formErrors.deviceId}</span>}
                    </div>
                  </div>

                  <div className="checkbox-group" style={{ marginBottom: '20px' }}>
                    <input type="checkbox" checked={isInternational} onChange={(e) => setIsInternational(e.target.checked)} />
                    <label>International Transaction</label>
                  </div>
                  <button type="submit" disabled={loading} style={{ width: '100%' }}>{loading ? 'Analyzing...' : 'Analyze Transaction'}</button>
                </form>

                {/* RESULT BOX (3-Tier) */}
                {result && (
                  <div style={{ 
                    marginTop: '20px', padding: '20px', borderRadius: '8px', textAlign: 'center',
                    background: result.prediction === 'BLOCK' ? 'rgba(255, 75, 75, 0.1)' : result.prediction === 'STEP-UP AUTH' ? 'rgba(255, 184, 0, 0.1)' : 'rgba(44, 160, 44, 0.1)',
                    border: `1px solid ${result.prediction === 'BLOCK' ? '#FF4B4B' : result.prediction === 'STEP-UP AUTH' ? '#FFB800' : '#2ca02c'}`,
                  }}>
                    <h2 style={{ margin: 0, color: result.prediction === 'BLOCK' ? '#FF4B4B' : result.prediction === 'STEP-UP AUTH' ? '#FFB800' : '#2ca02c' }}>
                      {result.prediction}
                    </h2>
                    <p style={{ margin: '5px 0', color: '#FAFAFA' }}>Confidence: {result.fraud_probability}%</p>
                  </div>
                )}
              </div>

              {/* Mini Table */}
              <div className="dashboard-card">
                <h3 style={{ marginBottom: '20px', color: '#FAFAFA' }}>Recent Transactions</h3>
                {history.length > 0 ? (
                  <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid #2E3A4E', color: '#A0AABF' }}>
                        <th style={{ padding: '10px' }}>Card</th>
                        <th style={{ padding: '10px' }}>Amount</th>
                        <th style={{ padding: '10px' }}>Result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {history.slice(0, 5).map((tx, index) => (
                        <tr key={index} className="table-row-hover" style={{ borderBottom: '1px solid #1E232E', color: '#FAFAFA' }}>
                          <td style={{ padding: '15px 10px' }}>{tx.card_network} (...{tx.card_number.slice(-4)})</td>
                          <td style={{ padding: '15px 10px' }}>${tx.amount}</td>
                          <td style={{ padding: '15px 10px' }}>
                            <span style={{ 
                              padding: '5px 10px', borderRadius: '20px', fontSize: '0.8rem', fontWeight: 'bold', 
                              background: tx.prediction === 'BLOCK' ? 'rgba(255, 75, 75, 0.1)' : tx.prediction === 'STEP-UP AUTH' ? 'rgba(255, 184, 0, 0.1)' : 'rgba(44, 160, 44, 0.1)', 
                              color: tx.prediction === 'BLOCK' ? '#FF4B4B' : tx.prediction === 'STEP-UP AUTH' ? '#FFB800' : '#2ca02c' 
                            }}>
                              {tx.prediction}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : <p style={{ textAlign: 'center', color: '#A0AABF', marginTop: '30px' }}>No recent transactions.</p>}
              </div>
            </div>
          </>
        )}

        {/* --- TAB: TRANSACTIONS --- (Same as before) */}
        {activeTab === 'transactions' && (
          <div className="dashboard-card">
            <h3 style={{ marginBottom: '20px', color: '#FAFAFA' }}>Full Transaction History</h3>
            {history.length > 0 ? (
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #2E3A4E', color: '#A0AABF' }}>
                    <th style={{ padding: '10px' }}>Card</th>
                    <th style={{ padding: '10px' }}>Amount</th>
                    <th style={{ padding: '10px' }}>IP / Device</th>
                    <th style={{ padding: '10px' }}>Time</th>
                    <th style={{ padding: '10px' }}>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((tx, index) => (
                    <tr key={index} className="table-row-hover" style={{ borderBottom: '1px solid #1E232E', color: '#FAFAFA' }}>
                      <td style={{ padding: '15px 10px' }}>{tx.card_network} (...{tx.card_number.slice(-4)})</td>
                      <td style={{ padding: '15px 10px' }}>${tx.amount}</td>
                      <td style={{ padding: '15px 10px', color: '#A0AABF', fontSize: '0.8rem' }}>{tx.ip_address} <br/> {tx.device_id}</td>
                      <td style={{ padding: '15px 10px', color: '#A0AABF' }}>
                        {tx.hour_of_day === 0 ? '12:00 AM' : tx.hour_of_day === 12 ? '12:00 PM' : tx.hour_of_day > 12 ? `${tx.hour_of_day - 12}:00 PM` : `${tx.hour_of_day}:00 AM`}
                      </td>
                      <td style={{ padding: '15px 10px' }}>
                        <span style={{ 
                          padding: '5px 10px', borderRadius: '20px', fontSize: '0.8rem', fontWeight: 'bold', 
                          background: tx.prediction === 'BLOCK' ? 'rgba(255, 75, 75, 0.1)' : tx.prediction === 'STEP-UP AUTH' ? 'rgba(255, 184, 0, 0.1)' : 'rgba(44, 160, 44, 0.1)', 
                          color: tx.prediction === 'BLOCK' ? '#FF4B4B' : tx.prediction === 'STEP-UP AUTH' ? '#FFB800' : '#2ca02c' 
                        }}>
                          {tx.prediction}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <p style={{ textAlign: 'center', color: '#A0AABF', marginTop: '30px' }}>No transactions analyzed yet.</p>}
          </div>
        )}

        {/* --- TAB: RISK ANALYTICS --- (Same as before) */}
        {activeTab === 'analytics' && (
          <div className="workspace-grid">
            <div className="dashboard-card">
              <h3 style={{ color: '#FAFAFA', marginBottom: '20px' }}>Fraud Rate Overview</h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '20px', marginBottom: '30px' }}>
                <div style={{ fontSize: '3rem', fontWeight: 'bold', color: '#FF4B4B' }}>{fraudRate}%</div>
                <div style={{ color: '#A0AABF' }}>of recent transactions<br/>were blocked</div>
              </div>
              <div style={{ width: '100%', height: '20px', background: '#1E232E', borderRadius: '10px', overflow: 'hidden', display: 'flex' }}>
                <div style={{ width: `${fraudRate}%`, background: '#FF4B4B' }}></div>
                <div style={{ width: `${100 - fraudRate}%`, background: '#2ca02c' }}></div>
              </div>
            </div>

            {role === 'admin' && driftData && (
              <div className="dashboard-card" style={{ borderColor: driftData.drift_detected ? '#FF4B4B' : '#2ca02c' }}>
                <h3 style={{ margin: '0 0 15px 0', color: driftData.drift_detected ? '#FF4B4B' : '#2ca02c' }}>Model Drift Monitor</h3>
                <p style={{ color: '#FAFAFA', marginBottom: '15px' }}>{driftData.message}</p>
                {driftData.recent_average_amount !== undefined && (
                  <div style={{ display: 'flex', gap: '20px', fontSize: '0.9rem', color: '#A0AABF' }}>
                    <span>Recent Avg: <strong>${driftData.recent_average_amount}</strong></span>
                    <span>Baseline Avg: <strong>${driftData.baseline_average_amount}</strong></span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* --- TAB: ADMIN SETTINGS --- (Same as before) */}
        {activeTab === 'admin' && role === 'admin' && (
          <div className="dashboard-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h3 style={{ color: '#FAFAFA', margin: 0 }}>MLOps Model Management</h3>
              <button onClick={triggerRetrain} disabled={retraining} className="primary-btn" style={{ width: 'auto', padding: '10px 20px', fontSize: '0.9rem' }}>
                {retraining ? 'Retraining...' : '🔄 Retrain AI Model'}
              </button>
            </div>
            
            <h4 style={{ color: '#A0AABF', marginBottom: '10px' }}>Immutable Audit Trail (Regulatory Compliance)</h4>
            <div style={{ maxHeight: '400px', overflowY: 'auto', background: '#0E1117', padding: '15px', borderRadius: '8px', border: '1px solid #2E3A4E' }}>
              {auditLogs.length > 0 ? auditLogs.map((log, i) => (
                <div key={i} style={{ fontSize: '0.9rem', borderBottom: '1px solid #1E232E', paddingBottom: '10px', marginBottom: '10px', color: '#FAFAFA' }}>
                  <span style={{ color: log.prediction === 'BLOCK' ? '#FF4B4B' : log.prediction === 'STEP-UP AUTH' ? '#FFB800' : '#2ca02c', fontWeight: 'bold' }}>{log.prediction}</span> - ${log.amount} ({log.card_network}: {log.card_number}) <br/>
                  <span style={{ color: '#A0AABF', fontSize: '0.8rem' }}>Model: v{log.model_version} | {new Date(log.timestamp).toLocaleString()}</span>
                </div>
              )) : <p style={{ color: '#A0AABF' }}>No audit logs yet. Run a prediction!</p>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;