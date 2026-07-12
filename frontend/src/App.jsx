import { useState } from 'react';
import { RadialBarChart, RadialBar, PolarAngleAxis, BarChart, Bar, XAxis, YAxis, Cell, ReferenceLine, ResponsiveContainer } from 'recharts';
import './App.css';

function App() {
  const [selectedPlayer, setSelectedPlayer] = useState("Star Striker");
  const [fatigue, setFatigue] = useState(0.5);
  const [gkBias, setGkBias] = useState("Left");
  const [shotDir, setShotDir] = useState("Left");
  
  const [agents, setAgents] = useState(null);
  const [report, setReport] = useState(null);
  const [cvFeatures, setCvFeatures] = useState(null);
  const [liveSim, setLiveSim] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [scoreA, setScoreA] = useState(0);
  const [scoreB, setScoreB] = useState(0);
  const [kicksTaken, setKicksTaken] = useState(0);

  const fetchToken = async () => {
    const tokenRes = await fetch('http://127.0.0.1:8000/token');
    const tokenData = await tokenRes.json();
    return tokenData.access_token;
  };

  const getPrediction = async () => {
    setLoading(true); setError(null);
    const playerSkills = { "Star Striker": 0.85, "Solid Mid": 0.75, "Defender": 0.65, "Winger": 0.70, "Veteran": 0.80 };
    const payload = {
      player_id: 101, player_penalty_conversion_rate: playerSkills[selectedPlayer],
      career_penalty_attempts: 25, recent_penalty_form: 0.80, is_shootout: 1,
      match_stage: "Final", minute_of_match: 120, fatigue_index: parseFloat(fatigue),
      run_up_style: "Straight", pause_before_shot: 0, 
      cv_body_lean_angle: cvFeatures ? cvFeatures.cv_body_lean_angle : 15.5, 
      cv_run_up_speed: cvFeatures ? cvFeatures.cv_run_up_speed : 8.2,
      goalkeeper_id: 501, goalkeeper_penalty_save_rate: 0.22, goalkeeper_diving_bias: gkBias,
      keeper_experience_level: 4, shooter_vs_keeper_history: 0, psychological_advantage_index: 80,
      shot_direction: shotDir
    };

    try {
      const token = await fetchToken();
      const response = await fetch('http://127.0.0.1:8000/predict/agents', {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error(`API Error: ${response.status}`);
      const data = await response.json();
      setAgents(data);
      setReport(null);
    } catch (err) { setError("Failed to connect to AI Engine."); }
    setLoading(false);
  };

  const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setLoading(true); setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const token = await fetchToken();
      const res = await fetch('http://127.0.0.1:8000/analyze/video', {
        method: 'POST', headers: { 'Authorization': `Bearer ${token}` }, body: formData
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Video Upload Failed');
      }
      const data = await res.json();
      setCvFeatures(data);
    } catch (e) { setError(e.message); }
    setLoading(false);
  };

  const generateCoachReport = async () => {
    if (!agents) return;
    setLoading(true);
    try {
      const token = await fetchToken();
      const res = await fetch('http://127.0.0.1:8000/generate/report', {
        method: 'POST', 
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ 
          xg: agents.interaction_outcome.probability_of_goal, 
          reasons: agents.ai_explanation,
          cv_lean: cvFeatures ? cvFeatures.cv_body_lean_angle : 15.5,
          fatigue: parseFloat(fatigue)
        })
      });
      const data = await res.json();
      setReport(data.tactical_report);
    } catch (e) { setError('Failed to generate report.'); }
    setLoading(false);
  };

  const handleLiveKick = async (team, scored) => {
    let newScoreA = scoreA, newScoreB = scoreB, newKicks = kicksTaken + 1;
    if (team === 'A' && scored) newScoreA++;
    if (team === 'B' && scored) newScoreB++;
    
    setScoreA(newScoreA); setScoreB(newScoreB); setKicksTaken(newKicks);
    
    try {
      const token = await fetchToken();
      const res = await fetch('http://127.0.0.1:8000/simulate/live', {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ score_a: newScoreA, score_b: newScoreB, kicks_taken: newKicks })
      });
      const data = await res.json();
      setLiveSim(data);
    } catch (e) { setError('Live simulation failed.'); }
  };

  const resetLiveTracker = () => {
    setScoreA(0); setScoreB(0); setKicksTaken(0); setLiveSim(null);
  };

  const probValue = agents ? agents.interaction_outcome.probability_of_goal : 0;
  const barColor = probValue > 0.7 ? '#22c55e' : probValue > 0.4 ? '#f59e0b' : '#ef4444';

  // Recharts Data Formats
  const xgGaugeData = [{ name: 'xG', value: probValue * 100, fill: barColor }];
  
  const shapChartData = agents ? agents.ai_explanation.map(r => ({
    name: r.feature.replace(/_/g, ' '),
    value: r.direction === 'Increased' ? r.impact : -r.impact
  })) : [];

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>⚔️ DuelTactix AI</h1>
        <p>Penalty Duel Intelligence & Live Simulation Engine</p>
      </header>

      <div className="main-grid">
        {/* LEFT: Controls & Video Upload */}
        <div className="visual-panel">
          <div className="controls-panel" style={{border: 'none', padding: '0'}}>
            <div className="input-group">
              <label>Select Striker</label>
              <select value={selectedPlayer} onChange={(e) => setSelectedPlayer(e.target.value)} className="dropdown">
                <option value="Star Striker">Star Striker (85%)</option>
                <option value="Veteran">Veteran (80%)</option>
                <option value="Solid Mid">Solid Mid (75%)</option>
                <option value="Winger">Winger (70%)</option>
                <option value="Defender">Defender (65%)</option>
              </select>
            </div>

            <div className="input-group">
              <label>Fatigue Index</label>
              <div className="value-display">{parseFloat(fatigue).toFixed(2)}</div>
              <input type="range" min="0" max="1" step="0.05" value={fatigue} onChange={(e) => setFatigue(e.target.value)} className="slider"/>
            </div>
            
            <div className="grid-2-cols">
              <div className="input-group">
                <label>Shot Direction</label>
                <select value={shotDir} onChange={(e) => setShotDir(e.target.value)} className="dropdown">
                  <option value="Left">Left</option><option value="Center">Center</option><option value="Right">Right</option>
                </select>
              </div>
              <div className="input-group">
                <label>GK Bias</label>
                <select value={gkBias} onChange={(e) => setGkBias(e.target.value)} className="dropdown">
                  <option value="Left">Left</option><option value="Right">Right</option><option value="Center">Center</option>
                </select>
              </div>
            </div>

            <button onClick={getPrediction} disabled={loading} className="predict-btn">
              {loading ? 'Running Engine...' : 'Analyze Penalty Duel'}
            </button>
          </div>

          <div className="video-upload-section">
            <h3>📹 CV Video Analysis</h3>
            <input type="file" accept="video/*" onChange={handleVideoUpload} className="file-input"/>
            {cvFeatures && (
              <div className="cv-results">
                <span>Lean Angle: {cvFeatures.cv_body_lean_angle}°</span>
                <span>Run-up Speed: {cvFeatures.cv_run_up_speed} m/s</span>
              </div>
            )}
          </div>
          {error && <div className="result-card" style={{borderColor: 'red', marginTop: '20px'}}><p style={{color: 'red'}}>{error}</p></div>}
        </div>

        {/* RIGHT: Prediction, Charts & GenAI */}
        <div className="visual-panel">
          {agents ? (
            <div className="result-card">
              <h2>🤖 Prediction Panel</h2>
              <div className="reasons-list">
                <div className="reason-item positive"><span className="arrow">⚽</span><span className="reason-text">Striker: {shotDir}</span></div>
                <div className="reason-item negative"><span className="arrow">🧤</span><span className="reason-text">GK: {agents.gk_prediction.action}</span></div>
              </div>

              {/* xG RADIAL GAUGE */}
              <div className="chart-container">
                <h3>Expected Outcome (xG)</h3>
                <ResponsiveContainer width="100%" height={180}>
                  <RadialBarChart innerRadius="70%" outerRadius="100%" data={xgGaugeData} startAngle={90} endAngle={-270}>
                    <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
                    <RadialBar background dataKey="value" cornerRadius={10} />
                    <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" className="gauge-text">
                      {agents.interaction_outcome.probability_as_percentage}
                    </text>
                  </RadialBarChart>
                </ResponsiveContainer>
              </div>

              {/* SHAP DIVERGING BAR CHART */}
              {shapChartData.length > 0 && (
                <div className="chart-container" style={{marginTop: '20px'}}>
                  <h3>🧠 SHAP Tactical Breakdown</h3>
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={shapChartData} layout="vertical">
                      <XAxis type="number" hide domain={[-0.5, 0.5]} />
                      <YAxis type="category" dataKey="name" width={100} stroke="#94a3b8" fontSize={12} />
                      <ReferenceLine x={0} stroke="#fff" />
                      <Bar dataKey="value" radius={[0, 5, 5, 0]}>
                        {shapChartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#22c55e' : '#ef4444'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              <button onClick={generateCoachReport} disabled={loading} className="predict-btn" style={{background: '#7c3aed', marginTop: '20px'}}>
                {loading ? 'Thinking...' : '🧠 Generate Tactical Report'}
              </button>
              
              {report && (
                <div className="genai-report">
                  <p>{report}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="result-card placeholder"><p>Select parameters and click "Analyze Penalty Duel".</p></div>
          )}
        </div>
      </div>

      {/* BOTTOM: LIVE TRACKER */}
      <div className="visual-panel" style={{marginTop: '30px'}}>
        <h2>📡 Live Shootout Momentum Tracker</h2>
        <p style={{color: '#94a3b8', marginBottom: '20px'}}>Click goals/isses to update win probability in real-time.</p>
        
        <div className="live-tracker-grid">
          <div className="team-controls">
            <h3>Team A</h3>
            <div className="score">{scoreA}</div>
            <div className="live-buttons">
              <button onClick={() => handleLiveKick('A', true)} className="live-btn goal">Goal</button>
              <button onClick={() => handleLiveKick('A', false)} className="live-btn miss">Miss</button>
            </div>
          </div>

          <div className="live-probs">
            {liveSim ? (
              <>
                <div className="prob-ring">
                  <span>A: {(liveSim.team_a_win_probability * 100).toFixed(1)}%</span>
                  <span>B: {(liveSim.team_b_win_probability * 100).toFixed(1)}%</span>
                </div>
                <button onClick={resetLiveTracker} className="reset-btn">Reset</button>
              </>
            ) : <p>Start tracking to see probabilities</p>}
          </div>

          <div className="team-controls">
            <h3>Team B</h3>
            <div className="score">{scoreB}</div>
            <div className="live-buttons">
              <button onClick={() => handleLiveKick('B', true)} className="live-btn goal">Goal</button>
              <button onClick={() => handleLiveKick('B', false)} className="live-btn miss">Miss</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;