import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart3, MessageSquare, Send, CheckCircle, 
  AlertTriangle, ShieldCheck, ChevronDown, ChevronUp,
  Activity, Users, Zap, Info, Layers, Lightbulb, Target,
  Gauge, BookOpen
} from 'lucide-react';
import { 
  Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer 
} from 'recharts';
import Dashboard from './Dashboard';

const API_BASE = 'http://localhost:8000';

const App = () => {
  const [input, setInput] = useState('');
  const [lastResult, setLastResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
  const [expandedTip, setExpandedTip] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await axios.get(`${API_BASE}/history`);
      setHistory(res.data);
    } catch (err) {
      console.error("History fetch error:", err);
    }
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/analyze`, { user_input: input });
      setLastResult(res.data);
      fetchHistory(); 
      setInput('');
    } catch (err) {
      alert("Analysis failed. Refresh or check backend status.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getRadarData = (scores) => {
    if (!scores) return [];
    return Object.entries(scores).map(([key, value]) => ({
      subject: key.replace('_', ' ').toUpperCase(),
      A: value * 100,
    }));
  };

  const totalAnalyzed = history.length;
  const highRiskCount = history.filter(h => h.toxicity_score > 0.8).length;
  const avgToxicity = history.length > 0 
    ? (history.reduce((acc, curr) => acc + curr.toxicity_score, 0) / history.length * 100).toFixed(1)
    : 0;

  const severityTiers = [
    { 
      name: "Tier 1: Healthy", range: "0-15%", desc: "Maintain professional tone.",
      strategy: "Everything looks great. The AI will preserve your professional tone while ensuring clarity.",
      benchmark: "Thank you for your support, I appreciate it."
    },
    { 
      name: "Tier 2: Low-Risk", range: "15-30%", desc: "Refine minor harshness.",
      strategy: "Minor polishing applied. Softening subtle edges for better recipient reception.",
      benchmark: "This could be improved further to meet expectations."
    },
    { 
      name: "Tier 3: Noticeable", range: "30-45%", desc: "Neutralize passive-aggression.",
      strategy: "Diplomacy mode active. Identifying and neutralizing passive-aggressive indicators.",
      benchmark: "I think there might have been an issue; let’s review it together."
    },
    { 
      name: "Tier 4: Moderate", range: "45-60%", desc: "Reconstruct direct insults.",
      strategy: "Construction mode. Rebuilding direct critiques into constructive, project-focused feedback.",
      benchmark: "There seems to be a recurring issue; let’s work on improving it."
    },
    { 
      name: "Tier 5: High Risk", range: "60-75%", desc: "Transform hostile intent.",
      strategy: "Restorative shift. Transforming hostile intent into collaborative, supportive guidance.",
      benchmark: "I believe we can improve this by refining a few aspects together."
    },
    { 
      name: "Tier 6: Severe", range: "75-90%", desc: "Deep de-escalation grace.",
      strategy: "Deep De-escalation. Implementing extreme professional grace for severe interpersonal friction.",
      benchmark: "Let’s focus on improving collaboration and addressing the challenges constructively."
    },
    { 
      name: "Tier 7: Critical", range: "90-100%", desc: "Masterful Restoration.",
      strategy: "Masterful Restoration. Deep neutralization of extreme hostility into sincere, humble cooperation.",
      benchmark: "I sincerely apologize for any frustration caused and would like to work together more positively moving forward."
    }
  ];

  const getTierForScore = (score) => {
    if (score <= 0.15) return severityTiers[0];
    if (score <= 0.30) return severityTiers[1];
    if (score <= 0.45) return severityTiers[2];
    if (score <= 0.60) return severityTiers[3];
    if (score <= 0.75) return severityTiers[4];
    if (score <= 0.90) return severityTiers[5];
    return severityTiers[6];
  };

  const currentTier = lastResult ? getTierForScore(lastResult.toxicity_score) : null;

  return (
    <div className="container">
      <header className="header-aesthetic">
        <h1>Toxic Message Analyzer</h1>
        <p>Pro-Grade 7-Tier Severity Engine & Restorative AI Strategy</p>
      </header>

      <div className="info-cards-row">
        <div className="info-card">
          <div className="info-card-icon"><MessageSquare size={24} /></div>
          <div className="info-card-content">
            <h4>Total Scanned</h4>
            <p>{totalAnalyzed}</p>
          </div>
        </div>
        <div className="info-card" style={{ borderColor: highRiskCount > 0 ? '#ef4444' : '#e2e8f0' }}>
          <div className="info-card-icon" style={{ color: '#ef4444' }}><AlertTriangle size={24} /></div>
          <div className="info-card-content">
            <h4>High Risk</h4>
            <p>{highRiskCount}</p>
          </div>
        </div>
        <div className="info-card">
          <div className="info-card-icon" style={{ color: '#10b981' }}><Zap size={24} /></div>
          <div className="info-card-content">
            <h4>Avg Severity</h4>
            <p>{avgToxicity}%</p>
          </div>
        </div>
      </div>

      <div className="main-layout">
        <div className="analysis-column">
          <main className="card" style={{ marginBottom: '2rem' }}>
            <form onSubmit={handleAnalyze} className="input-group">
              <div className="section-label">Input Feedback Analysis</div>
              <textarea
                placeholder="Analyze feedback for restorative professional rephrasing..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.85rem', color: '#64748b', fontWeight: 500 }}>{input.length} characters</span>
                <button type="submit" className="btn-primary" disabled={loading || !input}>
                  {loading ? 'Calibrating...' : 'Execute Analysis'}
                </button>
              </div>
            </form>
          </main>

          <AnimatePresence>
            {lastResult && (
              <motion.section initial={{ opacity: 0, scale: 0.98, y: 10 }} animate={{ opacity: 1, scale: 1, y: 0 }} style={{ marginBottom: '2rem' }}>
                <div className="card">
                  <div className="result-card">
                    <div style={{ padding: '0.5rem', borderRight: '1px solid #f1f5f9' }}>
                      <div className="section-label">Level Segmentation</div>
                      <div className={`badge ${lastResult.prediction_label === 'Toxic' ? 'toxic' : 'safe'}`}>
                        {lastResult.prediction_label}
                      </div>

                      <div style={{ 
                        marginTop: '1rem', padding: '0.75rem', background: '#f8fafc', borderRadius: '0.75rem', border: '1px solid #e2e8f0',
                        textAlign: 'center'
                      }}>
                        <div style={{ fontSize: '0.65rem', fontWeight: 800, textTransform: 'uppercase', color: '#64748b', letterSpacing: '0.1em' }}>Toxicity Intensity</div>
                        <div style={{ fontSize: '1.75rem', fontWeight: 900, color: lastResult.toxicity_score > 0.6 ? '#ef4444' : '#4f46e5' }}>
                          {(lastResult.toxicity_score * 100).toFixed(0)}%
                        </div>
                      </div>

                      <div style={{ height: '180px', marginTop: '1rem' }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart cx="50%" cy="50%" outerRadius="70%" data={getRadarData(lastResult.segmentation_scores)}>
                            <PolarGrid stroke="#e2e8f0" />
                            <PolarAngleAxis dataKey="subject" fontSize={9} fontWeight="bold" stroke="#64748b" />
                            <Radar name="Score" dataKey="A" stroke="#4f46e5" fill="#4f46e5" fillOpacity={0.4} />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div style={{ padding: '0.5rem' }}>
                      <div className="section-label">Restorative AI Output</div>
                      <div style={{ 
                        background: '#f8fafc', padding: '1.5rem', borderRadius: '1.25rem', border: '1px solid #e2e8f0',
                        minHeight: '130px', display: 'flex', alignItems: 'center'
                      }}>
                        <p style={{ fontSize: '1.1rem', fontWeight: 600, color: '#1e293b', lineHeight: 1.5 }}>
                          {lastResult.rephrased_output || "Status: Optimal. No reconstruction required."}
                        </p>
                      </div>

                      {/* NEW: PREDEFINED EXPERT REPHRASE BOX */}
                      <div className="section-label" style={{ marginTop: '1.5rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <BookOpen size={14} /> Expert Benchmark Rephrase
                      </div>
                      <div style={{ 
                        background: '#eff6ff', padding: '1.5rem', borderRadius: '1.25rem', border: '1px solid #bfdbfe',
                        minHeight: '100px', display: 'flex', alignItems: 'center'
                      }}>
                        <p style={{ fontSize: '1rem', fontWeight: 600, color: '#1e40af', lineHeight: 1.5, fontStyle: 'italic' }}>
                          "{currentTier?.benchmark}"
                        </p>
                      </div>

                      <div style={{ marginTop: '1.25rem', display: 'flex', gap: '8px', alignItems: 'center' }}>
                        <ShieldCheck size={18} className="text-primary" />
                        <p className="text-muted" style={{ fontSize: '0.8rem', fontWeight: 600 }}>
                          Calibrated via Tier-7 Expert Severity Engine.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.section>
            )}
          </AnimatePresence>
        </div>

        <aside className="tips-sidebar">
          <div className="tips-title">
            <Target size={18} className="text-primary" />
            Strategic Guide: 7 Tiers
          </div>
          <div style={{ maxHeight: '600px', overflowY: 'auto', paddingRight: '4px' }}>
            {severityTiers.map((tier, idx) => (
              <div 
                key={idx} 
                className="tip-item" 
                style={{ cursor: 'pointer' }}
                onClick={() => setExpandedTip(expandedTip === idx ? null : idx)}
              >
                <div className="tip-header">
                  <span className="tip-name">{tier.name}</span>
                  <span className="tip-range">{tier.range}</span>
                </div>
                <p className="tip-desc">{tier.desc}</p>
                
                <AnimatePresence>
                  {expandedTip === idx && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} style={{ overflow: 'hidden', marginTop: '0.5rem' }}>
                      <div style={{ fontSize: '0.75rem', padding: '1rem', background: '#f8fafc', color: '#64748b', borderRadius: '0.75rem', border: '1px solid #e2e8f0', lineHeight: 1.5, fontWeight: 500 }}>
                        <span style={{ color: '#4f46e5', fontWeight: 800 }}>RESTORATIVE STRATEGY:</span>
                        <br />
                        <span style={{ color: '#1e293b' }}>{tier.strategy}</span>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ))}
          </div>
          <div style={{ marginTop: '1rem', padding: '0.75rem', background: '#f1f5f9', borderRadius: '0.75rem', fontSize: '0.65rem', color: '#64748b', fontWeight: 700 }}>
            <Lightbulb size={12} style={{ display: 'inline', marginRight: '4px' }} />
            Click any tier to view the restorative strategy for that level.
          </div>
        </aside>
      </div>

      <section className="dashboard-section" style={{ marginTop: '3rem' }}>
        <div className="dashboard-toggle text-center">
          <button onClick={() => setShowDashboard(!showDashboard)} className="btn-primary" style={{ backgroundColor: 'white', color: '#4f46e5', border: '1px solid #e2e8f0' }}>
            {showDashboard ? <ChevronUp size={20} /> : <BarChart3 size={20} />}
            <span style={{ marginLeft: '0.5rem' }}>{showDashboard ? 'Hide Historical Data' : 'Explore Detailed Analytics Dashboard'}</span>
          </button>
        </div>
        {showDashboard && <Dashboard messages={history} />}
      </section>

      <footer style={{ textAlign: 'center', padding: '4rem 0', color: '#94a3b8', fontSize: '0.7rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.15em' }}>
        7-Tier Analytical Engine v2.8 | Multi-Model Benchmark Verification
      </footer>
    </div>
  );
};

export default App;
