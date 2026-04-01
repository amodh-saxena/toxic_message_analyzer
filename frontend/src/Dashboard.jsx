import React from 'react';
import { 
  PieChart, Pie, Cell, 
  AreaChart, Area, 
  BarChart, Bar,
  LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, Legend
} from 'recharts';

const Dashboard = ({ messages }) => {
  // 1. Toxicity Distribution
  const toxicCount = messages.filter(m => m.prediction_label === 'Toxic').length;
  const safeCount = messages.filter(m => m.prediction_label === 'Non-toxic').length;
  const pieData = [
    { name: 'Toxic', value: toxicCount },
    { name: 'Safe', value: safeCount },
  ];
  const PIE_COLORS = ['#ef4444', '#10b981'];

  // 2. Category Trend (Area Chart)
  const categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'];
  const trendData = messages.slice(0, 20).reverse().map((m, idx) => {
    let scores = {};
    try {
      scores = typeof m.segmentation_scores === 'string' 
        ? JSON.parse(m.segmentation_scores) 
        : m.segmentation_scores || {};
    } catch(e) { scores = {}; }
    return { name: idx + 1, ...scores, avg: m.toxicity_score };
  });

  // 3. Category Prevalence (Bar Chart)
  const prevalenceData = categories.map(cat => {
    const total = messages.reduce((acc, current) => {
      let scores = {};
      try {
        scores = typeof current.segmentation_scores === 'string' 
          ? JSON.parse(current.segmentation_scores) 
          : current.segmentation_scores || {};
      } catch(e) { scores = {}; }
      return acc + (scores[cat] || 0);
    }, 0);
    return { name: cat.replace('_', ' ').toUpperCase(), value: total / (messages.length || 1) };
  });

  // Colors for multi-label segments
  const CAT_COLORS = {
    toxic: '#ef4444',
    severe_toxic: '#991b1b',
    obscene: '#f59e0b',
    threat: '#7c3aed',
    insult: '#2563eb',
    identity_hate: '#0891b2'
  };

  return (
    <div className="dashboard-4-grid">
      
      {/* 1. PIE DISTRIBUTION */}
      <div className="chart-card shadow-sm">
        <div className="section-label" style={{ fontSize: '0.7rem' }}>Global Sentiment Profile (%)</div>
        <ResponsiveContainer width="100%" height="90%">
          <PieChart>
            <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
              {pieData.map((entry, index) => <Cell key={`cell-${index}`} fill={PIE_COLORS[index]} />)}
            </Pie>
            <Tooltip />
            <Legend verticalAlign="bottom" height={36}/>
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* 2. CATEGORY TRENDS (AREA) */}
      <div className="chart-card shadow-sm">
        <div className="section-label" style={{ fontSize: '0.7rem' }}>Layered Toxicity Volume</div>
        <ResponsiveContainer width="100%" height="90%">
          <AreaChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis dataKey="name" fontSize={10} hide />
            <YAxis fontSize={10} axisLine={false} tickLine={false} />
            <Tooltip />
            {categories.map(cat => (
              <Area key={cat} type="monotone" dataKey={cat} stackId="1" stroke={CAT_COLORS[cat]} fill={CAT_COLORS[cat]} fillOpacity={0.4} />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* 3. CATEGORY PREVALENCE (BAR) */}
      <div className="chart-card shadow-sm">
        <div className="section-label" style={{ fontSize: '0.7rem' }}>Segment Prevalence Hierarchy</div>
        <ResponsiveContainer width="100%" height="90%">
          <BarChart data={prevalenceData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
            <XAxis type="number" fontSize={10} domain={[0, 1]} hide />
            <YAxis dataKey="name" type="category" fontSize={9} width={80} tick={{ fontWeight: 600, fill: '#64748b' }} />
            <Tooltip />
            <Bar dataKey="value" fill="#4f46e5" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* 4. SEVERITY TREND (LINE) */}
      <div className="chart-card shadow-sm">
        <div className="section-label" style={{ fontSize: '0.7rem' }}>Average Severity Index Trend</div>
        <ResponsiveContainer width="100%" height="90%">
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="name" hide />
            <YAxis fontSize={10} domain={[0, 1]} />
            <Tooltip />
            <Line type="monotone" dataKey="avg" stroke="#4f46e5" strokeWidth={3} dot={{ r: 4, fill: '#4f46e5' }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

    </div>
  );
};

export default Dashboard;
