// MiniTimeline.jsx
import React, { useMemo } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  TimeScale,
  Title,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Title);

function formatDateISO(date) {
  const d = new Date(date);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/**
 * MiniTimeline
 * props:
 *  - clusters: array of cluster objects with first_time / last_time (ISO strings)
 * Renders a 7-day bar chart of detection counts (clusters assigned to day of first_time)
 */
export default function MiniTimeline({ clusters = [] }) {
  const { labels, counts } = useMemo(() => {
    // build last 7 days labels (today included)
    const days = [];
    const countsMap = {};
    const today = new Date();
    for (let i = 6; i >= 0; --i) {
      const d = new Date(today);
      d.setDate(today.getDate() - i);
      const iso = d.toISOString().slice(0, 10); // YYYY-MM-DD
      days.push(d);
      countsMap[iso] = 0;
    }

    // increment by cluster.first_time (if present), else use last_time
    clusters.forEach((c) => {
      const ts = c.first_time || c.last_time || null;
      if (!ts) return;
      const day = new Date(ts).toISOString().slice(0, 10);
      if (day in countsMap) countsMap[day] += c.count || 1;
    });

    const labels = days.map(formatDateISO);
    const counts = days.map((d) => countsMap[d.toISOString().slice(0, 10)] || 0);
    return { labels, counts };
  }, [clusters]);

  const data = {
    labels,
    datasets: [
      {
        label: "Detections",
        data: counts,
        backgroundColor: counts.map((v) => (v > 0 ? "rgba(239,68,68,0.9)" : "rgba(148,163,184,0.25)")),
        borderRadius: 6,
        barThickness: 18,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: (ctx) => `${ctx.parsed.y} detections` } },
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: "#0f172a" } },
      y: {
        beginAtZero: true,
        ticks: { precision: 0, color: "#0f172a" },
        grid: { color: "rgba(2,6,23,0.04)" },
      },
    },
  };

  return (
    <div className="mini-timeline bg-white rounded p-3 mt-4 shadow-sm" style={{ height: 160 }}>
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-medium">Recent detections (7d)</div>
        <div className="text-xs text-slate-500">counts</div>
      </div>
      <Bar data={data} options={options} />
    </div>
  );
}