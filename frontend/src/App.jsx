import React, { useEffect, useState, useRef } from "react";
import { MapContainer, TileLayer, Marker, Popup, Circle } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import clsx from "clsx";
import MiniTimeline from "./MiniTimeline";

// Fix Leaflet default icon paths for Vite/Esm
import iconUrl from "leaflet/dist/images/marker-icon.png";
import iconRetinaUrl from "leaflet/dist/images/marker-icon-2x.png";
import iconShadowUrl from "leaflet/dist/images/marker-shadow.png";
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl,
  iconUrl,
  shadowUrl: iconShadowUrl,
});

/* add after imports */
function Legend() {
  return (
    <div className="map-legend pointer-events-auto select-none">
      <div className="legend-title">Legend</div>
      <div className="legend-row"><span className="legend-dot" style={{ background: "#ef4444" }} /> High (≥ 0.90)</div>
      <div className="legend-row"><span className="legend-dot" style={{ background: "#f97316" }} /> Medium (0.80–0.89)</div>
      <div className="legend-row"><span className="legend-dot" style={{ background: "#f59e0b" }} /> Low (0.50–0.79)</div>
      <div className="legend-row"><span className="legend-dot" style={{ background: "#10b981" }} /> Very low (&lt; 0.50)</div>
      <div className="legend-note text-xs text-slate-500 mt-1">Circle size ≈ cluster count</div>
    </div>
  );
}

export default function App() {
  const [clusters, setClusters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [center, setCenter] = useState([20, 0]);
  const [zoom, setZoom] = useState(2);
  const [selected, setSelected] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [error, setError] = useState(null);

  // map ref so we can call fitBounds
  const mapRef = useRef(null);

  useEffect(() => {
    fetchClusters();
    const id = setInterval(fetchClusters, 5 * 60 * 1000); // poll every 5 minutes
    return () => clearInterval(id);
  }, []);

  // whenever clusters update, fit bounds
    // whenever clusters update, animate to show them nicely
  useEffect(() => {
    if (!mapRef.current) return;
    if (clusters.length === 0) return;

    try {
      const latlngs = clusters.map((c) => [c.mean_lat, c.mean_lon]);

      // If we only have one cluster, smoothly fly to it (with a modest zoom)
      if (latlngs.length === 1) {
        const [lat, lon] = latlngs[0];
        // flyTo takes (latlng, zoom, options)
        mapRef.current.flyTo([lat, lon], 6, { duration: 1.2 });
        return;
      }

      // For multiple clusters, compute bounds and fly to fit them with padding.
      // flyToBounds animates; maxZoom prevents zooming in too far.
      const bounds = L.latLngBounds(latlngs);
      // If the computed max zoom would be too high, we can restrict after fitting.
      mapRef.current.flyToBounds(bounds, { padding: [80, 80] });
      // after a short delay, ensure we don't exceed max zoom level
      setTimeout(() => {
        const currentZoom = mapRef.current.getZoom();
        const maxZoom = 8;
        if (currentZoom > maxZoom) mapRef.current.setZoom(maxZoom);
      }, 600);
    } catch (err) {
      // ignore non-fatal map errors
      console.warn("fit/fly bounds error:", err);
    }
  }, [clusters]);

  async function fetchClusters() {
    setLoading(true);
    setError(null);
    try {
      // If your backend is at a different origin, change to http://127.0.0.1:8000/api/clusters
      const res = await fetch("/api/clusters");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setClusters(data);
      if (data.length > 0) {
        setCenter([data[0].mean_lat, data[0].mean_lon]);
        setZoom(4);
      }
      setLastUpdated(new Date().toISOString());
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to fetch");
      setClusters([]);
    }
    setLoading(false);
  }

    function handleMarkerClick(cluster) {
    setSelected(cluster);
    // smooth fly to the cluster (if map ready)
    if (mapRef.current) {
      mapRef.current.flyTo([cluster.mean_lat, cluster.mean_lon], 7, { duration: 1.0 });
    } else {
      setCenter([cluster.mean_lat, cluster.mean_lon]);
      setZoom(7);
    }
  }

  // color scale
  function getMarkerColor(p) {
    if (p >= 0.9) return "#ef4444"; // red
    if (p >= 0.8) return "#f97316"; // orange
    if (p >= 0.5) return "#f59e0b"; // amber
    return "#10b981"; // green
  }

  // create a small divIcon with colored circle
  function createColorIcon(color, size = 16) {
    return L.divIcon({
      className: "custom-div-icon",
      html: `<span style="
        display:inline-block;
        width:${size}px;
        height:${size}px;
        background:${color};
        border-radius:50%;
        border:2px solid white;
        box-shadow:0 0 6px rgba(0,0,0,0.25);
      "></span>`,
      iconSize: [size + 4, size + 4],
      iconAnchor: [size / 2 + 2, size + 2],
      popupAnchor: [0, -size - 8],
    });
  }

  // circle radius (meters) from count
  function circleRadius(count) {
    // base radius 4km, + scale by count
    return Math.min(60000, 4000 + count * 1500);
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <header className="bg-white shadow p-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold">WildGuard — Wildfire Dashboard</h1>
        <div className="text-sm text-slate-500">Last updated: {lastUpdated ? new Date(lastUpdated).toLocaleString() : "—"}</div>
      </header>

      <main className="p-4 grid grid-cols-12 gap-4">
        <section className="col-span-8 bg-white rounded-lg shadow p-2 relative">
          {/* Loading overlay */}
          {/* Legend overlay */}
          <Legend />

          <div className="h-[70vh] rounded">
            <MapContainer
              center={center}
              zoom={zoom}
              style={{ height: "100%", width: "100%" }}
              whenCreated={(mapInstance) => (mapRef.current = mapInstance)}
            >
            <Legend />
              <TileLayer attribution='© OpenStreetMap contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

              {/* Render markers and circles */}
              {clusters.map((c) => {
                const color = getMarkerColor(c.max_prob);
                // size marker slightly by probability or keep constant
                const icon = createColorIcon(color, 16);
                return (
                  <React.Fragment key={c.cluster}>
                    <Marker
                      position={[c.mean_lat, c.mean_lon]}
                      icon={icon}
                      eventHandlers={{ click: () => handleMarkerClick(c) }}
                    >
                      <Popup>
                        <div className="text-sm">
                          <strong>Cluster {c.cluster}</strong>
                          <div>Count: {c.count}</div>
                          <div>Max Prob: {c.max_prob.toFixed(3)}</div>
                          <div>
                            Time: {c.first_time ? new Date(c.first_time).toLocaleString() : "—"} →{" "}
                            {c.last_time ? new Date(c.last_time).toLocaleString() : "—"}
                          </div>
                          <div>
                            <a
                              className="text-blue-600 underline"
                              target="_blank"
                              rel="noreferrer"
                              href={`https://www.google.com/maps/search/?api=1&query=${c.mean_lat},${c.mean_lon}`}
                            >
                              Open in Maps
                            </a>
                          </div>
                        </div>
                      </Popup>
                    </Marker>

                    {/* colored circle (translucent) */}
                    <Circle
                      center={[c.mean_lat, c.mean_lon]}
                      radius={circleRadius(c.count)}
                      pathOptions={{ color, fillColor: color, fillOpacity: 0.12, weight: 2 }}
                    />
                  </React.Fragment>
                );
              })}
            </MapContainer>
          </div>
        </section>

        <aside className="col-span-4 bg-white rounded-lg shadow p-4 overflow-auto" style={{ maxHeight: "70vh" }}>
          <div className="mb-2 flex items-center justify-between">
            <h2 className="font-semibold">Top Clusters</h2>
            <button onClick={fetchClusters} className="text-sm text-blue-600">
              Refresh
            </button>
          </div>

          {loading && (
  <div className="absolute inset-0 z-40 flex items-center justify-center bg-white/70 rounded">
    <div className="flex flex-col items-center">
      <div className="spinner" />
      <div className="spinner-label">Loading clusters…</div>
    </div>
  </div>
)} 
        <MiniTimeline clusters={clusters} />
        </aside>
      </main>

      <footer className="p-4 text-center text-xs text-slate-500">
        WildGuard • Real-time wildfire detection • Model: RandomForest • Data: NASA FIRMS
      </footer>
    </div>
  );
}