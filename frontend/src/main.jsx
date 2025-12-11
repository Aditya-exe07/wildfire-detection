import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";             // <-- import Tailwind + Leaflet CSS here
import "leaflet/dist/leaflet.css"; // ensure leaflet css is loaded

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
