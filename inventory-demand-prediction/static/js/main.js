// Theme toggle (light / dark), persisted per browser session via localStorage
// is NOT used here because artifacts/some environments disallow it -- but
// this is a real Flask app served from disk, so localStorage is fine here.
(function () {
  const root = document.documentElement;
  const stored = localStorage.getItem("idp_theme");
  if (stored) root.setAttribute("data-theme", stored);

  window.toggleTheme = function () {
    const current = root.getAttribute("data-theme") === "dark" ? "dark" : "light";
    const next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("idp_theme", next);
  };
})();

// Generic helper to render a Chart.js chart from a JSON API endpoint
function renderChartFromApi(canvasId, apiUrl, type, color) {
  fetch(apiUrl)
    .then((res) => res.json())
    .then((data) => {
      const ctx = document.getElementById(canvasId);
      if (!ctx) return;
      new Chart(ctx, {
        type: type,
        data: {
          labels: data.labels,
          datasets: [{
            label: ctx.dataset.label || "",
            data: data.values,
            backgroundColor: type === "bar" ? color + "cc" : color + "33",
            borderColor: color,
            borderWidth: 2,
            tension: 0.35,
            fill: type === "line",
            pointRadius: type === "line" ? 2 : 0,
          }],
        },
        options: {
          responsive: true,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false } },
            y: { grid: { color: "rgba(148,163,184,0.15)" } },
          },
        },
      });
    })
    .catch((err) => console.error("Chart load failed:", apiUrl, err));
}
