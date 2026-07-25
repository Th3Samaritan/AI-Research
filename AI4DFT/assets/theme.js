/* ============================================================
   AI4DFT — theme controller (shared by index / walkthrough / exam)
   - light is the default; dark is opt-in and remembered
   - respects the OS preference on first visit
   - exposes AI4DFT.colors() so <canvas> drawings stay theme-aware
   - fires a "themechange" event on window so canvases can redraw
   ============================================================ */
(function () {
  var KEY = "ai4dft-theme";
  var root = document.documentElement;

  function preferred() {
    var saved = null;
    try { saved = localStorage.getItem(KEY); } catch (e) {}
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark" : "light";
  }

  function apply(mode) {
    root.setAttribute("data-theme", mode);
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", mode === "dark" ? "#0B0E13" : "#FFFFFF");
    document.querySelectorAll("[data-theme-toggle]").forEach(function (b) {
      b.querySelector(".ico").textContent = mode === "dark" ? "☀" : "☾";
      b.querySelector(".lbl").textContent = mode === "dark" ? "Light" : "Dark";
      b.setAttribute("aria-label", "Switch to " + (mode === "dark" ? "light" : "dark") + " mode");
    });
    window.dispatchEvent(new CustomEvent("themechange", { detail: { mode: mode } }));
  }

  // set the attribute before first paint to avoid a flash
  root.setAttribute("data-theme", preferred());

  function toggle() {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    try { localStorage.setItem(KEY, next); } catch (e) {}
    apply(next);
  }

  function mountToggle() {
    var bar = document.querySelector(".topbar");
    if (!bar) {
      bar = document.createElement("div");
      bar.className = "topbar";
      document.body.appendChild(bar);
    }
    if (!bar.querySelector("[data-theme-toggle]")) {
      var b = document.createElement("button");
      b.className = "theme-toggle";
      b.setAttribute("data-theme-toggle", "");
      b.innerHTML = '<span class="ico">☾</span><span class="lbl">Dark</span>';
      b.addEventListener("click", toggle);
      bar.appendChild(b);
    }
    apply(root.getAttribute("data-theme"));
  }

  /* colours for canvas drawing, read live from the CSS custom properties */
  function colors() {
    var s = getComputedStyle(root);
    var g = function (n) { return s.getPropertyValue(n).trim(); };
    return {
      bg: g("--surface"), grid: g("--line"), line: g("--line-soft"),
      txt: g("--txt"), muted: g("--txt-2"), faint: g("--txt-3"),
      accent: g("--accent"), accent2: g("--accent-2"), cool: g("--cool"),
      violet: g("--violet"), warm: g("--warm"), bad: g("--bad")
    };
  }

  /* sidebar scroll-spy */
  function scrollSpy() {
    var links = Array.prototype.slice.call(document.querySelectorAll("nav a[href^='#']"));
    if (!links.length) return;
    var secs = links.map(function (l) { return document.querySelector(l.getAttribute("href")); });
    function upd() {
      var cur = 0;
      secs.forEach(function (s, i) { if (s && window.scrollY + 140 >= s.offsetTop) cur = i; });
      links.forEach(function (l, i) { l.classList.toggle("active", i === cur); });
    }
    addEventListener("scroll", upd, { passive: true });
    upd();
  }

  window.AI4DFT = { colors: colors, toggleTheme: toggle };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { mountToggle(); scrollSpy(); });
  } else { mountToggle(); scrollSpy(); }
})();
