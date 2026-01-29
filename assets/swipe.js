// assets/swipe.js  (pure client-side swipe)
(function () {
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function attach() {
    const container = document.getElementById("swipe-container");
    const handle = document.getElementById("swipe-handle");
    const afterClip = document.getElementById("after-clip");
    if (!container || !handle || !afterClip) return false;

    if (handle.__swipeBound) return true;
    handle.__swipeBound = true;

    console.log("[swipe] attached ✅");

    let dragging = false;

    function pctFromX(clientX) {
      const rect = container.getBoundingClientRect();
      const x = clamp(clientX - rect.left, 0, rect.width);
      return Math.round((x / rect.width) * 100);
    }

    function applyPct(pct) {
      handle.style.left = pct + "%";
      afterClip.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    }

    handle.addEventListener("mousedown", (e) => {
      dragging = true;
      applyPct(pctFromX(e.clientX));
      e.preventDefault();
      e.stopPropagation();
    }, true);

    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      applyPct(pctFromX(e.clientX));
      e.preventDefault();
    }, true);

    window.addEventListener("mouseup", () => { dragging = false; }, true);

    // click anywhere to jump
    container.addEventListener("mousedown", (e) => {
      if (e.target.closest("#swipe-handle")) return;
      applyPct(pctFromX(e.clientX));
    }, true);

    return true;
  }

  // Dash can render after load; keep trying until it exists
  const t = setInterval(() => {
    if (attach()) clearInterval(t);
  }, 100);
})();
