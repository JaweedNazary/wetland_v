window.dash_clientside = Object.assign({}, window.dash_clientside, {
  swipe: {
    to_store: function (val, currentData) {
      const pct = parseInt(val || "50", 10);
      const safe = isNaN(pct) ? 50 : Math.max(0, Math.min(100, pct));
      return { pct: safe };
    },
  },
});
