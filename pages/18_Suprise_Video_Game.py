import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Neon Market Arena", layout="wide")
st.title("Neon Market Arena")
st.caption("WASD move | Mouse aim | Click shoot | Space dash | P pause | R restart | Click canvas to focus")

html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body { margin:0; padding:0; background: transparent; }
    .wrap { width:100%; display:flex; justify-content:center; align-items:center; }
    .frame { position: relative; width: min(1320px, 98vw); }
    canvas {
      width: 100%;
      height: auto;
      display:block;
      border-radius: 22px;
      border: 1px solid rgba(255,255,255,0.10);
      background: radial-gradient(1200px 700px at 30% 20%, rgba(110,160,255,0.16), transparent),
                  radial-gradient(1000px 650px at 78% 22%, rgba(255,110,210,0.12), transparent),
                  linear-gradient(180deg, rgba(9,10,16,1), rgba(6,7,10,1));
      box-shadow: 0 18px 60px rgba(0,0,0,0.55);
      outline: none;
    }
    .hud {
      position:absolute;
      inset: 14px 14px auto 14px;
      display:flex;
      justify-content:space-between;
      gap: 10px;
      pointer-events:none;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      color: rgba(245,245,250,0.92);
      font-size: 13px;
      z-index: 5;
    }
    .pill {
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.06);
      backdrop-filter: blur(7px);
      box-shadow: 0 10px 30px rgba(0,0,0,0.30);
      white-space: nowrap;
    }
    .overlay {
      position:absolute;
      inset:0;
      display:flex;
      align-items:center;
      justify-content:center;
      pointer-events:none;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      color: rgba(245,245,250,0.92);
      z-index: 6;
    }
    .card {
      width: min(720px, 92%);
      border-radius: 20px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.07);
      padding: 18px 18px 16px 18px;
      box-shadow: 0 18px 70px rgba(0,0,0,0.60);
      text-align:left;
      pointer-events:none;
    }
    .title { font-size: 22px; font-weight: 820; margin: 0 0 8px 0; letter-spacing: 0.2px; }
    .sub { opacity: 0.90; margin: 0 0 10px 0; line-height: 1.42; }
    .kbd {
      display:inline-block;
      padding: 2px 8px;
      border-radius: 9px;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(0,0,0,0.22);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
      margin-right: 6px;
    }

    .shop {
      position:absolute;
      left: 50%;
      bottom: 18px;
      transform: translateX(-50%);
      width: min(980px, 94%);
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(8,10,14,0.70);
      backdrop-filter: blur(10px);
      box-shadow: 0 16px 50px rgba(0,0,0,0.55);
      padding: 14px 14px 12px 14px;
      display:none;
      z-index: 7;
      pointer-events:auto;
      color: rgba(245,245,250,0.92);
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }
    .shopTop { display:flex; justify-content:space-between; align-items:center; gap: 10px; }
    .shopTitle { font-size: 16px; font-weight: 760; }
    .shopHint { opacity: 0.78; font-size: 13px; }
    .opts { display:grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px; }
    .opt {
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06);
      padding: 12px 12px 10px 12px;
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease;
      user-select:none;
    }
    .opt:hover { transform: translateY(-2px); background: rgba(255,255,255,0.08); }
    .optName { font-weight: 750; margin-bottom: 4px; }
    .optDesc { opacity: 0.84; font-size: 12.5px; line-height: 1.35; }
    .optCost { margin-top: 8px; opacity: 0.90; font-size: 12.5px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="frame">
      <div class="hud">
        <div class="pill" id="hudL">Score: 0 | Alpha: 0 | Wave: 1</div>
        <div class="pill" id="hudM">HP: 100 | Heat: 0 | Regime: Risk-On</div>
        <div class="pill" id="hudR">High: 0</div>
      </div>

      <canvas id="game" width="1320" height="760" tabindex="0"></canvas>

      <div class="overlay" id="overlay" style="display:flex;">
        <div class="card">
          <div class="title">Neon Market Arena</div>
          <p class="sub">
            A fully client-side Streamlit game with bloom, trails, particles, screen shake, and procedural synthwave city.<br/>
            Survive waves. Farm alpha. Upgrade. Regimes shift and the arena tightens.
          </p>
          <p class="sub">
            <span class="kbd">W</span><span class="kbd">A</span><span class="kbd">S</span><span class="kbd">D</span> move |
            Aim with mouse |
            <span class="kbd">Click</span> shoot |
            <span class="kbd">Space</span> dash |
            <span class="kbd">P</span> pause |
            <span class="kbd">R</span> restart
          </p>
          <p class="sub" style="opacity:0.78; margin:0;">
            Click the game area, then click to begin.
          </p>
        </div>
      </div>

      <div class="shop" id="shop">
        <div class="shopTop">
          <div class="shopTitle">Upgrade Terminal</div>
          <div class="shopHint">Click an upgrade. Press P to resume. You can also keep playing.</div>
        </div>
        <div class="opts">
          <div class="opt" id="opt1">
            <div class="optName">Pulse Cannon</div>
            <div class="optDesc">Higher fire rate and tighter spread. Your baseline edge.</div>
            <div class="optCost" id="cost1">Cost: 40 alpha</div>
          </div>
          <div class="opt" id="opt2">
            <div class="optName">Phase Shield</div>
            <div class="optDesc">Adds a damage buffer. Recharges slowly over time.</div>
            <div class="optCost" id="cost2">Cost: 60 alpha</div>
          </div>
          <div class="opt" id="opt3">
            <div class="optName">Volatility Engine</div>
            <div class="optDesc">Dash cooldown down. Movement becomes a weapon.</div>
            <div class="optCost" id="cost3">Cost: 55 alpha</div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
(() => {
  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d");
  const hudL = document.getElementById("hudL");
  const hudM = document.getElementById("hudM");
  const hudR = document.getElementById("hudR");
  const overlay = document.getElementById("overlay");

  const shop = document.getElementById("shop");
  const opt1 = document.getElementById("opt1");
  const opt2 = document.getElementById("opt2");
  const opt3 = document.getElementById("opt3");

  const W = canvas.width;
  const H = canvas.height;

  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const lerp = (a, b, t) => a + (b - a) * t;

  function mulberry32(a) {
    return function() {
      let t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
  }

  function loadHigh() {
    const v = localStorage.getItem("neon_market_arena_high");
    const n = v ? parseInt(v, 10) : 0;
    return Number.isFinite(n) ? n : 0;
  }
  function saveHigh(v) {
    localStorage.setItem("neon_market_arena_high", String(v));
  }

  let highScore = loadHigh();

  // Offscreen buffers for bloom
  const scene = document.createElement("canvas");
  scene.width = W; scene.height = H;
  const sctx = scene.getContext("2d");

  const bloom = document.createElement("canvas");
  bloom.width = W; bloom.height = H;
  const bctx = bloom.getContext("2d");

  // Input
  const keys = new Set();
  let mouse = { x: W * 0.5, y: H * 0.5, down: false };

  function canvasToLocal(e) {
    const r = canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) * (W / r.width);
    const y = (e.clientY - r.top) * (H / r.height);
    return { x, y };
  }

  // Visual palette changes by regime
  const palettes = {
    riskOn: {
      bgA: "rgba(110,160,255,0.18)",
      bgB: "rgba(255,110,210,0.14)",
      neon1: "rgba(92,255,176,0.92)",
      neon2: "rgba(120,170,255,0.86)",
      neon3: "rgba(255,140,230,0.70)",
      enemy: "rgba(255,92,92,0.92)",
      bullet: "rgba(255,140,230,0.90)",
      alpha: "rgba(255,212,92,0.96)"
    },
    riskOff: {
      bgA: "rgba(60,220,255,0.10)",
      bgB: "rgba(255,120,80,0.10)",
      neon1: "rgba(120,210,255,0.88)",
      neon2: "rgba(255,170,110,0.80)",
      neon3: "rgba(255,110,110,0.62)",
      enemy: "rgba(255,140,90,0.92)",
      bullet: "rgba(120,210,255,0.88)",
      alpha: "rgba(255,212,92,0.96)"
    }
  };

  const state = {
    seed: 42069,
    rng: mulberry32(42069),
    running: false,
    paused: false,
    over: false,
    t: 0,
    score: 0,
    alpha: 0,
    wave: 1,

    // pacing
    heat: 0, // spawn pressure
    regime: "riskOn",
    regimeT: 0,
    regimeEvery: 1100, // frames

    // spawners
    enemyTimer: 0,
    pickupTimer: 0,

    // camera
    shake: 0,

    // shop
    shopOpen: false
  };

  const player = {
    x: W * 0.5,
    y: H * 0.62,
    vx: 0,
    vy: 0,
    r: 16,

    hp: 100,
    maxHp: 100,

    shield: 0,
    shieldMax: 0,
    shieldRegen: 0,

    dashCd: 0,
    dashCdMax: 140,
    dashTime: 0,

    fireCd: 0,
    fireCdMax: 10,
    spread: 0.06,
    bulletSpeed: 13.0,
    bulletDmg: 12,

    // visuals
    trail: []
  };

  let bullets = [];
  let enemies = [];
  let pickups = [];
  let particles = [];
  let city = [];
  let stars = [];

  function initBackground() {
    stars = [];
    for (let i = 0; i < 140; i++) {
      stars.push({
        x: state.rng() * W,
        y: state.rng() * H,
        r: 0.6 + state.rng() * 1.7,
        a: 0.12 + state.rng() * 0.35,
        s: 0.15 + state.rng() * 0.85
      });
    }

    city = [];
    // procedural skyline blocks
    for (let i = 0; i < 48; i++) {
      const baseX = (i / 48) * W;
      const w = 26 + state.rng() * 70;
      const h = 60 + state.rng() * 220;
      const layer = state.rng() < 0.55 ? 1 : 2;
      city.push({
        x: baseX + (state.rng() * 50 - 25),
        w, h,
        y: H * (layer === 1 ? 0.38 : 0.30),
        layer
      });
    }
  }

  function reset(seed=42069) {
    state.seed = seed;
    state.rng = mulberry32(seed);
    state.running = false;
    state.paused = false;
    state.over = false;
    state.t = 0;
    state.score = 0;
    state.alpha = 0;
    state.wave = 1;
    state.heat = 0;
    state.regime = "riskOn";
    state.regimeT = 0;
    state.enemyTimer = 30;
    state.pickupTimer = 140;
    state.shake = 0;
    state.shopOpen = false;

    player.x = W * 0.5;
    player.y = H * 0.62;
    player.vx = 0;
    player.vy = 0;
    player.hp = 100;
    player.maxHp = 100;

    player.shield = 0;
    player.shieldMax = 0;
    player.shieldRegen = 0;

    player.dashCd = 0;
    player.dashCdMax = 140;
    player.dashTime = 0;

    player.fireCd = 0;
    player.fireCdMax = 10;
    player.spread = 0.06;
    player.bulletSpeed = 13.0;
    player.bulletDmg = 12;

    player.trail = [];

    bullets = [];
    enemies = [];
    pickups = [];
    particles = [];

    initBackground();

    shop.style.display = "none";
    overlay.style.display = "flex";
    overlay.innerHTML = `
      <div class="card">
        <div class="title">Neon Market Arena</div>
        <p class="sub">
          Survive waves. Pick up alpha. Upgrade between waves.<br/>
          Regimes shift and spawn pressure changes.
        </p>
        <p class="sub">
          <span class="kbd">W</span><span class="kbd">A</span><span class="kbd">S</span><span class="kbd">D</span> move |
          Aim with mouse |
          <span class="kbd">Click</span> shoot |
          <span class="kbd">Space</span> dash |
          <span class="kbd">P</span> pause |
          <span class="kbd">R</span> restart
        </p>
        <p class="sub" style="opacity:0.78; margin:0;">
          Click the game area, then click to begin.
        </p>
      </div>
    `;
    updateHud();
  }

  function start() {
    if (state.over) return;
    state.running = true;
    state.paused = false;
    overlay.style.display = "none";
    canvas.focus();
  }

  function gameOver() {
    state.running = false;
    state.over = true;
    state.paused = false;
    shop.style.display = "none";

    if (state.score > highScore) {
      highScore = state.score;
      saveHigh(highScore);
    }
    updateHud();

    overlay.style.display = "flex";
    overlay.innerHTML = `
      <div class="card">
        <div class="title">Liquidated</div>
        <p class="sub">
          Score: <b>${state.score}</b> | Alpha: <b>${state.alpha}</b> | Wave: <b>${state.wave}</b> | High: <b>${highScore}</b><br/>
          Press <span class="kbd">R</span> to restart.
        </p>
        <p class="sub" style="opacity:0.78; margin:0;">
          Tip: dash through gaps, farm alpha early, and buy survivability before damage.
        </p>
      </div>
    `;
  }

  function updateHud() {
    const regimeName = (state.regime === "riskOn") ? "Risk-On" : "Risk-Off";
    hudL.textContent = `Score: ${state.score} | Alpha: ${state.alpha} | Wave: ${state.wave}`;
    hudM.textContent = `HP: ${Math.max(0, Math.floor(player.hp))} | Heat: ${Math.floor(state.heat)} | Regime: ${regimeName}`;
    hudR.textContent = `High: ${highScore}`;
  }

  function addParticles(x, y, color, n=14, spread=1.0) {
    for (let i = 0; i < n; i++) {
      const a = (state.rng() * Math.PI * 2);
      const sp = (0.8 + state.rng() * 5.0) * spread;
      particles.push({
        x, y,
        vx: Math.cos(a) * sp,
        vy: Math.sin(a) * sp,
        life: 18 + Math.floor(state.rng() * 22),
        r: 1.3 + state.rng() * 3.1,
        c: color,
      });
    }
  }

  function spawnEnemy() {
    // spawn from outside bounds
    const side = Math.floor(state.rng() * 4);
    let x, y;
    if (side === 0) { x = -40; y = state.rng() * H; }
    else if (side === 1) { x = W + 40; y = state.rng() * H; }
    else if (side === 2) { x = state.rng() * W; y = -40; }
    else { x = state.rng() * W; y = H + 40; }

    const tier = (state.rng() < 0.16 + state.wave * 0.01) ? 2 : 1;
    const hp = tier === 2 ? (40 + state.wave * 4) : (22 + state.wave * 3);
    const r = tier === 2 ? 20 : 16;

    enemies.push({
      x, y,
      vx: 0, vy: 0,
      r,
      hp,
      maxHp: hp,
      tier,
      hit: 0
    });
  }

  function spawnPickup() {
    const p = (state.rng() < 0.70) ? "alpha" : "med";
    pickups.push({
      x: 60 + state.rng() * (W - 120),
      y: 90 + state.rng() * (H - 160),
      r: 12,
      kind: p,
      t: state.rng() * Math.PI * 2
    });
  }

  function shoot() {
    if (!state.running || state.paused) return;
    if (player.fireCd > 0) return;

    const dx = mouse.x - player.x;
    const dy = mouse.y - player.y;
    const ang = Math.atan2(dy, dx);

    const spread = player.spread * (0.75 + state.heat * 0.004);
    const a = ang + (state.rng() * spread - spread * 0.5);
    const sp = player.bulletSpeed;

    bullets.push({
      x: player.x + Math.cos(a) * (player.r + 6),
      y: player.y + Math.sin(a) * (player.r + 6),
      vx: Math.cos(a) * sp,
      vy: Math.sin(a) * sp,
      r: 5.2,
      life: 90,
      dmg: player.bulletDmg
    });

    player.fireCd = player.fireCdMax;
    state.heat = Math.min(240, state.heat + 1.8);
  }

  function dash() {
    if (!state.running || state.paused) return;
    if (player.dashCd > 0) return;

    const dx = mouse.x - player.x;
    const dy = mouse.y - player.y;
    const ang = Math.atan2(dy, dx);

    player.vx += Math.cos(ang) * 18;
    player.vy += Math.sin(ang) * 18;
    player.dashCd = player.dashCdMax;
    player.dashTime = 12;
    state.shake = Math.min(18, state.shake + 8);

    const pal = palettes[state.regime];
    addParticles(player.x, player.y, pal.neon2, 26, 1.2);
  }

  function circleHit(ax, ay, ar, bx, by, br) {
    const dx = ax - bx, dy = ay - by;
    const rr = ar + br;
    return (dx*dx + dy*dy) <= rr*rr;
  }

  function screenShake() {
    if (state.shake <= 0) return {dx: 0, dy: 0};
    const mag = state.shake;
    return {
      dx: (state.rng() * 2 - 1) * mag,
      dy: (state.rng() * 2 - 1) * mag
    };
  }

  function drawBackground(pal) {
    // base
    sctx.clearRect(0,0,W,H);

    const g = sctx.createLinearGradient(0,0,0,H);
    g.addColorStop(0, "rgba(10,12,18,1)");
    g.addColorStop(1, "rgba(5,6,9,1)");
    sctx.fillStyle = g;
    sctx.fillRect(0,0,W,H);

    // nebula glows
    const g1 = sctx.createRadialGradient(W*0.24, H*0.20, 20, W*0.24, H*0.20, 920);
    g1.addColorStop(0, pal.bgA);
    g1.addColorStop(1, "rgba(0,0,0,0)");
    sctx.fillStyle = g1; sctx.fillRect(0,0,W,H);

    const g2 = sctx.createRadialGradient(W*0.80, H*0.22, 20, W*0.80, H*0.22, 860);
    g2.addColorStop(0, pal.bgB);
    g2.addColorStop(1, "rgba(0,0,0,0)");
    sctx.fillStyle = g2; sctx.fillRect(0,0,W,H);

    // stars parallax
    for (const st of stars) {
      st.y += st.s * (0.35 + (state.regime === "riskOff" ? 0.25 : 0.18));
      if (st.y > H + 10) { st.y = -10; st.x = state.rng() * W; }
      sctx.beginPath();
      sctx.arc(st.x, st.y, st.r, 0, Math.PI*2);
      sctx.fillStyle = `rgba(255,255,255,${st.a})`;
      sctx.fill();
    }

    // skyline silhouette
    sctx.save();
    const drift = (state.t * 0.25) % W;
    sctx.globalAlpha = 0.70;
    for (const b of city) {
      const layerMul = (b.layer === 1) ? 0.7 : 0.45;
      const x = (b.x - drift * layerMul + W) % (W + 100) - 50;
      const y = b.y;
      const h = b.h * (b.layer === 1 ? 1.0 : 1.15);

      sctx.fillStyle = "rgba(0,0,0,0.34)";
      sctx.fillRect(x, y, b.w, h);

      // windows glow
      sctx.globalAlpha = 0.10 + (b.layer === 1 ? 0.07 : 0.04);
      sctx.fillStyle = (state.regime === "riskOn") ? "rgba(120,170,255,0.22)" : "rgba(255,170,110,0.18)";
      for (let k = 0; k < 8; k++) {
        const wx = x + 6 + (k * 6);
        const wy = y + 10 + ((k * 17) % 70);
        sctx.fillRect(wx, wy, 2, 8);
      }
      sctx.globalAlpha = 0.70;
    }
    sctx.restore();

    // synth grid
    sctx.save();
    const horizonY = H * 0.40;
    const gridY0 = horizonY;
    const gridY1 = H;

    sctx.globalAlpha = 0.85;
    for (let i = 0; i < 34; i++) {
      const p = i / 34;
      const y = lerp(gridY0, gridY1, p);
      const w = lerp(W * 0.15, W * 1.30, p);
      const x0 = W*0.5 - w*0.5;
      const x1 = W*0.5 + w*0.5;

      sctx.strokeStyle = "rgba(255,255,255,0.04)";
      sctx.lineWidth = 1;
      sctx.beginPath();
      sctx.moveTo(x0, y);
      sctx.lineTo(x1, y);
      sctx.stroke();
    }

    const cols = 22;
    const scroll = (state.t * 0.010) % 1.0;
    for (let c = -cols; c <= cols; c++) {
      const p = (c / cols) * 0.5;
      const xTop = W*0.5 + p * (W*0.22);
      const xBot = W*0.5 + p * (W*1.1);

      sctx.strokeStyle = "rgba(255,255,255,0.05)";
      sctx.lineWidth = 1;
      sctx.beginPath();
      sctx.moveTo(xTop, horizonY);
      sctx.lineTo(xBot, H);
      sctx.stroke();
    }

    // moving neon band
    const bandY = lerp(horizonY + 20, H - 30, scroll);
    const bandG = sctx.createLinearGradient(0, bandY - 40, 0, bandY + 40);
    bandG.addColorStop(0, "rgba(0,0,0,0)");
    bandG.addColorStop(0.5, (state.regime === "riskOn") ? "rgba(255,110,210,0.06)" : "rgba(120,210,255,0.05)");
    bandG.addColorStop(1, "rgba(0,0,0,0)");
    sctx.fillStyle = bandG;
    sctx.fillRect(0, bandY - 60, W, 120);

    sctx.restore();
  }

  function drawEntities(pal) {
    // Player
    // trail
    player.trail.unshift({x: player.x, y: player.y});
    if (player.trail.length > 18) player.trail.pop();

    sctx.save();
    sctx.globalAlpha = 0.55;
    for (let i = 0; i < player.trail.length; i++) {
      const t = i / player.trail.length;
      const p = player.trail[i];
      sctx.beginPath();
      sctx.arc(p.x, p.y, player.r * (0.65 + t*0.35), 0, Math.PI*2);
      sctx.fillStyle = `rgba(92,255,176,${0.18 * (1 - t)})`;
      sctx.fill();
    }
    sctx.restore();

    // player glow
    sctx.save();
    sctx.shadowColor = pal.neon1;
    sctx.shadowBlur = 26;
    sctx.beginPath();
    sctx.arc(player.x, player.y, player.r + 3, 0, Math.PI*2);
    const pg = sctx.createRadialGradient(player.x - 8, player.y - 10, 2, player.x, player.y, player.r * 2.6);
    pg.addColorStop(0, "rgba(255,255,255,0.92)");
    pg.addColorStop(0.35, pal.neon1);
    pg.addColorStop(1, "rgba(0,0,0,0)");
    sctx.fillStyle = pg;
    sctx.fill();

    // shield
    if (player.shieldMax > 0) {
      const shA = clamp(player.shield / Math.max(1, player.shieldMax), 0, 1);
      if (shA > 0.02) {
        sctx.shadowColor = pal.neon2;
        sctx.shadowBlur = 30;
        sctx.beginPath();
        sctx.arc(player.x, player.y, player.r + 12, 0, Math.PI*2);
        sctx.strokeStyle = `rgba(120,170,255,${0.16 + shA*0.22})`;
        sctx.lineWidth = 2;
        sctx.stroke();
      }
    }
    sctx.restore();

    // Aim line
    sctx.save();
    sctx.globalAlpha = 0.35;
    sctx.strokeStyle = "rgba(255,255,255,0.16)";
    sctx.lineWidth = 1.5;
    sctx.beginPath();
    sctx.moveTo(player.x, player.y);
    sctx.lineTo(mouse.x, mouse.y);
    sctx.stroke();
    sctx.restore();

    // Bullets
    for (const b of bullets) {
      sctx.save();
      sctx.shadowColor = pal.bullet;
      sctx.shadowBlur = 18;
      sctx.beginPath();
      sctx.arc(b.x, b.y, b.r, 0, Math.PI*2);
      sctx.fillStyle = pal.bullet;
      sctx.fill();
      sctx.restore();
    }

    // Enemies
    for (const e of enemies) {
      const hpT = e.hp / Math.max(1, e.maxHp);
      sctx.save();
      sctx.shadowColor = pal.enemy;
      sctx.shadowBlur = 20 + e.hit * 10;
      sctx.beginPath();
      sctx.arc(e.x, e.y, e.r, 0, Math.PI*2);
      const eg = sctx.createRadialGradient(e.x - 6, e.y - 8, 2, e.x, e.y, e.r * 2.2);
      eg.addColorStop(0, "rgba(255,255,255,0.75)");
      eg.addColorStop(0.35, pal.enemy);
      eg.addColorStop(1, "rgba(0,0,0,0)");
      sctx.fillStyle = eg;
      sctx.fill();

      // hp ring
      sctx.shadowBlur = 0;
      sctx.beginPath();
      sctx.arc(e.x, e.y, e.r + 8, -Math.PI/2, -Math.PI/2 + Math.PI*2*hpT);
      sctx.strokeStyle = `rgba(255,255,255,${0.08 + 0.20*hpT})`;
      sctx.lineWidth = 2;
      sctx.stroke();
      sctx.restore();
    }

    // Pickups
    for (const p of pickups) {
      p.t += 0.06;
      const bob = Math.sin(p.t) * 4.5;
      sctx.save();
      sctx.shadowColor = pal.alpha;
      sctx.shadowBlur = 18;
      sctx.beginPath();
      sctx.arc(p.x, p.y + bob, p.r, 0, Math.PI*2);
      sctx.fillStyle = (p.kind === "alpha") ? pal.alpha : "rgba(92,255,176,0.90)";
      sctx.fill();
      sctx.strokeStyle = "rgba(255,255,255,0.16)";
      sctx.lineWidth = 2;
      sctx.stroke();
      sctx.restore();
    }

    // Particles
    for (const p of particles) {
      sctx.save();
      sctx.globalAlpha = clamp(p.life / 28, 0, 1);
      sctx.shadowColor = p.c;
      sctx.shadowBlur = 14;
      sctx.beginPath();
      sctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
      sctx.fillStyle = p.c;
      sctx.fill();
      sctx.restore();
    }

    // Small vignette
    sctx.save();
    const vg = sctx.createRadialGradient(W*0.5, H*0.5, H*0.18, W*0.5, H*0.5, H*0.62);
    vg.addColorStop(0, "rgba(0,0,0,0)");
    vg.addColorStop(1, "rgba(0,0,0,0.40)");
    sctx.fillStyle = vg;
    sctx.fillRect(0,0,W,H);
    sctx.restore();

    // Scanlines and noise
    sctx.save();
    sctx.globalAlpha = 0.10;
    sctx.fillStyle = "rgba(0,0,0,0.18)";
    for (let y = 0; y < H; y += 3) sctx.fillRect(0, y, W, 1);

    sctx.globalAlpha = 0.055;
    for (let i = 0; i < 160; i++) {
      const x = state.rng() * W;
      const y = state.rng() * H;
      sctx.fillStyle = "rgba(255,255,255,0.07)";
      sctx.fillRect(x, y, 1.2, 1.2);
    }
    sctx.restore();
  }

  function compositeBloom() {
    // Bloom extraction
    bctx.clearRect(0,0,W,H);
    bctx.drawImage(scene, 0, 0);

    // Blur pass
    bctx.save();
    bctx.globalCompositeOperation = "source-in";
    bctx.filter = "blur(10px)";
    bctx.globalAlpha = 0.85;
    bctx.drawImage(scene, 0, 0);
    bctx.restore();

    // Second blur pass for softer glow
    bctx.save();
    bctx.filter = "blur(22px)";
    bctx.globalAlpha = 0.40;
    bctx.drawImage(scene, 0, 0);
    bctx.restore();

    // Final composite to main
    ctx.clearRect(0,0,W,H);
    ctx.drawImage(scene, 0, 0);

    ctx.save();
    ctx.globalCompositeOperation = "screen";
    ctx.globalAlpha = 0.75;
    ctx.drawImage(bloom, 0, 0);
    ctx.restore();

    // Subtle chroma shift feel (fake): offset additive layer
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = 0.12;
    ctx.drawImage(scene, 1.5, 0.0);
    ctx.drawImage(scene, -1.0, 0.8);
    ctx.restore();
  }

  function openShopIfReady() {
    // shop opens briefly at wave boundaries
    if (state.wave >= 2 && enemies.length === 0 && state.running && !state.paused && !state.shopOpen) {
      state.shopOpen = true;
      shop.style.display = "block";
    }
    if (state.shopOpen && enemies.length > 0) {
      state.shopOpen = false;
      shop.style.display = "none";
    }
  }

  function buy(kind) {
    if (!state.running) return;
    if (!state.shopOpen) return;

    if (kind === 1) {
      const cost = 40;
      if (state.alpha < cost) return;
      state.alpha -= cost;
      player.fireCdMax = Math.max(4, player.fireCdMax - 1);
      player.spread = Math.max(0.018, player.spread * 0.88);
      player.bulletDmg += 2;
      state.shake = Math.min(14, state.shake + 6);
      addParticles(player.x, player.y, "rgba(255,140,230,0.45)", 30, 1.2);
    } else if (kind === 2) {
      const cost = 60;
      if (state.alpha < cost) return;
      state.alpha -= cost;
      player.shieldMax = Math.min(120, player.shieldMax + 40);
      player.shield = Math.min(player.shieldMax, player.shield + 40);
      player.shieldRegen = Math.min(0.30, player.shieldRegen + 0.08);
      state.shake = Math.min(14, state.shake + 6);
      addParticles(player.x, player.y, "rgba(120,170,255,0.45)", 30, 1.2);
    } else if (kind === 3) {
      const cost = 55;
      if (state.alpha < cost) return;
      state.alpha -= cost;
      player.dashCdMax = Math.max(70, Math.floor(player.dashCdMax * 0.86));
      state.shake = Math.min(14, state.shake + 6);
      addParticles(player.x, player.y, "rgba(92,255,176,0.40)", 30, 1.2);
    }

    updateHud();
  }

  opt1.addEventListener("click", () => buy(1));
  opt2.addEventListener("click", () => buy(2));
  opt3.addEventListener("click", () => buy(3));

  function step() {
    requestAnimationFrame(step);

    // idle draw for menu
    const pal = palettes[state.regime];

    // handle pause
    if (!state.running || state.paused || state.over) {
      drawBackground(pal);
      drawEntities(pal);
      compositeBloom();
      updateHud();
      return;
    }

    const dt = 1.0;
    state.t += dt;

    // regime shift
    state.regimeT += dt;
    if (state.regimeT >= state.regimeEvery) {
      state.regimeT = 0;
      state.regime = (state.regime === "riskOn") ? "riskOff" : "riskOn";
      state.shake = Math.min(18, state.shake + 10);
      const pcol = (state.regime === "riskOn") ? "rgba(255,140,230,0.40)" : "rgba(120,210,255,0.35)";
      addParticles(W*0.5, H*0.40, pcol, 70, 1.6);
    }

    // heat decay
    state.heat = Math.max(0, state.heat - 0.35);

    // spawners and wave logic
    const basePressure = 0.55 + state.wave * 0.08;
    const regimeMul = (state.regime === "riskOff") ? 1.12 : 0.98;
    const pressure = basePressure * regimeMul;

    state.enemyTimer -= dt;
    state.pickupTimer -= dt;

    const enemyEvery = clamp(38 - pressure * 7.5, 12, 34);
    if (state.enemyTimer <= 0) {
      spawnEnemy();
      if (state.rng() < 0.12 + state.wave * 0.01) spawnEnemy();
      state.enemyTimer = enemyEvery;
      state.heat = Math.min(240, state.heat + 2.2);
    }

    if (state.pickupTimer <= 0) {
      spawnPickup();
      state.pickupTimer = clamp(240 - state.wave * 12, 110, 220);
    }

    // wave increment conditions
    if (Math.floor(state.t) % 560 === 0) state.wave += 1;

    openShopIfReady();

    // player movement
    const accel = 0.52;
    const maxV = 7.2 + state.wave * 0.05;
    const friction = 0.88;

    let ax = 0, ay = 0;
    if (keys.has("w") || keys.has("arrowup")) ay -= accel;
    if (keys.has("s") || keys.has("arrowdown")) ay += accel;
    if (keys.has("a") || keys.has("arrowleft")) ax -= accel;
    if (keys.has("d") || keys.has("arrowright")) ax += accel;

    player.vx = clamp(player.vx + ax, -maxV, maxV);
    player.vy = clamp(player.vy + ay, -maxV, maxV);

    // dash state
    if (player.dashTime > 0) {
      player.dashTime -= dt;
      player.vx *= 0.985;
      player.vy *= 0.985;
    }

    player.x += player.vx;
    player.y += player.vy;

    player.vx *= friction;
    player.vy *= friction;

    // bounds
    player.x = clamp(player.x, 40, W - 40);
    player.y = clamp(player.y, 70, H - 60);

    // cooldowns
    player.fireCd = Math.max(0, player.fireCd - dt);
    player.dashCd = Math.max(0, player.dashCd - dt);

    // shield regen
    if (player.shieldMax > 0 && player.shieldRegen > 0) {
      player.shield = Math.min(player.shieldMax, player.shield + player.shieldRegen);
    }

    // shooting
    if (mouse.down) shoot();

    // bullets
    for (let i = bullets.length - 1; i >= 0; i--) {
      const b = bullets[i];
      b.x += b.vx;
      b.y += b.vy;
      b.life -= dt;
      if (b.life <= 0 || b.x < -40 || b.x > W+40 || b.y < -40 || b.y > H+40) bullets.splice(i, 1);
    }

    // enemies seek player
    for (const e of enemies) {
      const dx = player.x - e.x;
      const dy = player.y - e.y;
      const d = Math.hypot(dx, dy) + 1e-6;
      const sp = (e.tier === 2 ? 2.2 : 2.6) + state.wave * 0.02 + (state.regime === "riskOff" ? 0.22 : 0.0);
      e.vx = lerp(e.vx, (dx / d) * sp, 0.08);
      e.vy = lerp(e.vy, (dy / d) * sp, 0.08);
      e.x += e.vx;
      e.y += e.vy;
      e.hit = Math.max(0, e.hit - 0.08);
    }

    // collisions: bullets vs enemies
    for (let i = enemies.length - 1; i >= 0; i--) {
      const e = enemies[i];
      for (let j = bullets.length - 1; j >= 0; j--) {
        const b = bullets[j];
        if (circleHit(e.x, e.y, e.r, b.x, b.y, b.r)) {
          e.hp -= b.dmg;
          e.hit = 1.0;
          bullets.splice(j, 1);

          const palNow = palettes[state.regime];
          addParticles(b.x, b.y, palNow.bullet, 10, 0.9);

          if (e.hp <= 0) {
            enemies.splice(i, 1);
            const palNow2 = palettes[state.regime];
            addParticles(e.x, e.y, palNow2.enemy, 40, 1.35);
            state.shake = Math.min(18, state.shake + (e.tier === 2 ? 10 : 6));
            state.score += (e.tier === 2 ? 420 : 220) + state.wave * 10;
            state.alpha += (e.tier === 2 ? 10 : 6);

            // occasional extra alpha pickup drop
            if (state.rng() < 0.18) {
              pickups.push({ x: e.x, y: e.y, r: 12, kind: "alpha", t: 0 });
            }
          }
          break;
        }
      }
    }

    // collisions: enemies vs player
    for (let i = enemies.length - 1; i >= 0; i--) {
      const e = enemies[i];
      if (circleHit(e.x, e.y, e.r, player.x, player.y, player.r + 4)) {
        enemies.splice(i, 1);

        const dmg = (e.tier === 2 ? 22 : 14) + Math.floor(state.wave * 0.7);
        let remaining = dmg;

        if (player.shield > 0) {
          const sTake = Math.min(player.shield, remaining);
          player.shield -= sTake;
          remaining -= sTake;
        }
        player.hp -= remaining;

        const palNow = palettes[state.regime];
        addParticles(player.x, player.y, palNow.enemy, 56, 1.5);
        state.shake = Math.min(22, state.shake + 14);
        state.heat = Math.min(240, state.heat + 10);

        if (player.hp <= 0) {
          gameOver();
          break;
        }
      }
    }

    // pickups
    for (let i = pickups.length - 1; i >= 0; i--) {
      const p = pickups[i];
      if (circleHit(p.x, p.y, p.r, player.x, player.y, player.r + 8)) {
        pickups.splice(i, 1);
        const palNow = palettes[state.regime];

        if (p.kind === "alpha") {
          state.alpha += 12;
          state.score += 180;
          addParticles(p.x, p.y, palNow.alpha, 26, 1.2);
        } else {
          player.hp = Math.min(player.maxHp, player.hp + 26);
          state.score += 120;
          addParticles(p.x, p.y, palNow.neon1, 26, 1.2);
        }
      }
    }

    // particles
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.97;
      p.vy *= 0.97;
      p.life -= 1;
      if (p.life <= 0) particles.splice(i, 1);
    }

    // camera shake decay
    state.shake = Math.max(0, state.shake * 0.88);

    // draw
    const pal2 = palettes[state.regime];
    drawBackground(pal2);

    // apply shake as transform on scene
    const sh = screenShake();
    sctx.save();
    sctx.translate(sh.dx, sh.dy);
    drawEntities(pal2);
    sctx.restore();

    compositeBloom();

    // score drift
    state.score += 1 + Math.floor(state.wave * 0.15);

    // high score live
    if (state.score > highScore) { highScore = state.score; saveHigh(highScore); }

    updateHud();
  }

  // Input handlers
  window.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    keys.add(k);

    if (k === "p") {
      e.preventDefault();
      if (!state.running || state.over) return;
      state.paused = !state.paused;
      if (state.paused) {
        shop.style.display = "none";
        overlay.style.display = "flex";
        overlay.innerHTML = `
          <div class="card">
            <div class="title">Paused</div>
            <p class="sub">
              Press <span class="kbd">P</span> to resume. Press <span class="kbd">R</span> to restart.
            </p>
            <p class="sub" style="opacity:0.78; margin:0;">
              Tip: upgrades are strongest when you buy survivability early, then convert it into fire rate.
            </p>
          </div>
        `;
      } else {
        overlay.style.display = "none";
      }
      return;
    }

    if (k === "r") {
      e.preventDefault();
      reset(state.seed);
      return;
    }

    if (k === " " ) {
      e.preventDefault();
      if (!state.running) start();
      if (state.paused || state.over) return;
      dash();
      return;
    }
  });

  window.addEventListener("keyup", (e) => {
    keys.delete(e.key.toLowerCase());
  });

  canvas.addEventListener("mousemove", (e) => {
    const p = canvasToLocal(e);
    mouse.x = p.x; mouse.y = p.y;
  });

  canvas.addEventListener("mousedown", (e) => {
    canvas.focus();
    const p = canvasToLocal(e);
    mouse.x = p.x; mouse.y = p.y;
    mouse.down = true;
    if (!state.running && !state.over) start();
  });

  window.addEventListener("mouseup", () => {
    mouse.down = false;
  });

  canvas.addEventListener("click", () => {
    canvas.focus();
    if (!state.running && !state.over) start();
  });

  // Touch support: drag aim, press to shoot, double tap dash
  let lastTap = 0;
  canvas.addEventListener("touchstart", (e) => {
    canvas.focus();
    if (!state.running && !state.over) start();
    const t0 = e.touches[0];
    const r = canvas.getBoundingClientRect();
    mouse.x = (t0.clientX - r.left) * (W / r.width);
    mouse.y = (t0.clientY - r.top) * (H / r.height);
    mouse.down = true;

    const now = performance.now();
    if (now - lastTap < 260) dash();
    lastTap = now;
  }, {passive:true});

  canvas.addEventListener("touchmove", (e) => {
    const t0 = e.touches[0];
    const r = canvas.getBoundingClientRect();
    mouse.x = (t0.clientX - r.left) * (W / r.width);
    mouse.y = (t0.clientY - r.top) * (H / r.height);
  }, {passive:true});

  canvas.addEventListener("touchend", () => {
    mouse.down = false;
  }, {passive:true});

  // Boot
  hudR.textContent = `High: ${highScore}`;
  reset(42069);
  step();
})();
</script>
</body>
</html>
"""

components.html(html, height=820, scrolling=False)
