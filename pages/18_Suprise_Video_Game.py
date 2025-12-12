import json
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Neon Market Arena", layout="wide")
st.title("Neon Market Arena")

with st.sidebar:
    st.subheader("Game Settings")
    quality = st.selectbox("Graphics quality", ["Ultra", "High", "Medium", "Low"], index=1)
    difficulty = st.selectbox("Difficulty", ["Chill", "Normal", "Hard"], index=1)
    reduce_motion = st.toggle("Reduce motion (less shake, less blur)", value=False)
    show_fps = st.toggle("Show FPS", value=False)
    seed = st.number_input("Seed", min_value=1, max_value=999999999, value=42069, step=1)

cfg = {
    "quality": quality,
    "difficulty": difficulty,
    "reduce_motion": reduce_motion,
    "show_fps": show_fps,
    "seed": int(seed),
}

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
      width: min(760px, 92%);
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
            Delta-time physics, bloom downsampling, and proper waves and intermissions.<br/>
            Survive. Farm alpha. Upgrade. Regimes shift.
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
          <div class="shopHint">Click an upgrade. Intermission ends automatically.</div>
        </div>
        <div class="opts">
          <div class="opt" id="opt1">
            <div class="optName">Pulse Cannon</div>
            <div class="optDesc">Higher fire rate and tighter spread, plus a small damage bump.</div>
            <div class="optCost">Cost: 40 alpha</div>
          </div>
          <div class="opt" id="opt2">
            <div class="optName">Phase Shield</div>
            <div class="optDesc">Adds a damage buffer. Recharges slowly over time.</div>
            <div class="optCost">Cost: 60 alpha</div>
          </div>
          <div class="opt" id="opt3">
            <div class="optName">Volatility Engine</div>
            <div class="optDesc">Dash cooldown down and stronger dash impulse.</div>
            <div class="optCost">Cost: 55 alpha</div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
(() => {
  const CONFIG = __CONFIG_JSON__;

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
    const v = localStorage.getItem("neon_market_arena_high_v2");
    const n = v ? parseInt(v, 10) : 0;
    return Number.isFinite(n) ? n : 0;
  }
  function saveHigh(v) {
    localStorage.setItem("neon_market_arena_high_v2", String(v));
  }

  let highScore = loadHigh();

  const Q = (() => {
    const rm = !!CONFIG.reduce_motion;
    const q = (CONFIG.quality || "High").toLowerCase();
    if (q === "ultra") return { bloomScale: 0.5, blur1: rm ? 8 : 12, blur2: rm ? 16 : 26, stars: 160, noise: 160, scanlines: true, shakeMul: rm ? 0.55 : 1.0 };
    if (q === "high")  return { bloomScale: 0.5, blur1: rm ? 7 : 10, blur2: rm ? 14 : 22, stars: 140, noise: 140, scanlines: true, shakeMul: rm ? 0.55 : 1.0 };
    if (q === "medium")return { bloomScale: 0.66, blur1: rm ? 6 : 8,  blur2: rm ? 12 : 18, stars: 110, noise: 90,  scanlines: true, shakeMul: rm ? 0.50 : 0.90 };
    return               { bloomScale: 0.75, blur1: rm ? 5 : 6,  blur2: rm ? 10 : 12, stars: 80,  noise: 60,  scanlines: false, shakeMul: rm ? 0.40 : 0.75 };
  })();

  const D = (() => {
    const d = (CONFIG.difficulty || "Normal").toLowerCase();
    if (d === "chill")  return { enemyHpMul: 0.85, enemySpMul: 0.90, spawnMul: 0.82, dmgMul: 0.85, alphaMul: 1.10 };
    if (d === "hard")   return { enemyHpMul: 1.18, enemySpMul: 1.12, spawnMul: 1.18, dmgMul: 1.15, alphaMul: 0.95 };
    return               { enemyHpMul: 1.00, enemySpMul: 1.00, spawnMul: 1.00, dmgMul: 1.00, alphaMul: 1.00 };
  })();

  const scene = document.createElement("canvas");
  scene.width = W; scene.height = H;
  const sctx = scene.getContext("2d");

  const bloom = document.createElement("canvas");
  bloom.width = Math.floor(W * Q.bloomScale);
  bloom.height = Math.floor(H * Q.bloomScale);
  const bctx = bloom.getContext("2d");

  const keys = new Set();
  let mouse = { x: W * 0.5, y: H * 0.5, down: false };

  function canvasToLocal(e) {
    const r = canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) * (W / r.width);
    const y = (e.clientY - r.top) * (H / r.height);
    return { x, y };
  }

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
    seed: CONFIG.seed || 42069,
    rng: mulberry32(CONFIG.seed || 42069),

    running: false,
    paused: false,
    over: false,

    t: 0,

    score: 0,
    alpha: 0,

    wave: 1,
    waveTime: 0,
    waveDuration: 35.0,
    intermission: false,
    intermissionTime: 0,
    intermissionDuration: 8.0,

    heat: 0,

    regime: "riskOn",
    regimeTime: 0,
    regimeEvery: 55.0,

    enemyTimer: 0,
    pickupTimer: 0,

    shake: 0,
    hitStop: 0
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
    dashCdMax: 2.3,
    dashTime: 0,

    fireCd: 0,
    fireCdMax: 0.11,
    spread: 0.06,
    bulletSpeed: 780,
    bulletDmg: 12,

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
    for (let i = 0; i < Q.stars; i++) {
      stars.push({
        x: state.rng() * W,
        y: state.rng() * H,
        r: 0.6 + state.rng() * 1.7,
        a: 0.12 + state.rng() * 0.35,
        s: 0.15 + state.rng() * 0.85
      });
    }

    city = [];
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

  function updateHud(fps=0) {
    const regimeName = (state.regime === "riskOn") ? "Risk-On" : "Risk-Off";
    const hp = Math.max(0, Math.floor(player.hp));
    const heat = Math.floor(state.heat);
    const extra = CONFIG.show_fps ? ` | FPS: ${fps.toFixed(0)}` : "";
    hudL.textContent = `Score: ${state.score} | Alpha: ${state.alpha} | Wave: ${state.wave}`;
    hudM.textContent = `HP: ${hp} | Heat: ${heat} | Regime: ${regimeName}${extra}`;
    hudR.textContent = `High: ${highScore}`;
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
    state.waveTime = 0;
    state.intermission = false;
    state.intermissionTime = 0;

    state.heat = 0;
    state.regime = "riskOn";
    state.regimeTime = 0;

    state.enemyTimer = 0.6;
    state.pickupTimer = 2.8;

    state.shake = 0;
    state.hitStop = 0;

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
    player.dashCdMax = 2.3;
    player.dashTime = 0;

    player.fireCd = 0;
    player.fireCdMax = 0.11;
    player.spread = 0.06;
    player.bulletSpeed = 780;
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
          Quality: <b>${CONFIG.quality}</b> | Difficulty: <b>${CONFIG.difficulty}</b><br/>
          Survive waves. Intermission each wave. Regimes shift.
        </p>
        <p class="sub">
          <span class="kbd">W</span><span class="kbd">A</span><span class="kbd">S</span><span class="kbd">D</span> move |
          Aim with mouse |
          <span class="kbd">Click</span> shoot |
          <span class="kbd">Space</span> dash |
          <span class="kbd">P</span> pause |
          <span class="kbd">R</span> restart
        </p>
        <p class="sub" style="opacity:0.78; margin:0;">Click the game area, then click to begin.</p>
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

    overlay.style.display = "flex";
    overlay.innerHTML = `
      <div class="card">
        <div class="title">Liquidated</div>
        <p class="sub">
          Score: <b>${state.score}</b> | Alpha: <b>${state.alpha}</b> | Wave: <b>${state.wave}</b> | High: <b>${highScore}</b><br/>
          Press <span class="kbd">R</span> to restart.
        </p>
      </div>
    `;
  }

  function addParticles(x, y, color, n=14, spread=1.0) {
    for (let i = 0; i < n; i++) {
      const a = (state.rng() * Math.PI * 2);
      const sp = (40 + state.rng() * 260) * spread;
      particles.push({
        x, y,
        vx: Math.cos(a) * sp,
        vy: Math.sin(a) * sp,
        life: 0.35 + state.rng() * 0.35,
        r: 1.3 + state.rng() * 3.1,
        c: color,
      });
    }
  }

  function circleHit(ax, ay, ar, bx, by, br) {
    const dx = ax - bx, dy = ay - by;
    const rr = ar + br;
    return (dx*dx + dy*dy) <= rr*rr;
  }

  function spawnEnemy() {
    const side = Math.floor(state.rng() * 4);
    let x, y;
    if (side === 0) { x = -40; y = state.rng() * H; }
    else if (side === 1) { x = W + 40; y = state.rng() * H; }
    else if (side === 2) { x = state.rng() * W; y = -40; }
    else { x = state.rng() * W; y = H + 40; }

    const tierRoll = 0.14 + state.wave * 0.01 + (state.regime === "riskOff" ? 0.05 : 0.0);
    const tier = (state.rng() < tierRoll) ? 2 : 1;

    const hpBase = tier === 2 ? (44 + state.wave * 5) : (24 + state.wave * 3.2);
    const hp = Math.floor(hpBase * D.enemyHpMul);
    const r = tier === 2 ? 20 : 16;

    enemies.push({ x, y, vx: 0, vy: 0, r, hp, maxHp: hp, tier, hit: 0 });
  }

  function spawnPickup() {
    const p = (state.rng() < 0.72) ? "alpha" : "med";
    pickups.push({
      x: 60 + state.rng() * (W - 120),
      y: 90 + state.rng() * (H - 160),
      r: 12,
      kind: p,
      t: state.rng() * Math.PI * 2
    });
  }

  function shoot() {
    if (!state.running || state.paused || state.intermission) return;
    if (player.fireCd > 0) return;

    const dx = mouse.x - player.x;
    const dy = mouse.y - player.y;
    const ang = Math.atan2(dy, dx);

    const spread = player.spread * (0.80 + state.heat * 0.0035);
    const a = ang + (state.rng() * spread - spread * 0.5);

    bullets.push({
      x: player.x + Math.cos(a) * (player.r + 7),
      y: player.y + Math.sin(a) * (player.r + 7),
      vx: Math.cos(a) * player.bulletSpeed,
      vy: Math.sin(a) * player.bulletSpeed,
      r: 5.2,
      life: 0.95,
      dmg: player.bulletDmg
    });

    player.fireCd = player.fireCdMax;
    state.heat = Math.min(240, state.heat + 2.0);
  }

  function dash() {
    if (!state.running || state.paused) return;
    if (player.dashCd > 0) return;

    const dx = mouse.x - player.x;
    const dy = mouse.y - player.y;
    const ang = Math.atan2(dy, dx);

    const impulse = 980;
    player.vx += Math.cos(ang) * impulse;
    player.vy += Math.sin(ang) * impulse;

    player.dashCd = player.dashCdMax;
    player.dashTime = 0.10;

    state.shake = Math.min(18, state.shake + 10);

    const pal = palettes[state.regime];
    addParticles(player.x, player.y, pal.neon2, 28, 1.1);
  }

  function buy(kind) {
    if (!state.running || !state.intermission) return;

    if (kind === 1) {
      const cost = 40;
      if (state.alpha < cost) return;
      state.alpha -= cost;
      player.fireCdMax = Math.max(0.05, player.fireCdMax - 0.01);
      player.spread = Math.max(0.018, player.spread * 0.88);
      player.bulletDmg += 2;
      addParticles(player.x, player.y, palettes[state.regime].neon3, 32, 1.2);
    } else if (kind === 2) {
      const cost = 60;
      if (state.alpha < cost) return;
      state.alpha -= cost;
      player.shieldMax = Math.min(140, player.shieldMax + 40);
      player.shield = Math.min(player.shieldMax, player.shield + 50);
      player.shieldRegen = Math.min(32, player.shieldRegen + 10);
      addParticles(player.x, player.y, palettes[state.regime].neon2, 32, 1.2);
    } else if (kind === 3) {
      const cost = 55;
      if (state.alpha < cost) return;
      state.alpha -= cost;
      player.dashCdMax = Math.max(1.1, player.dashCdMax * 0.86);
      addParticles(player.x, player.y, palettes[state.regime].neon1, 32, 1.2);
    }
    updateHud();
  }

  opt1.addEventListener("click", () => buy(1));
  opt2.addEventListener("click", () => buy(2));
  opt3.addEventListener("click", () => buy(3));

  function drawBackground(pal, dt) {
    sctx.clearRect(0,0,W,H);

    const g = sctx.createLinearGradient(0,0,0,H);
    g.addColorStop(0, "rgba(10,12,18,1)");
    g.addColorStop(1, "rgba(5,6,9,1)");
    sctx.fillStyle = g;
    sctx.fillRect(0,0,W,H);

    const g1 = sctx.createRadialGradient(W*0.24, H*0.20, 20, W*0.24, H*0.20, 920);
    g1.addColorStop(0, pal.bgA);
    g1.addColorStop(1, "rgba(0,0,0,0)");
    sctx.fillStyle = g1; sctx.fillRect(0,0,W,H);

    const g2 = sctx.createRadialGradient(W*0.80, H*0.22, 20, W*0.80, H*0.22, 860);
    g2.addColorStop(0, pal.bgB);
    g2.addColorStop(1, "rgba(0,0,0,0)");
    sctx.fillStyle = g2; sctx.fillRect(0,0,W,H);

    for (const st of stars) {
      st.y += st.s * (50 * dt);
      if (st.y > H + 10) { st.y = -10; st.x = state.rng() * W; }
      sctx.beginPath();
      sctx.arc(st.x, st.y, st.r, 0, Math.PI*2);
      sctx.fillStyle = `rgba(255,255,255,${st.a})`;
      sctx.fill();
    }

    sctx.save();
    const drift = (state.t * 40) % W;
    sctx.globalAlpha = 0.70;
    for (const b of city) {
      const layerMul = (b.layer === 1) ? 0.7 : 0.45;
      const x = (b.x - drift * layerMul + W) % (W + 100) - 50;
      const y = b.y;
      const h = b.h * (b.layer === 1 ? 1.0 : 1.15);

      sctx.fillStyle = "rgba(0,0,0,0.34)";
      sctx.fillRect(x, y, b.w, h);

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

    sctx.save();
    const horizonY = H * 0.40;

    for (let i = 0; i < 34; i++) {
      const p = i / 34;
      const y = lerp(horizonY, H, p);
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
    sctx.restore();
  }

  function drawEntities(pal) {
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
    sctx.restore();

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

    for (const e of enemies) {
      const hpT = e.hp / Math.max(1, e.maxHp);
      sctx.save();
      sctx.shadowColor = pal.enemy;
      sctx.shadowBlur = 20 + e.hit * 12;
      sctx.beginPath();
      sctx.arc(e.x, e.y, e.r, 0, Math.PI*2);
      const eg = sctx.createRadialGradient(e.x - 6, e.y - 8, 2, e.x, e.y, e.r * 2.2);
      eg.addColorStop(0, "rgba(255,255,255,0.75)");
      eg.addColorStop(0.35, pal.enemy);
      eg.addColorStop(1, "rgba(0,0,0,0)");
      sctx.fillStyle = eg;
      sctx.fill();

      sctx.shadowBlur = 0;
      sctx.beginPath();
      sctx.arc(e.x, e.y, e.r + 8, -Math.PI/2, -Math.PI/2 + Math.PI*2*hpT);
      sctx.strokeStyle = `rgba(255,255,255,${0.08 + 0.20*hpT})`;
      sctx.lineWidth = 2;
      sctx.stroke();
      sctx.restore();
    }

    for (const p of pickups) {
      p.t += 2.6 / 60.0;
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

    for (const p of particles) {
      sctx.save();
      sctx.globalAlpha = clamp(p.life / 0.6, 0, 1);
      sctx.shadowColor = p.c;
      sctx.shadowBlur = 14;
      sctx.beginPath();
      sctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
      sctx.fillStyle = p.c;
      sctx.fill();
      sctx.restore();
    }

    sctx.save();
    if (Q.scanlines) {
      sctx.globalAlpha = 0.10;
      sctx.fillStyle = "rgba(0,0,0,0.18)";
      for (let y = 0; y < H; y += 3) sctx.fillRect(0, y, W, 1);
    }
    sctx.globalAlpha = 0.055;
    for (let i = 0; i < Q.noise; i++) {
      const x = state.rng() * W;
      const y = state.rng() * H;
      sctx.fillStyle = "rgba(255,255,255,0.07)";
      sctx.fillRect(x, y, 1.2, 1.2);
    }
    sctx.restore();
  }

  function compositeBloom() {
    bctx.clearRect(0,0,bloom.width,bloom.height);
    bctx.save();
    bctx.scale(bloom.width / W, bloom.height / H);
    bctx.drawImage(scene, 0, 0);
    bctx.restore();

    bctx.save();
    bctx.globalCompositeOperation = "source-in";
    bctx.filter = `blur(${Q.blur1}px)`;
    bctx.globalAlpha = 0.85;
    bctx.drawImage(bloom, 0, 0);
    bctx.restore();

    bctx.save();
    bctx.filter = `blur(${Q.blur2}px)`;
    bctx.globalAlpha = 0.40;
    bctx.drawImage(bloom, 0, 0);
    bctx.restore();

    ctx.clearRect(0,0,W,H);
    ctx.drawImage(scene, 0, 0);

    ctx.save();
    ctx.globalCompositeOperation = "screen";
    ctx.globalAlpha = 0.75;
    ctx.drawImage(bloom, 0, 0, W, H);
    ctx.restore();

    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = 0.10;
    ctx.drawImage(scene, 1.5, 0.0);
    ctx.drawImage(scene, -1.0, 0.8);
    ctx.restore();
  }

  function setIntermission(on) {
    state.intermission = on;
    state.intermissionTime = 0;
    shop.style.display = on ? "block" : "none";
  }

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

    if (k === " ") {
      e.preventDefault();
      if (!state.running) start();
      if (state.paused || state.over) return;
      dash();
      return;
    }
  });

  window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));

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

  window.addEventListener("mouseup", () => mouse.down = false);

  canvas.addEventListener("click", () => {
    canvas.focus();
    if (!state.running && !state.over) start();
  });

  let last = performance.now();
  let fps = 60;
  let fpsAcc = 0, fpsN = 0, fpsT = 0;

  function frame(now) {
    requestAnimationFrame(frame);

    let dtMs = now - last;
    last = now;
    dtMs = Math.min(dtMs, 50);
    let dt = dtMs / 1000.0;

    fpsT += dtMs;
    fpsAcc += 1000.0 / Math.max(1, dtMs);
    fpsN += 1;
    if (fpsT > 350) {
      fps = fpsAcc / fpsN;
      fpsAcc = 0; fpsN = 0; fpsT = 0;
    }

    const pal = palettes[state.regime];

    if (!state.running || state.paused || state.over) {
      drawBackground(pal, dt);
      drawEntities(pal);
      compositeBloom();
      updateHud(fps);
      return;
    }

    if (state.hitStop > 0) {
      state.hitStop = Math.max(0, state.hitStop - dt);
      dt *= 0.18;
    }

    state.t += dt;

    state.regimeTime += dt;
    if (state.regimeTime >= state.regimeEvery) {
      state.regimeTime = 0;
      state.regime = (state.regime === "riskOn") ? "riskOff" : "riskOn";
      state.shake = Math.min(18, state.shake + 12);
      const col = (state.regime === "riskOn") ? "rgba(255,140,230,0.40)" : "rgba(120,210,255,0.35)";
      addParticles(W*0.5, H*0.40, col, 70, 1.6);
    }

    if (!state.intermission) {
      state.waveTime += dt;
      if (state.waveTime >= state.waveDuration) {
        state.wave += 1;
        state.waveTime = 0;
        setIntermission(true);
      }
    } else {
      state.intermissionTime += dt;
      if (enemies.length === 0 && state.intermissionTime >= state.intermissionDuration) setIntermission(false);
    }

    state.heat = Math.max(0, state.heat - 18 * dt);

    player.fireCd = Math.max(0, player.fireCd - dt);
    player.dashCd = Math.max(0, player.dashCd - dt);

    if (player.shieldMax > 0 && player.shieldRegen > 0) {
      player.shield = Math.min(player.shieldMax, player.shield + player.shieldRegen * dt);
    }

    const basePressure = 0.65 + state.wave * 0.09;
    const regimeMul = (state.regime === "riskOff") ? 1.12 : 0.98;
    const pressure = basePressure * regimeMul * D.spawnMul;

    state.enemyTimer -= dt;
    state.pickupTimer -= dt;

    if (!state.intermission) {
      const enemyEvery = clamp(1.05 - pressure * 0.12, 0.35, 0.95);
      if (state.enemyTimer <= 0) {
        spawnEnemy();
        if (state.rng() < 0.10 + state.wave * 0.01) spawnEnemy();
        state.enemyTimer = enemyEvery;
        state.heat = Math.min(240, state.heat + 10);
      }

      const pickupEvery = clamp(4.2 - state.wave * 0.15, 1.8, 3.8);
      if (state.pickupTimer <= 0) {
        spawnPickup();
        state.pickupTimer = pickupEvery;
      }
    }

    const accel = 2100;
    const maxV = 520 + state.wave * 7;
    const friction = 0.86;

    let ax = 0, ay = 0;
    if (keys.has("w") || keys.has("arrowup")) ay -= 1;
    if (keys.has("s") || keys.has("arrowdown")) ay += 1;
    if (keys.has("a") || keys.has("arrowleft")) ax -= 1;
    if (keys.has("d") || keys.has("arrowright")) ax += 1;

    player.vx += ax * accel * dt;
    player.vy += ay * accel * dt;

    player.vx = clamp(player.vx, -maxV, maxV);
    player.vy = clamp(player.vy, -maxV, maxV);

    player.x += player.vx * dt;
    player.y += player.vy * dt;

    player.vx *= Math.pow(friction, dt * 60);
    player.vy *= Math.pow(friction, dt * 60);

    player.x = clamp(player.x, 40, W - 40);
    player.y = clamp(player.y, 70, H - 60);

    if (mouse.down) shoot();

    for (let i = bullets.length - 1; i >= 0; i--) {
      const b = bullets[i];
      b.x += b.vx * dt;
      b.y += b.vy * dt;
      b.life -= dt;
      if (b.life <= 0 || b.x < -60 || b.x > W+60 || b.y < -60 || b.y > H+60) bullets.splice(i, 1);
    }

    for (const e of enemies) {
      const dx = player.x - e.x;
      const dy = player.y - e.y;
      const d = Math.hypot(dx, dy) + 1e-6;

      const baseSp = (e.tier === 2 ? 170 : 205) + state.wave * 6;
      const sp = baseSp * D.enemySpMul + (state.regime === "riskOff" ? 18 : 0);

      e.vx = lerp(e.vx, (dx / d) * sp, 0.08);
      e.vy = lerp(e.vy, (dy / d) * sp, 0.08);
      e.x += e.vx * dt;
      e.y += e.vy * dt;
      e.hit = Math.max(0, e.hit - 3.8 * dt);
    }

    for (let i = enemies.length - 1; i >= 0; i--) {
      const e = enemies[i];
      for (let j = bullets.length - 1; j >= 0; j--) {
        const b = bullets[j];
        if (circleHit(e.x, e.y, e.r, b.x, b.y, b.r)) {
          e.hp -= b.dmg;
          e.hit = 1.0;
          bullets.splice(j, 1);

          addParticles(b.x, b.y, pal.bullet, 10, 0.9);

          if (e.hp <= 0) {
            enemies.splice(i, 1);
            addParticles(e.x, e.y, pal.enemy, 44, 1.35);
            state.shake = Math.min(22, state.shake + 10);
            state.score += (e.tier === 2 ? 460 : 240) + state.wave * 10;
            state.alpha += Math.floor((e.tier === 2 ? 10 : 6) * D.alphaMul);
            state.hitStop = Math.min(0.05, state.hitStop + 0.020);
            if (state.rng() < 0.18) pickups.push({ x: e.x, y: e.y, r: 12, kind: "alpha", t: 0 });
          }
          break;
        }
      }
    }

    for (let i = enemies.length - 1; i >= 0; i--) {
      const e = enemies[i];
      if (circleHit(e.x, e.y, e.r, player.x, player.y, player.r + 4)) {
        enemies.splice(i, 1);
        const dmgBase = (e.tier === 2 ? 22 : 14) + Math.floor(state.wave * 0.8);
        let remaining = Math.floor(dmgBase * D.dmgMul);

        if (player.shield > 0) {
          const sTake = Math.min(player.shield, remaining);
          player.shield -= sTake;
          remaining -= sTake;
        }
        player.hp -= remaining;

        addParticles(player.x, player.y, pal.enemy, 56, 1.5);
        state.shake = Math.min(28, state.shake + 16);
        state.heat = Math.min(240, state.heat + 40);

        if (player.hp <= 0) {
          gameOver();
          break;
        }
      }
    }

    for (let i = pickups.length - 1; i >= 0; i--) {
      const p = pickups[i];
      if (circleHit(p.x, p.y, p.r, player.x, player.y, player.r + 8)) {
        pickups.splice(i, 1);
        if (p.kind === "alpha") {
          state.alpha += Math.floor(12 * D.alphaMul);
          state.score += 190;
          addParticles(p.x, p.y, pal.alpha, 26, 1.2);
        } else {
          player.hp = Math.min(player.maxHp, player.hp + 26);
          state.score += 120;
          addParticles(p.x, p.y, pal.neon1, 26, 1.2);
        }
      }
    }

    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vx *= Math.pow(0.97, dt * 60);
      p.vy *= Math.pow(0.97, dt * 60);
      p.life -= dt;
      if (p.life <= 0) particles.splice(i, 1);
    }

    state.shake = Math.max(0, state.shake * Math.pow(0.88, dt * 60));
    state.score += Math.floor(2 * dt * 60);

    if (state.score > highScore) { highScore = state.score; saveHigh(highScore); }

    drawBackground(palettes[state.regime], dt);

    const shMag = state.shake * Q.shakeMul;
    const dx = (state.rng() * 2 - 1) * shMag;
    const dy = (state.rng() * 2 - 1) * shMag;

    sctx.save();
    sctx.translate(dx, dy);
    drawEntities(palettes[state.regime]);
    sctx.restore();

    if (state.intermission) {
      sctx.save();
      sctx.globalAlpha = 0.92;
      sctx.fillStyle = "rgba(0,0,0,0.32)";
      sctx.fillRect(W*0.5 - 160, 18, 320, 34);
      sctx.strokeStyle = "rgba(255,255,255,0.12)";
      sctx.strokeRect(W*0.5 - 160, 18, 320, 34);
      sctx.fillStyle = "rgba(255,255,255,0.86)";
      sctx.font = "700 14px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial";
      sctx.textAlign = "center";
      sctx.textBaseline = "middle";
      sctx.fillText("INTERMISSION: BUY UPGRADES", W*0.5, 35);
      sctx.restore();
    }

    compositeBloom();
    updateHud(fps);
  }

  hudR.textContent = `High: ${highScore}`;
  reset(CONFIG.seed || 42069);
  requestAnimationFrame(frame);
})();
</script>
</body>
</html>
"""

html = html.replace("__CONFIG_JSON__", json.dumps(cfg))

components.html(html, height=860, scrolling=False)
