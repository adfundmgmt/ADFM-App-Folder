import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Neon Runout", layout="wide")
st.title("Neon Runout")
st.caption("Controls: A/D or ←/→ to change lanes. F or Click to fire. Space to multi-fire (cooldown). R to restart. Click the game to focus.")

html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body { margin:0; padding:0; background: transparent; }
    .wrap { width:100%; display:flex; justify-content:center; align-items:center; }
    .frame { position: relative; width: min(1180px, 98vw); }
    canvas {
      width: 100%;
      height: auto;
      border-radius: 22px;
      border: 1px solid rgba(255,255,255,0.10);
      background: linear-gradient(180deg, rgba(9,10,16,1), rgba(6,7,10,1));
      box-shadow: 0 18px 60px rgba(0,0,0,0.55);
      outline: none;
      display:block;
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
    }
    .card {
      width: min(620px, 92%);
      border-radius: 20px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.07);
      padding: 18px 18px 16px 18px;
      box-shadow: 0 18px 70px rgba(0,0,0,0.60);
      text-align:left;
    }
    .title { font-size: 22px; font-weight: 800; margin: 0 0 8px 0; letter-spacing: 0.2px; }
    .sub { opacity: 0.90; margin: 0 0 12px 0; line-height: 1.42; }
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
  </style>
</head>
<body>
  <div class="wrap">
    <div class="frame">
      <div class="hud">
        <div class="pill" id="hudLeft">Score: 0 | Coins: 0 | Combo: 0</div>
        <div class="pill" id="hudMid">Speed: 0.0x | Balls: 1</div>
        <div class="pill" id="hudRight">High: 0</div>
      </div>

      <canvas id="game" width="1180" height="660" tabindex="0"></canvas>

      <div class="overlay" id="overlay" style="display:flex;">
        <div class="card">
          <div class="title">Neon Runout</div>
          <p class="sub">
            Subway Surfers lanes plus Brick Breaker bricks.<br/>
            Brick walls approach. Dodge them, or break them with ricocheting balls.
          </p>
          <p class="sub">
            <span class="kbd">←</span><span class="kbd">→</span> or <span class="kbd">A</span><span class="kbd">D</span> change lanes |
            <span class="kbd">F</span> or <span class="kbd">Click</span> fire |
            <span class="kbd">Space</span> burst fire |
            <span class="kbd">R</span> restart
          </p>
          <p class="sub" style="opacity:0.78; margin:0;">
            Click the game area, then press <span class="kbd">F</span> to start firing and survive.
          </p>
        </div>
      </div>
    </div>
  </div>

<script>
(() => {
  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d");
  const hudLeft = document.getElementById("hudLeft");
  const hudMid = document.getElementById("hudMid");
  const hudRight = document.getElementById("hudRight");
  const overlay = document.getElementById("overlay");

  const DPR = Math.max(1, Math.min(2.0, window.devicePixelRatio || 1));

  // Internal resolution stays stable. CSS scales it to container.
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
    const v = localStorage.getItem("neon_runout_high");
    const n = v ? parseInt(v, 10) : 0;
    return Number.isFinite(n) ? n : 0;
  }
  function saveHigh(v) {
    localStorage.setItem("neon_runout_high", String(v));
  }

  let highScore = loadHigh();

  // Track geometry
  const track = {
    topY: H * 0.10,
    botY: H * 0.96,
    topW: W * 0.22,
    botW: W * 0.64
  };

  const lanes = [-1, 0, 1];
  function laneX(lane, y) {
    // perspective: width grows with y
    const p = clamp((y - track.topY) / (track.botY - track.topY), 0, 1);
    const w = lerp(track.topW, track.botW, p);
    const laneGap = w / 3;
    return W * 0.5 + lane * laneGap;
  }

  // Visual palette
  const palette = {
    bg1: "rgba(8,10,14,1)",
    bg2: "rgba(4,5,7,1)",
    neonA: "rgba(92,255,176,0.92)",
    neonB: "rgba(120,170,255,0.85)",
    neonC: "rgba(255,120,210,0.75)",
    danger: "rgba(255,92,92,0.92)",
    coin: "rgba(255,212,92,0.95)",
    line: "rgba(255,255,255,0.08)",
    glow: "rgba(255,255,255,0.05)",
  };

  // Game state
  const state = {
    seed: 42069,
    rng: mulberry32(42069),
    running: false,
    gameOver: false,
    t: 0,
    speed: 7.0,
    speedMult: 1.0,
    score: 0,
    coins: 0,
    combo: 0,
    comboTimer: 0,
    shake: 0,
    spawnWallTimer: 0,
    spawnCoinTimer: 0,
    fireCooldown: 0,
    burstCooldown: 0,
    ballsMax: 3,
    ballsOwned: 1,
  };

  const player = {
    lane: 0,
    laneTarget: 0,
    y: H * 0.86,
    w: 64,
    h: 82,
    invuln: 0,
  };

  let balls = [];
  let bricks = [];
  let coins = [];
  let particles = [];
  let stars = [];

  function initStars() {
    stars = [];
    for (let i = 0; i < 120; i++) {
      stars.push({
        x: state.rng() * W,
        y: state.rng() * H,
        r: 0.6 + state.rng() * 1.6,
        a: 0.15 + state.rng() * 0.35,
        s: 0.2 + state.rng() * 0.9
      });
    }
  }

  function reset(seed = 42069) {
    state.seed = seed;
    state.rng = mulberry32(seed);
    state.running = false;
    state.gameOver = false;
    state.t = 0;
    state.speed = 7.0;
    state.speedMult = 1.0;
    state.score = 0;
    state.coins = 0;
    state.combo = 0;
    state.comboTimer = 0;
    state.shake = 0;
    state.spawnWallTimer = 10;
    state.spawnCoinTimer = 20;
    state.fireCooldown = 0;
    state.burstCooldown = 0;
    state.ballsOwned = 1;

    player.lane = 0;
    player.laneTarget = 0;
    player.invuln = 0;

    balls = [];
    bricks = [];
    coins = [];
    particles = [];

    initStars();

    overlay.style.display = "flex";
    overlay.innerHTML = `
      <div class="card">
        <div class="title">Neon Runout</div>
        <p class="sub">
          Subway Surfers lanes plus Brick Breaker bricks.<br/>
          Brick walls approach. Dodge them, or break them with ricocheting balls.
        </p>
        <p class="sub">
          <span class="kbd">←</span><span class="kbd">→</span> or <span class="kbd">A</span><span class="kbd">D</span> change lanes |
          <span class="kbd">F</span> or <span class="kbd">Click</span> fire |
          <span class="kbd">Space</span> burst fire |
          <span class="kbd">R</span> restart
        </p>
        <p class="sub" style="opacity:0.78; margin:0;">
          Click the game area, then press <span class="kbd">F</span> to start firing and survive.
        </p>
      </div>
    `;
    updateHud();
  }

  function start() {
    if (state.gameOver) return;
    state.running = true;
    overlay.style.display = "none";
    canvas.focus();
    if (balls.length === 0) {
      spawnBall(true);
    }
  }

  function endGame() {
    state.running = false;
    state.gameOver = true;
    if (state.score > highScore) {
      highScore = state.score;
      saveHigh(highScore);
    }
    updateHud();
    overlay.style.display = "flex";
    overlay.innerHTML = `
      <div class="card">
        <div class="title">Run ended</div>
        <p class="sub">
          Score: <b>${state.score}</b> | Coins: <b>${state.coins}</b> | High: <b>${highScore}</b><br/>
          Press <span class="kbd">R</span> to restart, or <span class="kbd">F</span> to play again.
        </p>
        <p class="sub" style="opacity:0.78; margin:0;">
          Tip: you can dodge walls, but breaking the “full-width” walls early is cheaper than panic lane changes.
        </p>
      </div>
    `;
  }

  function updateHud() {
    hudLeft.textContent = `Score: ${state.score} | Coins: ${state.coins} | Combo: ${state.combo}`;
    hudMid.textContent = `Speed: ${state.speedMult.toFixed(2)}x | Balls: ${state.ballsOwned}`;
    hudRight.textContent = `High: ${highScore}`;
  }

  function addParticles(x, y, color, n=10, spread=1.0) {
    for (let i = 0; i < n; i++) {
      const a = (state.rng() * Math.PI * 2);
      const sp = (1.2 + state.rng() * 4.5) * spread;
      particles.push({
        x, y,
        vx: Math.cos(a) * sp,
        vy: Math.sin(a) * sp,
        life: 18 + Math.floor(state.rng() * 18),
        r: 1.5 + state.rng() * 2.8,
        c: color,
      });
    }
  }

  function roundRect(x, y, w, h, r) {
    const rr = Math.min(r, w/2, h/2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function spawnBall(attach=false) {
    const y = player.y - player.h * 0.55;
    const x = laneX(player.lane, player.y);
    const baseV = 10.8;
    const ang = (-Math.PI/2) + (state.rng() * 0.38 - 0.19); // mostly up
    const vx = Math.cos(ang) * baseV;
    const vy = Math.sin(ang) * baseV;

    balls.push({
      x, y,
      vx: attach ? 0 : vx,
      vy: attach ? 0 : vy,
      r: 10,
      attached: attach,
      trail: [],
      ttl: 999999,
    });
  }

  function fireBall(single=true) {
    if (!state.running) start();

    if (state.fireCooldown > 0) return;

    // If we have attached balls, launch them. Else create a new ball if allowed.
    let launched = 0;
    for (const b of balls) {
      if (b.attached) {
        const baseV = 12.0;
        const ang = (-Math.PI/2) + (state.rng() * 0.32 - 0.16);
        b.vx = Math.cos(ang) * baseV;
        b.vy = Math.sin(ang) * baseV;
        b.attached = false;
        launched++;
        if (single) break;
      }
    }
    if (launched === 0) {
      // Add a new ball if we have capacity
      if (balls.length < state.ballsOwned) {
        spawnBall(false);
        launched++;
      }
    }

    if (launched > 0) {
      state.fireCooldown = 10;
    }
  }

  function burstFire() {
    if (!state.running) start();
    if (state.burstCooldown > 0) return;

    // temporary multi-fire
    const shots = Math.min(3, state.ballsOwned + 1);
    for (let i = 0; i < shots; i++) {
      // attach -> launch for consistent starting position
      const b = {
        x: laneX(player.lane, player.y),
        y: player.y - player.h * 0.55,
        vx: 0,
        vy: 0,
        r: 10,
        attached: true,
        trail: [],
        ttl: 260, // expires
      };
      balls.push(b);
    }
    state.burstCooldown = 160;
    fireBall(false);
  }

  function spawnCoinLine() {
    const y = track.topY - 60;
    const lane = lanes[Math.floor(state.rng() * lanes.length)];
    // small arc line
    const count = 5 + Math.floor(state.rng() * 5);
    for (let i = 0; i < count; i++) {
      coins.push({
        lane,
        y: y - i * 26,
        r: 12,
        phase: state.rng() * Math.PI * 2
      });
    }
  }

  function spawnBrickWall() {
    // A "wall" is a set of bricks placed across lanes with potential gaps.
    // Some walls are full-width "must break or dodge perfectly" depending on pattern.
    const y = track.topY - 140;
    const p = clamp((state.speedMult - 1.0) / 2.5, 0, 1);

    // hp scales with time
    const baseHp = 1 + Math.floor(p * 3.0) + (state.rng() < 0.20 ? 1 : 0);

    // Pattern types:
    // 0: three bricks, one per lane
    // 1: two bricks (one gap)
    // 2: full-width slab (3 bricks with higher hp)
    // 3: staggered double row
    const roll = state.rng();
    let type = 0;
    if (roll < 0.38) type = 0;
    else if (roll < 0.68) type = 1;
    else if (roll < 0.86) type = 2;
    else type = 3;

    const makeBrick = (lane, yy, hp, wMul=1.0) => {
      // dimensions depend on perspective at that y
      const x = laneX(lane, yy);
      const laneW = (laneX(1, yy) - laneX(0, yy));
      const bw = laneW * 0.72 * wMul;
      const bh = 46;
      bricks.push({
        lane,
        x,
        y: yy,
        w: bw,
        h: bh,
        hp,
        maxHp: hp,
        glow: 0,
        kind: "brick",
      });
    };

    if (type === 0) {
      for (const lane of lanes) makeBrick(lane, y, baseHp);
    } else if (type === 1) {
      // pick a gap lane
      const gap = lanes[Math.floor(state.rng() * 3)];
      for (const lane of lanes) {
        if (lane === gap) continue;
        makeBrick(lane, y, baseHp);
      }
    } else if (type === 2) {
      // full width, slightly higher hp
      const hp = baseHp + 1;
      for (const lane of lanes) makeBrick(lane, y, hp);
    } else {
      // staggered: two rows, first has a gap, second has different gap
      const gap1 = lanes[Math.floor(state.rng() * 3)];
      const gap2 = lanes[Math.floor(state.rng() * 3)];
      for (const lane of lanes) if (lane !== gap1) makeBrick(lane, y, baseHp);
      for (const lane of lanes) if (lane !== gap2) makeBrick(lane, y - 58, Math.max(1, baseHp - 1));
    }
  }

  function circleRectHit(cx, cy, cr, rx, ry, rw, rh) {
    const nx = clamp(cx, rx, rx + rw);
    const ny = clamp(cy, ry, ry + rh);
    const dx = cx - nx;
    const dy = cy - ny;
    return (dx * dx + dy * dy) <= cr * cr;
  }

  function reflectBallOnRect(ball, rx, ry, rw, rh) {
    // Find penetration axis by comparing distances to sides
    const cx = ball.x, cy = ball.y;
    const left = Math.abs(cx - rx);
    const right = Math.abs(cx - (rx + rw));
    const top = Math.abs(cy - ry);
    const bot = Math.abs(cy - (ry + rh));
    const m = Math.min(left, right, top, bot);

    if (m === left || m === right) ball.vx *= -1;
    else ball.vy *= -1;

    // add a little randomness for feel
    ball.vx += (state.rng() * 0.6 - 0.3);
    ball.vy += (state.rng() * 0.4 - 0.2);

    // clamp speed
    const sp = Math.hypot(ball.vx, ball.vy);
    const target = 12.5 + state.speedMult * 1.1;
    const k = target / Math.max(6.0, sp);
    ball.vx *= k;
    ball.vy *= k;
  }

  function drawBackground() {
    ctx.clearRect(0, 0, W, H);

    // Deep gradient
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, "rgba(10,12,18,1)");
    g.addColorStop(1, "rgba(5,6,9,1)");
    ctx.fillStyle = g;
    ctx.fillRect(0,0,W,H);

    // Nebula glows
    const g1 = ctx.createRadialGradient(W*0.22, H*0.18, 20, W*0.22, H*0.18, 840);
    g1.addColorStop(0, "rgba(110,160,255,0.20)");
    g1.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g1;
    ctx.fillRect(0,0,W,H);

    const g2 = ctx.createRadialGradient(W*0.80, H*0.25, 20, W*0.80, H*0.25, 760);
    g2.addColorStop(0, "rgba(255,110,210,0.16)");
    g2.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g2;
    ctx.fillRect(0,0,W,H);

    // Stars
    for (const s of stars) {
      s.y += s.s * (0.7 + state.speedMult * 0.35);
      if (s.y > H) { s.y = -5; s.x = state.rng() * W; }
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI*2);
      ctx.fillStyle = `rgba(255,255,255,${s.a})`;
      ctx.fill();
    }

    // Track
    ctx.save();
    ctx.globalAlpha = 1.0;

    const topY = track.topY;
    const botY = track.botY;
    const topW = track.topW;
    const botW = track.botW;

    ctx.beginPath();
    ctx.moveTo(W/2 - topW/2, topY);
    ctx.lineTo(W/2 + topW/2, topY);
    ctx.lineTo(W/2 + botW/2, botY);
    ctx.lineTo(W/2 - botW/2, botY);
    ctx.closePath();
    ctx.fillStyle = "rgba(255,255,255,0.035)";
    ctx.fill();

    // Lane lines
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 2;
    for (let i = -1; i <= 1; i++) {
      const xTop = W/2 + i * (topW/3);
      const xBot = W/2 + i * (botW/3);
      ctx.beginPath();
      ctx.moveTo(xTop, topY);
      ctx.lineTo(xBot, botY);
      ctx.stroke();
    }

    // Motion dashes
    const dashCount = 28;
    const scroll = (state.t * (0.028 * state.speedMult)) % 1.0;
    for (let k = 0; k < dashCount; k++) {
      const p = k / dashCount;
      const yy = lerp(topY, botY, (p + scroll) % 1.0);
      const ww = lerp(topW, botW, clamp((yy - topY) / (botY - topY), 0, 1));
      const dashW = 14 + (ww / botW) * 46;
      const dashH = 4 + (ww / botW) * 8;
      ctx.fillStyle = "rgba(255,255,255,0.06)";
      ctx.fillRect(W/2 - dashW/2, yy, dashW, dashH);
    }

    ctx.restore();
  }

  function drawPlayer() {
    const y = player.y;
    const x = laneX(player.lane, y);

    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.55)";
    ctx.shadowBlur = 18;

    // neon glow under
    ctx.beginPath();
    ctx.ellipse(x, y + 18, 52, 12, 0, 0, Math.PI*2);
    ctx.fillStyle = "rgba(92,255,176,0.18)";
    ctx.fill();

    // body
    const px = x - player.w/2;
    const py = y - player.h;
    roundRect(px, py, player.w, player.h, 22);

    const g = ctx.createLinearGradient(px, py, px + player.w, py + player.h);
    g.addColorStop(0, "rgba(92,255,176,0.92)");
    g.addColorStop(1, "rgba(120,170,255,0.80)");
    ctx.fillStyle = g;
    ctx.fill();

    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // visor
    roundRect(px + 10, py + 14, player.w - 20, 18, 10);
    ctx.fillStyle = "rgba(0,0,0,0.30)";
    ctx.fill();

    // accents
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = "rgba(255,110,210,0.22)";
    ctx.fillRect(px + 12, py + player.h - 18, player.w - 24, 6);

    ctx.restore();
  }

  function brickColor(hp, maxHp) {
    const t = hp / Math.max(1, maxHp);
    // high hp -> more purple/pink, low hp -> more blue/teal
    const a = 0.86;
    const r = Math.floor(lerp(110, 255, 1 - t));
    const g = Math.floor(lerp(170, 120, 1 - t));
    const b = Math.floor(lerp(255, 210, 1 - t));
    return `rgba(${r},${g},${b},${a})`;
  }

  function drawBricks() {
    for (const br of bricks) {
      if (br.hp <= 0) continue;

      const rx = br.x - br.w/2;
      const ry = br.y - br.h/2;

      const glow = clamp(br.glow, 0, 1);
      ctx.save();
      ctx.shadowColor = "rgba(120,170,255,0.55)";
      ctx.shadowBlur = 22 + glow * 18;

      roundRect(rx, ry, br.w, br.h, 14);
      const fill = brickColor(br.hp, br.maxHp);

      const gg = ctx.createLinearGradient(rx, ry, rx + br.w, ry + br.h);
      gg.addColorStop(0, fill);
      gg.addColorStop(1, "rgba(255,255,255,0.10)");
      ctx.fillStyle = gg;
      ctx.fill();

      ctx.strokeStyle = "rgba(255,255,255,0.16)";
      ctx.lineWidth = 2;
      ctx.stroke();

      // hp text
      ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(10,12,18,0.68)";
      ctx.font = "bold 16px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(br.hp), br.x, br.y + 1);

      ctx.restore();
    }
  }

  function drawCoins() {
    for (const c of coins) {
      const y = c.y;
      const x = laneX(c.lane, y) + Math.sin(state.t * 0.05 + c.phase) * 5;

      ctx.save();
      ctx.shadowColor = "rgba(255,212,92,0.55)";
      ctx.shadowBlur = 20;

      ctx.beginPath();
      ctx.arc(x, y, c.r, 0, Math.PI*2);
      ctx.fillStyle = palette.coin;
      ctx.fill();

      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(x, y, c.r * 0.45, 0, Math.PI*2);
      ctx.strokeStyle = "rgba(0,0,0,0.22)";
      ctx.stroke();

      ctx.restore();
    }
  }

  function drawBalls() {
    for (const b of balls) {
      // trail
      ctx.save();
      ctx.globalAlpha = 0.65;
      for (let i = 0; i < b.trail.length; i++) {
        const p = b.trail[i];
        const a = (i / b.trail.length) * 0.35;
        ctx.beginPath();
        ctx.arc(p.x, p.y, b.r * (0.65 + i*0.02), 0, Math.PI*2);
        ctx.fillStyle = `rgba(255,110,210,${a})`;
        ctx.fill();
      }
      ctx.restore();

      ctx.save();
      ctx.shadowColor = "rgba(255,110,210,0.55)";
      ctx.shadowBlur = 26;

      ctx.beginPath();
      ctx.arc(b.x, b.y, b.r, 0, Math.PI*2);
      const g = ctx.createRadialGradient(b.x - 4, b.y - 6, 2, b.x, b.y, b.r * 2.2);
      g.addColorStop(0, "rgba(255,255,255,0.92)");
      g.addColorStop(0.35, "rgba(255,140,230,0.75)");
      g.addColorStop(1, "rgba(255,110,210,0.25)");
      ctx.fillStyle = g;
      ctx.fill();

      ctx.restore();
    }
  }

  function drawParticles() {
    for (const p of particles) {
      ctx.save();
      ctx.globalAlpha = clamp(p.life / 26, 0, 1);
      ctx.shadowColor = p.c;
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
      ctx.fillStyle = p.c;
      ctx.fill();
      ctx.restore();
    }
  }

  function screenShake() {
    if (state.shake <= 0) return {dx: 0, dy: 0};
    const mag = state.shake;
    return {
      dx: (state.rng() * 2 - 1) * mag,
      dy: (state.rng() * 2 - 1) * mag
    };
  }

  function tick() {
    requestAnimationFrame(tick);

    const dt = 1.0;
    state.t += dt;

    // cool downs
    state.fireCooldown = Math.max(0, state.fireCooldown - dt);
    state.burstCooldown = Math.max(0, state.burstCooldown - dt);
    player.invuln = Math.max(0, player.invuln - dt);

    if (state.comboTimer > 0) state.comboTimer -= dt;
    else state.combo = 0;

    drawBackground();

    // When idle, still render scene
    if (!state.running) {
      ctx.save();
      const sh = screenShake();
      ctx.translate(sh.dx, sh.dy);
      drawCoins();
      drawBricks();
      drawBalls();
      drawPlayer();
      drawParticles();
      ctx.restore();
      updateHud();
      return;
    }

    // Difficulty curve
    state.speedMult = Math.min(2.75, 1.0 + state.t * 0.0014);
    const speed = state.speed * state.speedMult;

    // Spawning cadence
    state.spawnWallTimer -= dt;
    state.spawnCoinTimer -= dt;

    const wallEvery = clamp(44 - state.speedMult * 14, 18, 40);
    const coinEvery = clamp(60 - state.speedMult * 18, 26, 54);

    if (state.spawnWallTimer <= 0) {
      spawnBrickWall();
      if (state.rng() < 0.16 + state.speedMult * 0.05) spawnBrickWall();
      state.spawnWallTimer = wallEvery;
    }

    if (state.spawnCoinTimer <= 0) {
      spawnCoinLine();
      state.spawnCoinTimer = coinEvery;
    }

    // Smooth lane movement
    player.lane = lerp(player.lane, player.laneTarget, 0.22);

    // Move world down (runner illusion)
    const worldDy = speed * 1.85;
    for (const br of bricks) br.y += worldDy;
    for (const c of coins) c.y += worldDy;

    // Cull off-screen
    bricks = bricks.filter(br => br.y < H + 200 && br.hp > 0);
    coins = coins.filter(c => c.y < H + 140);

    // Ball update
    const leftWallX = (y) => laneX(-1, y) - (laneX(0, y) - laneX(-1, y)) * 0.52;
    const rightWallX = (y) => laneX(1, y) + (laneX(1, y) - laneX(0, y)) * 0.52;

    for (let i = balls.length - 1; i >= 0; i--) {
      const b = balls[i];

      if (b.attached) {
        b.x = laneX(player.lane, player.y);
        b.y = player.y - player.h * 0.55;
      } else {
        b.x += b.vx;
        b.y += b.vy;

        // trail
        b.trail.unshift({x: b.x, y: b.y});
        if (b.trail.length > 10) b.trail.pop();

        // Bounce side walls based on local track width at that y
        const lx = leftWallX(b.y);
        const rx = rightWallX(b.y);
        if (b.x - b.r < lx) { b.x = lx + b.r; b.vx *= -1; addParticles(b.x, b.y, "rgba(120,170,255,0.30)", 6, 0.8); }
        if (b.x + b.r > rx) { b.x = rx - b.r; b.vx *= -1; addParticles(b.x, b.y, "rgba(120,170,255,0.30)", 6, 0.8); }

        // Bounce top (keep ball in arena)
        if (b.y - b.r < track.topY - 120) {
          b.y = track.topY - 120 + b.r;
          b.vy *= -1;
        }

        // Bounce off player (acts like paddle)
        const px = laneX(player.lane, player.y) - player.w/2;
        const py = player.y - player.h;
        const rxp = px, ryp = py, rwp = player.w, rhp = player.h;
        if (circleRectHit(b.x, b.y, b.r, rxp, ryp, rwp, rhp) && b.vy > 0) {
          // angle based on horizontal offset
          const center = laneX(player.lane, player.y);
          const off = clamp((b.x - center) / (player.w * 0.55), -1, 1);
          const sp = 13.0 + state.speedMult * 1.25;
          b.vx = off * sp * 0.85;
          b.vy = -Math.sqrt(Math.max(1.0, sp*sp - b.vx*b.vx));
          state.combo = Math.min(99, state.combo + 1);
          state.comboTimer = 70;
          addParticles(b.x, b.y, "rgba(92,255,176,0.35)", 10, 1.0);
        }

        // Brick collisions
        for (const br of bricks) {
          if (br.hp <= 0) continue;
          const rxB = br.x - br.w/2;
          const ryB = br.y - br.h/2;
          if (circleRectHit(b.x, b.y, b.r, rxB, ryB, br.w, br.h)) {
            br.hp -= 1;
            br.glow = 1.0;
            reflectBallOnRect(b, rxB, ryB, br.w, br.h);

            const pop = br.hp <= 0;
            state.score += pop ? (220 + state.combo * 6) : (65 + state.combo * 2);
            state.shake = Math.min(10, state.shake + (pop ? 5 : 2));
            addParticles(br.x, br.y, pop ? "rgba(255,110,210,0.45)" : "rgba(120,170,255,0.35)", pop ? 22 : 12, pop ? 1.35 : 1.0);

            // reward: occasional extra ball
            if (pop && state.rng() < 0.06 && state.ballsOwned < state.ballsMax) {
              state.ballsOwned += 1;
              state.score += 180;
              addParticles(br.x, br.y, "rgba(92,255,176,0.45)", 18, 1.2);
            }
            break;
          }
        }

        // TTL and cull
        b.ttl -= dt;
        if (b.ttl <= 0 || b.y > H + 200) {
          balls.splice(i, 1);
        }
      }
    }

    // If we have fewer balls than owned, keep one attached as reload
    const attachedCount = balls.filter(b => b.attached).length;
    const totalCount = balls.length;
    if (totalCount < state.ballsOwned && attachedCount === 0) {
      spawnBall(true);
    }

    // Brick glow decay
    for (const br of bricks) br.glow = Math.max(0, br.glow - 0.06);

    // Coin pickup
    const plx = laneX(player.lane, player.y);
    const ply = player.y - 32;
    for (let i = coins.length - 1; i >= 0; i--) {
      const c = coins[i];
      const cx = laneX(c.lane, c.y);
      const cy = c.y;
      const dx = plx - cx;
      const dy = ply - cy;
      if ((dx*dx + dy*dy) < (c.r + 28) * (c.r + 28)) {
        coins.splice(i, 1);
        state.coins += 1;
        state.score += 120;
        addParticles(cx, cy, "rgba(255,212,92,0.45)", 16, 1.1);
      }
    }

    // Player collision with bricks (death)
    const pX = laneX(player.lane, player.y);
    const pxr = pX - player.w/2;
    const pyr = player.y - player.h;

    for (const br of bricks) {
      if (br.hp <= 0) continue;
      const rxB = br.x - br.w/2;
      const ryB = br.y - br.h/2;
      const hit = (pxr < rxB + br.w && pxr + player.w > rxB && pyr < ryB + br.h && pyr + player.h > ryB);
      if (hit) {
        // small mercy: if invuln, ignore; else die
        if (player.invuln <= 0) {
          state.shake = 14;
          addParticles(pX, player.y - 40, "rgba(255,92,92,0.55)", 44, 1.5);
          endGame();
        }
        break;
      }
    }

    // Particles
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.97;
      p.vy *= 0.97;
      p.life -= 1;
      if (p.life <= 0) particles.splice(i, 1);
    }

    // score tick
    state.score += Math.floor(1 + state.speedMult * 2);

    // shake decay
    state.shake = Math.max(0, state.shake * 0.88);

    // Render with shake
    ctx.save();
    const sh = screenShake();
    ctx.translate(sh.dx, sh.dy);

    drawCoins();
    drawBricks();
    drawBalls();
    drawPlayer();
    drawParticles();

    ctx.restore();

    // Update HUD and high score live
    if (state.score > highScore) {
      highScore = state.score;
      saveHigh(highScore);
    }
    updateHud();
  }

  // Controls
  function moveLeft() {
    player.laneTarget = clamp(player.laneTarget - 1, -1, 1);
  }
  function moveRight() {
    player.laneTarget = clamp(player.laneTarget + 1, -1, 1);
  }

  function onKeyDown(e) {
    const k = e.key.toLowerCase();

    if (k === "r") {
      e.preventDefault();
      reset(state.seed);
      return;
    }

    if (k === "f") {
      e.preventDefault();
      if (!state.running && !state.gameOver) start();
      if (state.gameOver) { reset(state.seed); start(); }
      fireBall(true);
      return;
    }

    if (k === " ") {
      e.preventDefault();
      if (!state.running && !state.gameOver) start();
      if (state.gameOver) { reset(state.seed); start(); }
      burstFire();
      return;
    }

    if (!state.running) return;

    if (k === "arrowleft" || k === "a") {
      e.preventDefault();
      moveLeft();
    } else if (k === "arrowright" || k === "d") {
      e.preventDefault();
      moveRight();
    }
  }

  canvas.addEventListener("keydown", onKeyDown);

  canvas.addEventListener("click", () => {
    canvas.focus();
    if (!state.running && !state.gameOver) start();
    else if (state.gameOver) { reset(state.seed); start(); }
    else fireBall(true);
  });

  // Touch: swipe lane, tap to fire
  let tStartX = null, tStartY = null, tLastTap = 0;
  canvas.addEventListener("touchstart", (e) => {
    const t0 = e.touches[0];
    tStartX = t0.clientX;
    tStartY = t0.clientY;
    canvas.focus();
    if (!state.running && !state.gameOver) start();
  }, {passive:true});

  canvas.addEventListener("touchend", (e) => {
    if (tStartX == null) return;
    const t0 = e.changedTouches[0];
    const dx = t0.clientX - tStartX;
    const dy = t0.clientY - tStartY;
    tStartX = null; tStartY = null;

    const ax = Math.abs(dx), ay = Math.abs(dy);

    if (ax > 35 && ax > ay) {
      if (dx < 0) moveLeft();
      else moveRight();
      return;
    }

    const now = performance.now();
    if (now - tLastTap < 260) {
      burstFire();
    } else {
      fireBall(true);
    }
    tLastTap = now;
  }, {passive:true});

  // Boot
  hudRight.textContent = `High: ${highScore}`;
  reset(42069);
  tick();
})();
</script>
</body>
</html>
"""

components.html(html, height=720, scrolling=False)
