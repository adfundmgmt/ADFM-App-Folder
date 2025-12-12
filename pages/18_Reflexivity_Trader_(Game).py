import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Runner Rush", layout="wide")

st.title("Runner Rush")
st.caption("Arrow keys or A/D to change lanes. Space or W to jump. R to restart.")

html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body { margin:0; padding:0; background: transparent; }
    .wrap {
      width: 100%;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    canvas {
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.10);
      background: linear-gradient(180deg, rgba(10,12,18,1), rgba(8,10,14,1));
      box-shadow: 0 12px 34px rgba(0,0,0,0.45);
      outline: none;
    }
    .hud {
      position: absolute;
      top: 18px;
      left: 18px;
      right: 18px;
      display: flex;
      justify-content: space-between;
      pointer-events: none;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      color: rgba(245,245,250,0.92);
      font-size: 14px;
    }
    .pill {
      padding: 8px 10px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.10);
      background: rgba(255,255,255,0.06);
      backdrop-filter: blur(6px);
    }
    .centerOverlay {
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
      width: min(520px, 92%);
      border-radius: 18px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06);
      padding: 18px 18px 16px 18px;
      box-shadow: 0 16px 44px rgba(0,0,0,0.50);
      text-align:left;
    }
    .title {
      font-size: 20px;
      font-weight: 750;
      margin: 0 0 8px 0;
    }
    .sub {
      opacity: 0.88;
      margin: 0 0 12px 0;
      line-height: 1.35;
    }
    .kbd {
      display:inline-block;
      padding: 2px 8px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(0,0,0,0.22);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
      margin-right: 6px;
    }
  </style>
</head>
<body>
  <div class="wrap" style="position:relative;">
    <div class="hud">
      <div class="pill" id="hudLeft">Score: 0 | Coins: 0</div>
      <div class="pill" id="hudRight">High: 0</div>
    </div>

    <canvas id="game" width="980" height="560" tabindex="0"></canvas>

    <div class="centerOverlay" id="overlay" style="display:flex;">
      <div class="card">
        <div class="title">Runner Rush</div>
        <p class="sub">
          Three lanes. Rising speed. Reflexes. Stay alive.<br/>
          <span class="kbd">←</span><span class="kbd">→</span> or <span class="kbd">A</span><span class="kbd">D</span> to change lanes,
          <span class="kbd">Space</span> or <span class="kbd">W</span> to jump,
          <span class="kbd">R</span> to restart.
        </p>
        <p class="sub" style="opacity:0.78; margin:0;">
          Click the game area, then press <span class="kbd">Space</span> to start.
        </p>
      </div>
    </div>
  </div>

<script>
(() => {
  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d");
  const hudLeft = document.getElementById("hudLeft");
  const hudRight = document.getElementById("hudRight");
  const overlay = document.getElementById("overlay");

  const W = canvas.width, H = canvas.height;

  const lanes = [-1, 0, 1];
  const laneX = (lane) => (W * 0.5) + lane * (W * 0.17);

  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  const loadHigh = () => {
    const v = localStorage.getItem("runner_rush_high");
    const n = v ? parseInt(v, 10) : 0;
    return Number.isFinite(n) ? n : 0;
  };
  const saveHigh = (v) => localStorage.setItem("runner_rush_high", String(v));

  let highScore = loadHigh();

  const state = {
    running: false,
    gameOver: false,
    t: 0,
    score: 0,
    coins: 0,
    speed: 7.0,
    spawnTimer: 0,
    coinTimer: 0,
    rng: mulberry32(42069),
  };

  const player = {
    lane: 0,
    y: H * 0.80,
    vy: 0,
    onGround: true,
    jumpStrength: 16.5,
    gravity: 0.95,
    w: 56,
    h: 70,
    slide: 0,
  };

  let obstacles = [];
  let coins = [];

  function mulberry32(a) {
    return function() {
      let t = a += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    }
  }

  function reset(runSeed = 42069) {
    state.running = false;
    state.gameOver = false;
    state.t = 0;
    state.score = 0;
    state.coins = 0;
    state.speed = 7.0;
    state.spawnTimer = 0;
    state.coinTimer = 0;
    state.rng = mulberry32(runSeed);

    player.lane = 0;
    player.y = H * 0.80;
    player.vy = 0;
    player.onGround = true;

    obstacles = [];
    coins = [];

    overlay.style.display = "flex";
    updateHud();
  }

  function updateHud() {
    hudLeft.textContent = `Score: ${state.score} | Coins: ${state.coins}`;
    hudRight.textContent = `High: ${highScore}`;
  }

  function start() {
    if (state.gameOver) return;
    state.running = true;
    overlay.style.display = "none";
    canvas.focus();
  }

  function endGame() {
    state.running = false;
    state.gameOver = true;
    if (state.score > highScore) {
      highScore = state.score;
      saveHigh(highScore);
    }
    updateHud();

    overlay.innerHTML = `
      <div class="card">
        <div class="title">Game Over</div>
        <p class="sub">
          Score: <b>${state.score}</b> | Coins: <b>${state.coins}</b> | High: <b>${highScore}</b><br/>
          Press <span class="kbd">R</span> to restart or <span class="kbd">Space</span> to replay.
        </p>
        <p class="sub" style="opacity:0.78; margin:0;">
          Tip: lane changes are cheaper than late jumps. Don’t drift into crowded lanes.
        </p>
      </div>
    `;
    overlay.style.display = "flex";
  }

  function spawnObstacle() {
    const lane = lanes[Math.floor(state.rng() * lanes.length)];
    const kindRoll = state.rng();
    const kind = kindRoll < 0.72 ? "barrier" : "train";
    const w = kind === "barrier" ? 62 : 78;
    const h = kind === "barrier" ? 58 : 120;

    obstacles.push({
      lane,
      x: laneX(lane),
      y: -140,
      w,
      h,
      kind,
    });
  }

  function spawnCoin() {
    const lane = lanes[Math.floor(state.rng() * lanes.length)];
    coins.push({
      lane,
      x: laneX(lane),
      y: -80,
      r: 14,
    });
  }

  function rectsOverlap(ax, ay, aw, ah, bx, by, bw, bh) {
    return ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;
  }

  function tick() {
    requestAnimationFrame(tick);

    const dt = 1.0; // fixed timestep

    drawBackground();

    if (!state.running) {
      drawScene();
      return;
    }

    state.t += dt;
    state.speed = Math.min(18.0, state.speed + 0.0028 * dt);
    state.score = Math.floor(state.score + (state.speed * 0.45));
    state.spawnTimer -= dt;
    state.coinTimer -= dt;

    const spawnEvery = clamp(34 - state.speed * 1.25, 14, 30);
    const coinEvery = clamp(42 - state.speed * 1.1, 18, 36);

    if (state.spawnTimer <= 0) {
      spawnObstacle();
      if (state.rng() < 0.25) spawnObstacle();
      state.spawnTimer = spawnEvery;
    }
    if (state.coinTimer <= 0) {
      spawnCoin();
      if (state.rng() < 0.20) spawnCoin();
      state.coinTimer = coinEvery;
    }

    // Player physics
    if (!player.onGround) {
      player.vy += player.gravity;
      player.y += player.vy;
      const groundY = H * 0.80;
      if (player.y >= groundY) {
        player.y = groundY;
        player.vy = 0;
        player.onGround = true;
      }
    }

    // Move obstacles and coins
    const moveY = state.speed * 1.9;
    for (const o of obstacles) o.y += moveY;
    for (const c of coins) c.y += moveY;

    obstacles = obstacles.filter(o => o.y < H + 180);
    coins = coins.filter(c => c.y < H + 120);

    // Collisions
    const px = laneX(player.lane) - player.w / 2;
    const py = player.y - player.h;
    for (const o of obstacles) {
      const ox = o.x - o.w / 2;
      const oy = o.y - o.h / 2;
      if (rectsOverlap(px, py, player.w, player.h, ox, oy, o.w, o.h)) {
        endGame();
        break;
      }
    }

    // Coin pickups
    for (let i = coins.length - 1; i >= 0; i--) {
      const c = coins[i];
      if (c.lane !== player.lane) continue;
      const cx = c.x, cy = c.y;
      const dx = (laneX(player.lane) - cx);
      const dy = (player.y - 30 - cy);
      if ((dx*dx + dy*dy) < (c.r + 30) * (c.r + 30)) {
        state.coins += 1;
        state.score += 250;
        coins.splice(i, 1);
      }
    }

    updateHud();
    drawScene();
  }

  function drawBackground() {
    ctx.clearRect(0,0,W,H);

    // glow bands
    const g1 = ctx.createRadialGradient(W*0.2, H*0.15, 30, W*0.2, H*0.15, 800);
    g1.addColorStop(0, "rgba(70,130,255,0.18)");
    g1.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g1;
    ctx.fillRect(0,0,W,H);

    const g2 = ctx.createRadialGradient(W*0.85, H*0.20, 30, W*0.85, H*0.20, 700);
    g2.addColorStop(0, "rgba(255,160,70,0.12)");
    g2.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g2;
    ctx.fillRect(0,0,W,H);

    // track perspective
    ctx.save();
    ctx.translate(W/2, H*0.12);

    const topW = W * 0.20;
    const botW = W * 0.58;
    const topY = 0;
    const botY = H * 0.95;

    ctx.beginPath();
    ctx.moveTo(-topW/2, topY);
    ctx.lineTo(topW/2, topY);
    ctx.lineTo(botW/2, botY);
    ctx.lineTo(-botW/2, botY);
    ctx.closePath();
    ctx.fillStyle = "rgba(255,255,255,0.03)";
    ctx.fill();

    // lane lines
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 2;

    for (let i = -1; i <= 1; i++) {
      const xTop = i * (topW/3);
      const xBot = i * (botW/3);
      ctx.beginPath();
      ctx.moveTo(xTop, topY);
      ctx.lineTo(xBot, botY);
      ctx.stroke();
    }

    // motion dashes
    const dashCount = 22;
    for (let k = 0; k < dashCount; k++) {
      const p = (k / dashCount);
      const y = topY + p * botY;
      const width = topW + p * (botW - topW);
      const dashW = 14 + p * 22;
      const dashH = 6 + p * 6;

      const scroll = (state.t * state.speed * 0.06) % 1.0;
      const shift = (scroll * botY);

      const yy = (y + shift) % botY;

      ctx.fillStyle = "rgba(255,255,255,0.06)";
      ctx.fillRect(-dashW/2, yy, dashW, dashH);
    }

    ctx.restore();
  }

  function drawScene() {
    // coins
    for (const c of coins) {
      const x = c.x;
      const y = c.y;
      ctx.beginPath();
      ctx.arc(x, y, c.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 212, 92, 0.95)";
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.20)";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, c.r*0.45, 0, Math.PI*2);
      ctx.strokeStyle = "rgba(0,0,0,0.25)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // obstacles
    for (const o of obstacles) {
      const x = o.x;
      const y = o.y;
      const w = o.w;
      const h = o.h;

      const rx = x - w/2;
      const ry = y - h/2;

      ctx.save();
      ctx.shadowColor = "rgba(0,0,0,0.55)";
      ctx.shadowBlur = 18;

      if (o.kind === "barrier") {
        roundRect(rx, ry, w, h, 12);
        ctx.fillStyle = "rgba(255, 92, 92, 0.92)";
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.18)";
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = "rgba(0,0,0,0.22)";
        ctx.fillRect(rx + 10, ry + 12, w - 20, 10);
      } else {
        roundRect(rx, ry, w, h, 14);
        ctx.fillStyle = "rgba(135, 180, 255, 0.85)";
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.18)";
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = "rgba(0,0,0,0.22)";
        ctx.fillRect(rx + 12, ry + 18, w - 24, h - 36);
      }

      ctx.restore();
    }

    // player
    const px = laneX(player.lane);
    const py = player.y;

    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.55)";
    ctx.shadowBlur = 18;

    // body
    roundRect(px - player.w/2, py - player.h, player.w, player.h, 18);
    ctx.fillStyle = "rgba(92, 255, 176, 0.90)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // visor
    roundRect(px - player.w/2 + 10, py - player.h + 14, player.w - 20, 18, 10);
    ctx.fillStyle = "rgba(0,0,0,0.28)";
    ctx.fill();

    // jet trail
    ctx.beginPath();
    ctx.moveTo(px, py + 10);
    ctx.lineTo(px - 12, py + 40);
    ctx.lineTo(px + 12, py + 40);
    ctx.closePath();
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    ctx.fill();

    ctx.restore();

    // small footer hint
    ctx.save();
    ctx.globalAlpha = 0.7;
    ctx.fillStyle = "rgba(245,245,250,0.86)";
    ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
    ctx.fillText("Lane runner | Dodge red barriers, blue trains | Collect coins", 18, H - 16);
    ctx.restore();
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

  // Controls
  const keyDown = (e) => {
    const k = e.key.toLowerCase();

    if (k === " " || k === "w") {
      e.preventDefault();
      if (!state.running && !state.gameOver) start();
      else if (state.gameOver) { reset(42069); start(); }
      else jump();
      return;
    }

    if (k === "r") {
      e.preventDefault();
      reset(42069);
      return;
    }

    if (!state.running) return;

    if (k === "arrowleft" || k === "a") {
      e.preventDefault();
      player.lane = clamp(player.lane - 1, -1, 1);
    } else if (k === "arrowright" || k === "d") {
      e.preventDefault();
      player.lane = clamp(player.lane + 1, -1, 1);
    }
  };

  function jump() {
    if (!player.onGround) return;
    player.onGround = false;
    player.vy = -player.jumpStrength;
  }

  canvas.addEventListener("keydown", keyDown);
  canvas.addEventListener("click", () => {
    canvas.focus();
    if (!state.running && !state.gameOver) start();
  });

  // Touch support: swipe left/right, tap to jump
  let touchStartX = null;
  let touchStartY = null;
  canvas.addEventListener("touchstart", (e) => {
    const t0 = e.touches[0];
    touchStartX = t0.clientX;
    touchStartY = t0.clientY;
    if (!state.running && !state.gameOver) start();
  }, {passive:true});

  canvas.addEventListener("touchend", (e) => {
    if (touchStartX == null) return;
    const t0 = e.changedTouches[0];
    const dx = t0.clientX - touchStartX;
    const dy = t0.clientY - touchStartY;
    touchStartX = null;
    touchStartY = null;

    const ax = Math.abs(dx);
    const ay = Math.abs(dy);

    if (ax > 35 && ax > ay) {
      if (dx < 0) player.lane = clamp(player.lane - 1, -1, 1);
      else player.lane = clamp(player.lane + 1, -1, 1);
    } else {
      jump();
    }
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

components.html(html, height=620, scrolling=False)
