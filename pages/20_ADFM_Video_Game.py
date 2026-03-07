import math
import random
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="DOOMLINE", page_icon="🔫", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: #0b0b0b;
        color: #e5e5e5;
    }
    .block {
        background: #111111;
        border: 1px solid #262626;
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        color: #f97316;
    }
    .sub {
        color: #bdbdbd;
        line-height: 1.55;
        font-size: 0.98rem;
    }
    .stat {
        background: #101010;
        border: 1px solid #232323;
        border-radius: 14px;
        padding: 12px 14px;
        min-height: 92px;
    }
    .stat-k {
        color: #8a8a8a;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .stat-v {
        color: #fafafa;
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 6px;
    }
    .divider {
        height: 1px;
        background: #262626;
        margin: 10px 0 14px 0;
    }
    .small {
        color: #9a9a9a;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "doom_seed" not in st.session_state:
    st.session_state.doom_seed = random.randint(1000, 999999)

if "doom_runs" not in st.session_state:
    st.session_state.doom_runs = []

st.markdown(
    """
    <div class="block">
        <div class="title">DOOMLINE</div>
        <div class="sub">
            This is a Doom-inspired Streamlit build, meaning a browser raycaster running inside a Streamlit shell with keyboard movement, enemies, shooting, health, ammo, pickups, score tracking, a minimap, atmosphere, and a game loop that actually works in one file. It is not the original Doom and it is not pretending to be a full id Software port, but it does get you to the right place visually and mechanically inside the limits of the platform.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat"><div class="stat-k">Mode</div><div class="stat-v">Raycaster FPS</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat"><div class="stat-k">Engine</div><div class="stat-v">HTML5 Canvas</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat"><div class="stat-k">Shell</div><div class="stat-v">Streamlit</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="stat"><div class="stat-k">Seed</div><div class="stat-v">{st.session_state.doom_seed}</div></div>', unsafe_allow_html=True)

left, right = st.columns([0.72, 0.28])

with right:
    st.markdown(
        """
        <div class="block">
            <div style="font-weight:700; font-size:1.1rem; color:#fafafa;">Controls</div>
            <div class="divider"></div>
            <div class="small">
            W / S move forward and back<br>
            A / D strafe<br>
            ← / → turn<br>
            Space fires<br>
            Shift sprints<br>
            R restarts after death or victory
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="block">
            <div style="font-weight:700; font-size:1.1rem; color:#fafafa;">What this is doing</div>
            <div class="divider"></div>
            <div class="small">
            The app embeds a full-screen canvas game with pseudo-3D raycasting, sprite enemies, collision, pickup logic, muzzle flash, damage model, and a minimap. Streamlit is the shell around it, not the render engine itself.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Generate New Seed", use_container_width=True):
        st.session_state.doom_seed = random.randint(1000, 999999)
        st.rerun()

with left:
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        html, body {{ margin: 0; padding: 0; background: #050505; overflow: hidden; }}
        #wrap {{ width: 100%; }}
        canvas {{ width: 100%; display: block; border-radius: 16px; border: 1px solid #262626; background: #000; }}
      </style>
    </head>
    <body>
      <div id="wrap">
        <canvas id="game" width="1200" height="760"></canvas>
      </div>

      <script>
      (() => {{
        const canvas = document.getElementById('game');
        const ctx = canvas.getContext('2d');

        const rngSeed = {st.session_state.doom_seed};
        function mulberry32(a) {{
          return function() {{
            let t = a += 0x6D2B79F5;
            t = Math.imul(t ^ t >>> 15, t | 1);
            t ^= t + Math.imul(t ^ t >>> 7, t | 61);
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
          }}
        }}
        const rand = mulberry32(rngSeed);

        const W = canvas.width;
        const H = canvas.height;
        const HUD_H = 110;
        const VIEW_H = H - HUD_H;
        const FOV = Math.PI / 3;
        const HALF_FOV = FOV / 2;
        const NUM_RAYS = 360;
        const MAX_DEPTH = 20;
        const TILE = 1;

        const MAP_W = 20;
        const MAP_H = 20;
        const map = [
          [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
          [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,1],
          [1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,1],
          [1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1],
          [1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1],
          [1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1],
          [1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1],
          [1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,1],
          [1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,1,0,1,0,1],
          [1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1],
          [1,0,1,1,1,1,1,0,1,1,1,0,1,0,0,1,1,1,0,1],
          [1,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,1],
          [1,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,0,1,0,1],
          [1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1],
          [1,0,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1],
          [1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,1],
          [1,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,0,1],
          [1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1],
          [1,2,0,0,1,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1],
          [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ];

        const wallColors = {{
          1: '#6b1d0f',
          2: '#7c2d12',
          3: '#a16207'
        }};

        const player = {{
          x: 1.5,
          y: 18.5,
          angle: -Math.PI / 2,
          health: 100,
          ammo: 24,
          score: 0,
          keys: 0,
          speed: 2.6,
          sprint: 1.85,
          fireCooldown: 0,
          hurtFlash: 0,
          muzzle: 0,
          alive: true,
          win: false,
        }};

        const enemies = [];
        const pickups = [];
        const particles = [];

        function spawnEnemies() {{
          const pts = [
            [5.5, 5.5], [9.5, 3.5], [13.5, 5.5], [16.5, 8.5], [4.5, 9.5],
            [12.5, 11.5], [6.5, 13.5], [15.5, 15.5], [9.5, 17.5], [17.5, 2.5]
          ];
          pts.forEach((p, i) => enemies.push({{
            x: p[0], y: p[1], hp: 40 + Math.floor(rand()*35), alive: true,
            cooldown: rand()*1.2, speed: 0.48 + rand()*0.24, radius: 0.22,
            kind: i % 3
          }}));
        }}

        function spawnPickups() {{
          const pts = [
            [3.5, 1.5, 'ammo'], [18.5, 3.5, 'med'], [11.5, 7.5, 'ammo'], [6.5, 11.5, 'med'],
            [14.5, 13.5, 'ammo'], [2.5, 15.5, 'med'], [17.5, 17.5, 'ammo']
          ];
          pts.forEach(p => pickups.push({{ x: p[0], y: p[1], type: p[2], taken: false }}));
        }}

        spawnEnemies();
        spawnPickups();

        const keys = {{}};
        addEventListener('keydown', e => keys[e.key.toLowerCase()] = true);
        addEventListener('keyup', e => keys[e.key.toLowerCase()] = false);

        function cell(x, y) {{
          if (x < 0 || y < 0 || x >= MAP_W || y >= MAP_H) return 1;
          return map[Math.floor(y)][Math.floor(x)];
        }}

        function isWall(x, y) {{
          return cell(x, y) === 1;
        }}

        function canMove(x, y) {{
          return !isWall(x, y);
        }}

        function normalize(a) {{
          while (a < -Math.PI) a += Math.PI * 2;
          while (a > Math.PI) a -= Math.PI * 2;
          return a;
        }}

        function castRay(angle) {{
          let depth = 0;
          const step = 0.02;
          const sin = Math.sin(angle);
          const cos = Math.cos(angle);
          while (depth < MAX_DEPTH) {{
            depth += step;
            const x = player.x + cos * depth;
            const y = player.y + sin * depth;
            const c = cell(x, y);
            if (c > 0) return {{ depth, wall: c, x, y }};
          }}
          return {{ depth: MAX_DEPTH, wall: 1, x: player.x, y: player.y }};
        }}

        function shoot() {{
          if (!player.alive || player.win) return;
          if (player.fireCooldown > 0 || player.ammo <= 0) return;
          player.ammo -= 1;
          player.fireCooldown = 0.28;
          player.muzzle = 0.08;

          let best = null;
          enemies.forEach((e) => {{
            if (!e.alive) return;
            const dx = e.x - player.x;
            const dy = e.y - player.y;
            const dist = Math.hypot(dx, dy);
            const ang = normalize(Math.atan2(dy, dx) - player.angle);
            if (Math.abs(ang) < 0.09) {{
              const ray = castRay(player.angle + ang * 0.3);
              if (dist < ray.depth + 0.2) {{
                if (!best || dist < best.dist) best = {{ enemy: e, dist }};
              }}
            }}
          }});

          if (best) {{
            best.enemy.hp -= 26 + Math.floor(rand()*12);
            particles.push({{x: best.enemy.x, y: best.enemy.y, t: 0.25, color: '#f97316'}});
            if (best.enemy.hp <= 0) {{
              best.enemy.alive = false;
              player.score += 100;
              if (rand() < 0.35) pickups.push({{ x: best.enemy.x, y: best.enemy.y, type: rand() < 0.5 ? 'ammo' : 'med', taken: false }});
            }}
          }}
        }}

        addEventListener('keydown', (e) => {{
          if (e.code === 'Space') {{ e.preventDefault(); shoot(); }}
          if (e.key.toLowerCase() === 'r' && (!player.alive || player.win)) location.reload();
        }});

        function update(dt) {{
          if (!player.alive || player.win) return;

          const turnSpeed = 2.2;
          if (keys['arrowleft']) player.angle -= turnSpeed * dt;
          if (keys['arrowright']) player.angle += turnSpeed * dt;

          let moveX = 0;
          let moveY = 0;
          const moveSpeed = player.speed * (keys['shift'] ? player.sprint : 1);
          const cs = Math.cos(player.angle);
          const sn = Math.sin(player.angle);

          if (keys['w']) {{ moveX += cs * moveSpeed * dt; moveY += sn * moveSpeed * dt; }}
          if (keys['s']) {{ moveX -= cs * moveSpeed * dt; moveY -= sn * moveSpeed * dt; }}
          if (keys['a']) {{ moveX += Math.cos(player.angle - Math.PI/2) * moveSpeed * dt; moveY += Math.sin(player.angle - Math.PI/2) * moveSpeed * dt; }}
          if (keys['d']) {{ moveX += Math.cos(player.angle + Math.PI/2) * moveSpeed * dt; moveY += Math.sin(player.angle + Math.PI/2) * moveSpeed * dt; }}

          const nx = player.x + moveX;
          const ny = player.y + moveY;
          if (canMove(nx, player.y)) player.x = nx;
          if (canMove(player.x, ny)) player.y = ny;

          if (player.fireCooldown > 0) player.fireCooldown -= dt;
          if (player.hurtFlash > 0) player.hurtFlash -= dt;
          if (player.muzzle > 0) player.muzzle -= dt;

          enemies.forEach(e => {{
            if (!e.alive) return;
            const dx = player.x - e.x;
            const dy = player.y - e.y;
            const dist = Math.hypot(dx, dy);
            if (dist > 0.8) {{
              const vx = dx / dist * e.speed * dt;
              const vy = dy / dist * e.speed * dt;
              const ex = e.x + vx;
              const ey = e.y + vy;
              if (canMove(ex, e.y)) e.x = ex;
              if (canMove(e.x, ey)) e.y = ey;
            }}
            e.cooldown -= dt;
            if (dist < 1.15 && e.cooldown <= 0) {{
              player.health -= 7 + Math.floor(rand()*6);
              player.hurtFlash = 0.16;
              e.cooldown = 0.9 + rand()*0.65;
              if (player.health <= 0) {{
                player.health = 0;
                player.alive = false;
              }}
            }}
          }});

          pickups.forEach(p => {{
            if (p.taken) return;
            if (Math.hypot(player.x - p.x, player.y - p.y) < 0.45) {{
              p.taken = true;
              if (p.type === 'ammo') player.ammo += 8;
              if (p.type === 'med') player.health = Math.min(100, player.health + 25);
              player.score += 15;
            }}
          }});

          particles.forEach(p => p.t -= dt);
          while (particles.length && particles[0].t <= 0) particles.shift();

          const aliveCount = enemies.filter(e => e.alive).length;
          if (aliveCount === 0 && Math.hypot(player.x - 18.5, player.y - 1.5) < 0.7) {{
            player.win = true;
            player.score += 500;
          }}
        }}

        function drawBackground() {{
          const sky = ctx.createLinearGradient(0, 0, 0, VIEW_H * 0.55);
          sky.addColorStop(0, '#2a0f0b');
          sky.addColorStop(1, '#5b1d16');
          ctx.fillStyle = sky;
          ctx.fillRect(0, 0, W, VIEW_H * 0.55);

          const floor = ctx.createLinearGradient(0, VIEW_H * 0.55, 0, VIEW_H);
          floor.addColorStop(0, '#242424');
          floor.addColorStop(1, '#080808');
          ctx.fillStyle = floor;
          ctx.fillRect(0, VIEW_H * 0.55, W, VIEW_H * 0.45);
        }}

        function drawWorld() {{
          drawBackground();
          const rayStep = FOV / NUM_RAYS;
          const colW = W / NUM_RAYS;
          const zBuffer = [];

          for (let i = 0; i < NUM_RAYS; i++) {{
            const rayAngle = player.angle - HALF_FOV + i * rayStep;
            const hit = castRay(rayAngle);
            const corrected = hit.depth * Math.cos(rayAngle - player.angle);
            zBuffer[i] = corrected;
            const wallH = Math.min(VIEW_H, (VIEW_H / Math.max(0.0001, corrected)) * 0.78);
            const y = (VIEW_H - wallH) / 2;
            const shade = Math.max(0.18, 1 - corrected / 11);
            const base = wallColors[hit.wall] || '#7c2d12';
            ctx.fillStyle = shadeColor(base, shade);
            ctx.fillRect(i * colW, y, colW + 1, wallH);
          }}

          const sprites = [];
          enemies.forEach(e => {{
            if (!e.alive) return;
            sprites.push({{x:e.x,y:e.y,type:'enemy',kind:e.kind,dist:Math.hypot(e.x-player.x,e.y-player.y),hp:e.hp}});
          }});
          pickups.forEach(p => {{
            if (p.taken) return;
            sprites.push({{x:p.x,y:p.y,type:p.type,dist:Math.hypot(p.x-player.x,p.y-player.y)}});
          }});
          particles.forEach(p => sprites.push({{x:p.x,y:p.y,type:'fx',dist:Math.hypot(p.x-player.x,p.y-player.y)}}));
          sprites.sort((a,b) => b.dist - a.dist);

          sprites.forEach(s => {{
            const dx = s.x - player.x;
            const dy = s.y - player.y;
            const angle = normalize(Math.atan2(dy, dx) - player.angle);
            if (Math.abs(angle) > HALF_FOV + 0.3) return;
            const dist = Math.hypot(dx, dy) * Math.cos(angle);
            if (dist <= 0.2) return;
            const size = Math.min(260, VIEW_H / dist * (s.type === 'enemy' ? 0.72 : 0.42));
            const sx = (0.5 + angle / FOV) * W;
            const sy = VIEW_H / 2 + 36 / dist - size / 2;
            const screenRay = Math.floor(sx / colW);
            if (screenRay >= 0 && screenRay < zBuffer.length && dist > zBuffer[screenRay] + 0.08) return;

            if (s.type === 'enemy') drawEnemy(sx, sy, size, s.kind, s.hp);
            if (s.type === 'ammo') drawPickup(sx, sy, size, '#60a5fa');
            if (s.type === 'med') drawPickup(sx, sy, size, '#22c55e');
            if (s.type === 'fx') drawPickup(sx, sy, size*0.6, '#f97316');
          }});
        }}

        function shadeColor(hex, k) {{
          const n = hex.replace('#','');
          const r = parseInt(n.substring(0,2),16);
          const g = parseInt(n.substring(2,4),16);
          const b = parseInt(n.substring(4,6),16);
          const rr = Math.max(0, Math.min(255, Math.floor(r*k)));
          const gg = Math.max(0, Math.min(255, Math.floor(g*k)));
          const bb = Math.max(0, Math.min(255, Math.floor(b*k)));
          return `rgb(${{rr}},${{gg}},${{bb}})`;
        }}

        function drawEnemy(x, y, size, kind, hp) {{
          const colors = ['#ef4444','#f97316','#dc2626'];
          const c = colors[kind % colors.length];
          ctx.fillStyle = c;
          ctx.beginPath();
          ctx.arc(x, y + size*0.18, size*0.2, 0, Math.PI*2);
          ctx.fill();
          ctx.fillRect(x - size*0.14, y + size*0.34, size*0.28, size*0.34);
          ctx.fillRect(x - size*0.22, y + size*0.36, size*0.08, size*0.25);
          ctx.fillRect(x + size*0.14, y + size*0.36, size*0.08, size*0.25);
          ctx.fillRect(x - size*0.12, y + size*0.68, size*0.08, size*0.22);
          ctx.fillRect(x + size*0.04, y + size*0.68, size*0.08, size*0.22);
          ctx.fillStyle = '#111827';
          ctx.fillRect(x - size*0.16, y + size*0.07, size*0.32, size*0.07);
          ctx.fillStyle = '#22d3ee';
          ctx.fillRect(x - size*0.15, y + size*0.08, size*0.11, size*0.05);
          ctx.fillRect(x + size*0.04, y + size*0.08, size*0.11, size*0.05);
          ctx.fillStyle = '#111';
          ctx.fillRect(x - size*0.18, y - size*0.02, size*0.36, size*0.05);
          ctx.fillStyle = '#ef4444';
          ctx.fillRect(x - size*0.18, y - size*0.02, size * Math.max(0, Math.min(1, hp / 75)), size*0.05);
        }}

        function drawPickup(x, y, size, color) {{
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y + size*0.45, size*0.14, 0, Math.PI*2);
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }}

        function drawWeapon() {{
          const bob = Math.sin(performance.now() / 120) * 6;
          const muzzle = player.muzzle > 0 ? 1 : 0;
          ctx.fillStyle = '#1f2937';
          ctx.fillRect(W*0.39, VIEW_H - 26 + bob, W*0.22, 90);
          ctx.fillStyle = '#6b7280';
          ctx.fillRect(W*0.47, VIEW_H - 76 + bob, W*0.06, 72);
          ctx.fillStyle = '#9ca3af';
          ctx.fillRect(W*0.485, VIEW_H - 130 + bob, W*0.03, 56);
          if (muzzle) {{
            ctx.fillStyle = 'rgba(251,191,36,0.85)';
            ctx.beginPath();
            ctx.moveTo(W*0.5, VIEW_H - 142 + bob);
            ctx.lineTo(W*0.465, VIEW_H - 178 + bob);
            ctx.lineTo(W*0.535, VIEW_H - 178 + bob);
            ctx.closePath();
            ctx.fill();
          }}
        }}

        function drawCrosshair() {{
          ctx.strokeStyle = 'rgba(255,255,255,0.85)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(W/2 - 10, VIEW_H/2); ctx.lineTo(W/2 + 10, VIEW_H/2);
          ctx.moveTo(W/2, VIEW_H/2 - 10); ctx.lineTo(W/2, VIEW_H/2 + 10);
          ctx.stroke();
        }}

        function drawMinimap() {{
          const scale = 10;
          const ox = 18;
          const oy = 18;
          ctx.fillStyle = 'rgba(0,0,0,0.42)';
          ctx.fillRect(ox - 8, oy - 8, MAP_W*scale + 16, MAP_H*scale + 16);
          for (let y = 0; y < MAP_H; y++) {{
            for (let x = 0; x < MAP_W; x++) {{
              const c = map[y][x];
              ctx.fillStyle = c === 1 ? '#5b1d16' : c === 2 ? '#166534' : c === 3 ? '#a16207' : '#111111';
              ctx.fillRect(ox + x*scale, oy + y*scale, scale-1, scale-1);
            }}
          }}
          pickups.forEach(p => {{
            if (p.taken) return;
            ctx.fillStyle = p.type === 'ammo' ? '#60a5fa' : '#22c55e';
            ctx.fillRect(ox + p.x*scale - 2, oy + p.y*scale - 2, 4, 4);
          }});
          enemies.forEach(e => {{
            if (!e.alive) return;
            ctx.fillStyle = '#ef4444';
            ctx.fillRect(ox + e.x*scale - 2, oy + e.y*scale - 2, 4, 4);
          }});
          ctx.fillStyle = '#fafafa';
          ctx.beginPath();
          ctx.arc(ox + player.x*scale, oy + player.y*scale, 3, 0, Math.PI*2);
          ctx.fill();
          ctx.strokeStyle = '#fafafa';
          ctx.beginPath();
          ctx.moveTo(ox + player.x*scale, oy + player.y*scale);
          ctx.lineTo(ox + player.x*scale + Math.cos(player.angle)*8, oy + player.y*scale + Math.sin(player.angle)*8);
          ctx.stroke();
        }}

        function drawHud() {{
          ctx.fillStyle = '#080808';
          ctx.fillRect(0, VIEW_H, W, HUD_H);
          ctx.strokeStyle = '#262626';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(0, VIEW_H); ctx.lineTo(W, VIEW_H);
          ctx.stroke();

          ctx.fillStyle = '#f97316';
          ctx.font = '800 34px Arial';
          ctx.fillText('DOOMLINE', 22, VIEW_H + 42);

          ctx.fillStyle = '#fafafa';
          ctx.font = '700 28px Arial';
          ctx.fillText('HEALTH ' + player.health, 22, VIEW_H + 84);
          ctx.fillText('AMMO ' + player.ammo, 260, VIEW_H + 84);
          ctx.fillText('SCORE ' + player.score, 465, VIEW_H + 84);
          ctx.fillText('DEMONS ' + enemies.filter(e => e.alive).length, 700, VIEW_H + 84);

          const barX = 970;
          const barW = 180;
          ctx.fillStyle = '#1f2937';
          ctx.fillRect(barX, VIEW_H + 58, barW, 18);
          ctx.fillStyle = player.health > 30 ? '#22c55e' : '#ef4444';
          ctx.fillRect(barX, VIEW_H + 58, barW * (player.health / 100), 18);
          ctx.strokeStyle = '#404040';
          ctx.strokeRect(barX, VIEW_H + 58, barW, 18);
        }}

        function drawOverlays() {{
          if (player.hurtFlash > 0) {{
            ctx.fillStyle = `rgba(239,68,68,${{player.hurtFlash * 1.8}})`;
            ctx.fillRect(0, 0, W, VIEW_H);
          }}

          if (!player.alive) {{
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.fillRect(0, 0, W, H);
            ctx.fillStyle = '#ef4444';
            ctx.font = '900 74px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('YOU DIED', W/2, H/2 - 10);
            ctx.fillStyle = '#fafafa';
            ctx.font = '700 28px Arial';
            ctx.fillText('Press R to restart', W/2, H/2 + 42);
            ctx.textAlign = 'left';
          }}

          if (player.win) {{
            ctx.fillStyle = 'rgba(0,0,0,0.55)';
            ctx.fillRect(0, 0, W, H);
            ctx.fillStyle = '#f59e0b';
            ctx.font = '900 72px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('LEVEL CLEARED', W/2, H/2 - 18);
            ctx.fillStyle = '#fafafa';
            ctx.font = '700 28px Arial';
            ctx.fillText('Press R to run it again', W/2, H/2 + 34);
            ctx.textAlign = 'left';
          }}
        }}

        let last = performance.now();
        function loop(ts) {{
          const dt = Math.min(0.033, (ts - last) / 1000);
          last = ts;
          update(dt);
          drawWorld();
          drawCrosshair();
          drawWeapon();
          drawMinimap();
          drawHud();
          drawOverlays();
          requestAnimationFrame(loop);
        }}

        loop(last);
      }})();
      </script>
    </body>
    </html>
    """
    components.html(html, height=780)

st.markdown(
    """
    <div class="block">
        <div style="font-weight:700; font-size:1.08rem; color:#fafafa;">Reality check</div>
        <div class="divider"></div>
        <div class="sub">
            A true Doom port inside raw Streamlit would mean shipping a much larger asset pipeline, more formal input handling, audio layers, texture packs, proper sprite sheets, level serialization, and probably a custom frontend app rather than a pure Streamlit page. But if the ask is to get as close as possible inside Streamlit itself, this is the right path: embed a canvas shooter and let Streamlit handle the surrounding shell.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
