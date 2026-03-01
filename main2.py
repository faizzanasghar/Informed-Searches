"""
Dynamic Pathfinding Agent — Phase 2  ⬡ CYBER EDITION
══════════════════════════════════════════════════════
Premium GUI featuring:
  • Cyber/neon dark theme with glowing grid cells
  • Neon-bordered buttons with glow effect
  • Glassmorphism metric cards
  • Animated agent pulse ring
  • Gradient panel background
  • Scanline grid overlay
#   • Color-coded cell types with brightnss
  • Scrollable side panel
  • Full Phase-2 logic: A*, GBFS, dynamic obstacles, re-planning
"""

import pygame, sys, math, heapq, random, time
from tkinter import Tk, font, simpledialog

# ══════════════════════════════════════════════════════════════
#  CYBER PALETTE
# ══════════════════════════════════════════════════════════════
#  Grid cells
C_BG        = (8,    8,   18)   # deep space background
C_EMPTY     = (14,  14,   28)   # empty cell
C_GRID_LINE = (25,  25,   45)   # subtle grid lines
C_WALL      = (38,  38,   58)   # dark slate wall
C_WALL_HL   = (55,  55,   80)   # wall highlight edge

C_START     = (0,  255,  160)   # neon mint
C_GOAL      = (255, 60,  120)   # neon pink
C_FRONTIER  = (255, 220,  0)    # neon yellow
C_VISITED   = (30,  80,  200)   # deep blue
C_VISITED2  = (60, 120,  255)   # bright blue (inner glow)
C_PATH      = (0,  230,  255)   # neon cyan
C_AGENT     = (255, 140,  0)    # neon orange
C_TRAIL     = (0,   80,   60)   # dark teal trail

#  Panel
P_BG1       = (10,  10,  22)    # panel gradient top
P_BG2       = (16,  12,  35)    # panel gradient bottom
P_CARD      = (20,  20,  40)    # metric card bg
P_CARD2     = (28,  18,  52)    # card variant
P_BORDER    = (60,  60,  90)    # subtle border
P_GLOW_C    = (0,  200, 255)    # cyan glow
P_GLOW_M    = (180,  0, 255)    # magenta glow
P_GLOW_G    = (0,  255, 130)    # green glow
P_GLOW_O    = (255, 140,  0)    # orange glow

WHITE   = (255, 255, 255)
BLACK   = (0,    0,   0)
DIM     = (100, 100, 130)
BRIGHT  = (200, 210, 255)

#  Flash
C_REPLAN = (255,  50,  80)
C_ARRIVE = (0,   255, 160)

PANEL_W  = 300
FPS      = 60
MIN_CELL = 10
MAX_CELL = 80

# ══════════════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ══════════════════════════════════════════════════════════════
def drawGlowRect(surf, color, rect, radius=6, glow_r=8, alpha=80):
    """Draw a filled rect with a soft neon glow border."""
    r = pygame.Rect(rect)
    # Glow layers (expanding rects with decreasing alpha)
    glow_surf = pygame.Surface((r.w + glow_r*2, r.h + glow_r*2), pygame.SRCALPHA)
    for i in range(glow_r, 0, -1):
        a = int(alpha * (i / glow_r) ** 1.5)
        gr = pygame.Rect(glow_r-i, glow_r-i, r.w+i*2, r.h+i*2)
        pygame.draw.rect(glow_surf, (*color, a), gr, border_radius=radius+i, width=2)
    surf.blit(glow_surf, (r.x - glow_r, r.y - glow_r))
    pygame.draw.rect(surf, color, r, border_radius=radius, width=1)

def drawNeonLine(surf, color, p1, p2, width=1, alpha=180):
    """Draw a glowing line."""
    pygame.draw.line(surf, color, p1, p2, width)

def fillGradientVertical(surf, rect, top_color, bot_color):
    """Fill a rect with a vertical gradient."""
    x, y, w, h = rect
    for i in range(h):
        t = i / max(h-1, 1)
        r = int(top_color[0] + (bot_color[0]-top_color[0])*t)
        g = int(top_color[1] + (bot_color[1]-top_color[1])*t)
        b = int(top_color[2] + (bot_color[2]-top_color[2])*t)
        pygame.draw.line(surf, (r,g,b), (x, y+i), (x+w, y+i))

def lerpColor(a, b, t):
    return tuple(int(a[i] + (b[i]-a[i])*t) for i in range(3))

def DrawPill(surf, color, rect, radius=None):
    r = pygame.Rect(rect)
    rad = radius if radius else r.h//2
    pygame.draw.rect(surf, color, r, border_radius=rad)

# ══════════════════════════════════════════════════════════════
#  HEURISTICS & SEARCH
# ══════════════════════════════════════════════════════════════
def manhattan(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
def euclidean(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
HEURISTICS = {"Manhattan": manhattan, "Euclidean": euclidean}

DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
def stepCost(dr,dc): return math.sqrt(2) if dr and dc else 1.0

def neighbors(node, rows, cols, walls):
    r,c = node
    return [((r+dr,c+dc), stepCost(dr,dc))
            for dr,dc in DIRS
            if 0<=r+dr<rows and 0<=c+dc<cols and (r+dr,c+dc) not in walls]

def searchInstantly(start, goal, rows, cols, walls, algo, heuristic):
    h = HEURISTICS[heuristic]; t0 = time.perf_counter()
    if algo == "A*":
        open_set = [(h(start,goal), 0.0, start)]
        came_from = {start: None}; g = {start: 0.0}; vis = set(); fr = {start}
        while open_set:
            _, gc, cur = heapq.heappop(open_set); fr.discard(cur)
            if cur in vis: continue
            vis.add(cur)
            if cur == goal:
                return dict(path=Recon(came_from,goal), visited=vis, frontier=fr,
                            nodes=len(vis), cost=g[goal],
                            time_ms=(time.perf_counter()-t0)*1000, found=True)
            for nb,cost in neighbors(cur,rows,cols,walls):
                tg = g[cur]+cost
                if nb not in g or tg < g[nb]:
                    g[nb]=tg; heapq.heappush(open_set,(tg+h(nb,goal),tg,nb))
                    came_from[nb]=cur; fr.add(nb)
    else:
        open_set = [(h(start,goal), start)]
        came_from = {start: None}; vis = set(); fr = {start}
        while open_set:
            _, cur = heapq.heappop(open_set); fr.discard(cur)
            if cur in vis: continue
            vis.add(cur)
            if cur == goal:
                p = Recon(came_from,goal)
                cost = sum(stepCost(p[i][0]-p[i-1][0],p[i][1]-p[i-1][1]) for i in range(1,len(p)))
                return dict(path=p, visited=vis, frontier=fr, nodes=len(vis),
                            cost=cost, time_ms=(time.perf_counter()-t0)*1000, found=True)
            for nb,_ in neighbors(cur,rows,cols,walls):
                if nb not in vis and nb not in came_from:
                    came_from[nb]=cur; heapq.heappush(open_set,(h(nb,goal),nb)); fr.add(nb)
    return dict(path=[], visited=set(), frontier=set(), nodes=0,
                cost=0, time_ms=(time.perf_counter()-t0)*1000, found=False)

def SearchAnimated(start, goal, rows, cols, walls, algo, heuristic):
    h = HEURISTICS[heuristic]; t0 = time.perf_counter()
    if algo == "A*":
        open_set = [(h(start,goal), 0.0, start)]
        came_from = {start: None}; g = {start: 0.0}; vis = set(); fr = {start}
        while open_set:
            _, gc, cur = heapq.heappop(open_set); fr.discard(cur)
            if cur in vis: continue
            vis.add(cur)
            if cur == goal:
                yield dict(done=True, path=Recon(came_from,goal),
                           visited=set(vis), frontier=set(fr), nodes=len(vis),
                           cost=g[goal], time_ms=(time.perf_counter()-t0)*1000); return
            for nb,cost in neighbors(cur,rows,cols,walls):
                tg = g[cur]+cost
                if nb not in g or tg < g[nb]:
                    g[nb]=tg; heapq.heappush(open_set,(tg+h(nb,goal),tg,nb))
                    came_from[nb]=cur; fr.add(nb)
            yield dict(done=False, visited=set(vis), frontier=set(fr),
                       nodes=len(vis), cost=0, time_ms=0)
    else:
        open_set = [(h(start,goal), start)]
        came_from = {start: None}; vis = set(); fr = {start}
        while open_set:
            _, cur = heapq.heappop(open_set); fr.discard(cur)
            if cur in vis: continue
            vis.add(cur)
            if cur == goal:
                p = Recon(came_from,goal)
                cost = sum(stepCost(p[i][0]-p[i-1][0],p[i][1]-p[i-1][1]) for i in range(1,len(p)))
                yield dict(done=True, path=p, visited=set(vis), frontier=set(fr),
                           nodes=len(vis), cost=cost,
                           time_ms=(time.perf_counter()-t0)*1000); return
            for nb,_ in neighbors(cur,rows,cols,walls):
                if nb not in vis and nb not in came_from:
                    came_from[nb]=cur; heapq.heappush(open_set,(h(nb,goal),nb)); fr.add(nb)
            yield dict(done=False, visited=set(vis), frontier=set(fr),
                       nodes=len(vis), cost=0, time_ms=0)
    yield dict(done=True, path=[], visited=set(), frontier=set(),
               nodes=0, cost=0, time_ms=(time.perf_counter()-t0)*1000)

def Recon(came_from, goal):
    path, n = [], goal
    while n is not None: path.append(n); n = came_from[n]
    return path[::-1]

def generateMaze(rows, cols, density, start, goal):
    return {(r,c) for r in range(rows) for c in range(cols)
            if (r,c) not in (start,goal) and random.random() < density}


# ══════════════════════════════════════════════════════════════
#  NEON BUTTON
# ══════════════════════════════════════════════════════════════
class NeonButton:
    _fonts = {}
    def __init__(self, rect, text, glow_color=P_GLOW_C,
                 bg=(18,18,38), fs=13, toggle=False, icon=""):
        self.rect       = pygame.Rect(rect)
        self.text       = text
        self.icon       = icon
        self.glow_color = glow_color
        self.bg         = bg
        self.fs         = fs
        self.toggle     = toggle
        self.active     = False

    def Font(self):
        key = (self.fs, True)
        if key not in NeonButton._fonts:
            NeonButton._fonts[key] = pygame.font.SysFont("Consolas", self.fs, bold=True)
        return NeonButton._fonts[key]

    def draw_on(self, surf, mouse_local):
        hov    = self.rect.collidepoint(mouse_local)
        active = self.toggle and self.active

        # Background fill
        bg = self.bg
        if active:
            bg = tuple(min(255, int(c*0.4 + g*0.6))
                       for c,g in zip(self.bg, self.glow_color))
        elif hov:
            bg = tuple(min(255, c+20) for c in self.bg)
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)

        # Neon border — full glow when active/hovered
        border_col = self.glow_color if (active or hov) else tuple(c//3 for c in self.glow_color)
        border_w   = 2 if (active or hov) else 1
        pygame.draw.rect(surf, border_col, self.rect, border_w, border_radius=6)

        # Glow effect when active or hovered
        if active or hov:
            for i in range(1, 4):
                a  = 60 - i*18
                gr = self.rect.inflate(i*2, i*2)
                gs = pygame.Surface((gr.w, gr.h), pygame.SRCALPHA)
                pygame.draw.rect(gs, (*self.glow_color, a), (0,0,gr.w,gr.h),
                                 border_radius=6+i, width=2)
                surf.blit(gs, (gr.x, gr.y))

        # Top highlight line
        hl_col = tuple(min(255, c+80) for c in border_col)
        pygame.draw.line(surf, hl_col,
                         (self.rect.x+8, self.rect.y+1),
                         (self.rect.right-8, self.rect.y+1))

        # Label
        label = (self.icon + " " + self.text).strip() if self.icon else self.text
        col   = WHITE if (active or hov) else BRIGHT
        lbl   = self.Font().render(label, True, col)
        surf.blit(lbl, lbl.get_rect(center=self.rect.center))

        # Label logic (Ensure it doesn't overflow)
        label = (self.icon + " " + self.text).strip() if self.icon else self.text
        col   = WHITE if (active or hov) else BRIGHT
    
    # Dynamic font sizing based on button width
        fs = self.fs
        if self.rect.width < 100: fs = 11 # Smaller font for triple buttons
    
        font = pygame.font.SysFont("Consolas", fs, bold=True)
        lbl  = font.render(label, True, col)
    
    # Clip text if it's still too long (Safety)
        text_rect = lbl.get_rect(center=self.rect.center)
        if text_rect.width > self.rect.width - 10:
            # If text is too wide, truncate and add ellipsis
            max_width = self.rect.width - 10
            while lbl.get_size()[0] > max_width and label:
                label = label[:-1]
                lbl = font.render(label + "...", True, col)
        surf.blit(lbl, text_rect)

    def clicked(self, lev):
        return (lev.type == pygame.MOUSEBUTTONDOWN
                and lev.button == 1
                and self.rect.collidepoint(lev.pos))


# ══════════════════════════════════════════════════════════════
#  NEON SLIDER
# ══════════════════════════════════════════════════════════════
class NeonSlider:
    _font = None
    def __init__(self, x, y, w, lo, hi, val, label, fmt="{:.2f}", color=P_GLOW_C):
        self.top   = y
        self.rect  = pygame.Rect(x, y+18, w, 6)
        self.lo=lo; self.hi=hi; self.value=val
        self.label=label; self.fmt=fmt
        self.color=color; self.dragging=False

    @classmethod
    def font(cls):
        if cls._font is None:
            cls._font = pygame.font.SysFont("Consolas", 11)
        return cls._font

    @property
    def total_height(self): return 34

    def draw(self, surf):
        frac = (self.value - self.lo) / max(self.hi - self.lo, 1e-9)
        # Label
        txt = f"{self.label}  {self.fmt.format(self.value)}"
        surf.blit(self.font().render(txt, True, DIM), (self.rect.x, self.top))

        # Track background
        pygame.draw.rect(surf, (30,30,50), self.rect, border_radius=3)

        # Filled portion
        fw = max(0, int(self.rect.w * frac))
        if fw:
            fill_r = pygame.Rect(self.rect.x, self.rect.y, fw, self.rect.h)
            pygame.draw.rect(surf, self.color, fill_r, border_radius=3)
            # Glow on fill
            gs = pygame.Surface((fw, self.rect.h+6), pygame.SRCALPHA)
            pygame.draw.rect(gs, (*self.color, 40), (0,0,fw,self.rect.h+6), border_radius=3)
            surf.blit(gs, (self.rect.x, self.rect.y-3))

        # Knob
        kx = self.rect.x + int(self.rect.w * frac)
        ky = self.rect.centery
        pygame.draw.circle(surf, (30,30,50), (kx, ky), 9)
        pygame.draw.circle(surf, self.color,  (kx, ky), 9, 2)
        pygame.draw.circle(surf, WHITE,        (kx, ky), 3)

    def handle(self, ev):
        hit = pygame.Rect(self.rect.x-10, self.top,
                          self.rect.w+20, self.total_height+8)
        if ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1 and hit.collidepoint(ev.pos):
            self.dragging=True
        if ev.type==pygame.MOUSEBUTTONUP:   self.dragging=False
        if ev.type==pygame.MOUSEMOTION and self.dragging:
            f = (ev.pos[0]-self.rect.x)/self.rect.w
            self.value = self.lo + max(0.0,min(1.0,f))*(self.hi-self.lo)

    @property
    def int_val(self): return int(round(self.value))


# ══════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════
class PathfindingApp:
    ST_IDLE      = "IDLE"
    ST_SEARCHING = "SEARCHING"
    ST_MOVING    = "MOVING"
    ST_REPLAN    = "RE-PLANNING"
    ST_ARRIVED   = "ARRIVED"
    ST_NO_PATH   = "NO PATH"

    ST_META = {
        # state: (glow_color, label, icon)
        "IDLE":        (DIM,       "◉  STANDBY",     ""),
        "SEARCHING":   (C_FRONTIER,"◈  SEARCHING",   ""),
        "MOVING":      (C_START,   "▶  MOVING",      ""),
        "RE-PLANNING": (C_REPLAN,  "⚠  RE-PLANNING", ""),
        "ARRIVED":     (C_ARRIVE,  "✔  ARRIVED",     ""),
        "NO PATH":     (C_GOAL,    "✘  NO PATH",     ""),
    }

    # ── Panel layout Y constants (panel-local, scrollable) ───
    Y_TITLE    =   10
    Y_ALG      =   65   # Increased from 52
    Y_ALG_BTN  =   92   # Increased gap
    Y_HEU      =  135   # Increased from 108
    Y_HEU_BTN  =  162
    Y_EDIT     =  205   # Increased from 164
    Y_EDIT_BTN =  232
    Y_ACT      =  280   # Increased from 218
    Y_RUN      =  305
    Y_ANI      =  345
    Y_STEP     =  385
    Y_CLEAR    =  425
    Y_MAZE     =  465
    Y_RESET    =  505
    Y_CFG      =  545
    Y_DYN      =  595
    Y_DYN_BTN  =  622
    Y_SLD      =  670
    Y_SL1      =  700
    Y_SL2      =  750
    Y_SL3      =  800
    Y_MET      =  860
    Y_LEG      = 1080
    CONTENT_H  = 1180  # Increased to accommodate new spacing

    def __init__(self, rows=20, cols=20, density=0.28):
        pygame.init()
        self.rows=rows; self.cols=cols; self.density=density
        self.cell = max(MIN_CELL, min(MAX_CELL, 650//max(rows,cols)))
        self.grid_w = cols*self.cell
        self.grid_h = rows*self.cell

        win_h = max(self.grid_h, 500)
        self.screen = pygame.display.set_mode((self.grid_w+PANEL_W, win_h))
        pygame.display.set_caption("⬡ Dynamic Pathfinding Agent — Cyber Edition")
        self.clock = pygame.time.Clock()

        self.start=(0,0); self.goal=(rows-1,cols-1)
        self.walls = generateMaze(rows,cols,density,self.start,self.goal)

        self.visited=set(); self.frontier=set()
        self.path=[]; self.search_gen=None

        self.agent_pos=self.start; self.trail=set()
        self.path_index=0; self.agent_timer=0
        self.tick = 0   # global frame counter for animations

        self.algorithm="A*"; self.heuristic="Manhattan"
        self.dynamic_mode=False; self.edit_mode="wall"

        self.m_nodes=0; self.m_cost=0.0; self.m_time=0.0
        self.m_replans=0; self.m_total=0.0; self.m_spawned=0

        self.state=self.ST_IDLE
        self.flash_frames=0; self.flash_color=None
        self.dragging=False

        # Scroll
        self.scroll_offset=0
        self.STATUS_BAR_H=36

        self.Build_UI()

    # ────────────────────────────────────────────
    #  UI BUILD
    # ────────────────────────────────────────────
    def Build_UI(self):
        OX = 10; W = PANEL_W - 20
        self.buttons = {}

        def half(key, y, text, left=True, gc=P_GLOW_C, **kw):
            hw = W//2 - 4
            x  = OX if left else OX + W//2 + 4
            self.buttons[key] = NeonButton((x,y,hw,30), text, glow_color=gc, **kw)

        def third(key, y, text, idx=0, gc=P_GLOW_C, **kw):
            tw = W//3 - 3
            x  = OX + idx*(tw+4)
            self.buttons[key] = NeonButton((x,y,tw,28), text, glow_color=gc, **kw)

        def full(key, y, text, gc=P_GLOW_C, h=30, **kw):
            self.buttons[key] = NeonButton((OX,y,W,h), text, glow_color=gc, **kw)

        # Algorithm
        half("alg_astar", self.Y_ALG_BTN, "A*   SEARCH", left=True,  gc=P_GLOW_C, toggle=True)
        half("alg_gbfs",  self.Y_ALG_BTN, "GREEDY BFS",  left=False, gc=P_GLOW_M, toggle=True)
        self.buttons["alg_astar"].active = True

        # Heuristic
        half("h_man", self.Y_HEU_BTN, "MANHATTAN", left=True,  gc=P_GLOW_G, toggle=True)
        half("h_euc", self.Y_HEU_BTN, "EUCLIDEAN", left=False, gc=P_GLOW_G, toggle=True)
        self.buttons["h_man"].active = True

        # Edit mode
        third("mode_wall",  self.Y_EDIT_BTN, "WALLS",  idx=0, gc=(180,180,255), toggle=True)
        third("mode_start", self.Y_EDIT_BTN, "START",  idx=1, gc=C_START,       toggle=True)
        third("mode_goal",  self.Y_EDIT_BTN, "GOAL",   idx=2, gc=C_GOAL,        toggle=True)
        self.buttons["mode_wall"].active = True

        # Actions
        full("run",    self.Y_RUN,   "RUN SEARCH",      gc=P_GLOW_G)
        full("animate",self.Y_ANI,   "ANIMATE AGENT",   gc=P_GLOW_C)
        full("step_s", self.Y_STEP,  "STEP SEARCH",     gc=P_GLOW_M)
        full("clear",  self.Y_CLEAR, "CLEAR",           gc=(255,160,0))
        full("maze",   self.Y_MAZE,  "NEW MAZE",        gc=P_GLOW_M)
        full("reset",  self.Y_RESET, "RESET ALL",       gc=C_GOAL)
        full("config", self.Y_CFG,   "RECONFIGURE",     gc=(160,160,255))

        # Dynamic mode
        full("dynmode", self.Y_DYN_BTN, "DYNAMIC MODE : OFF",
             gc=P_GLOW_M, toggle=True, h=32)

        # Sliders
        sw = W
        self.sl_spawn  = NeonSlider(OX, self.Y_SL1, sw, 0.0, 0.15, 0.03,
                                    "SPAWN PROB   ", "{:.3f}", color=C_GOAL)
        self.sl_aspeed = NeonSlider(OX, self.Y_SL2, sw, 1,   30,   6,
                                    "AGENT SPEED  ", "{:.0f}", color=P_GLOW_C)
        self.sl_sspeed = NeonSlider(OX, self.Y_SL3, sw, 1,   50,   8,
                                    "SEARCH SPEED ", "{:.0f}", color=P_GLOW_G)

        self.UpdAgentInterval()

    def UpdAgentInterval(self):
        self.agent_interval = max(1, FPS // max(1, self.sl_aspeed.int_val))

    # ────────────────────────────────────────────
    #  HELPERS
    # ────────────────────────────────────────────
    def cell_at(self, mx, my):
        c,r = mx//self.cell, my//self.cell
        if 0<=r<self.rows and 0<=c<self.cols: return (r,c)
        return None

    def cellRect(self, r, c):
        return pygame.Rect(c*self.cell, r*self.cell, self.cell-1, self.cell-1)

    def MaxScroll(self):
        vis_h = self.screen.get_height() - self.STATUS_BAR_H
        return max(0, self.CONTENT_H - vis_h)

    def ClampScroll(self):
        self.scroll_offset = max(0, min(self.scroll_offset, self.MaxScroll()))

    def TranslateEvent(self, ev):
        if ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            mx, my = ev.pos
            if mx >= self.grid_w:
                class _FE:
                    pass
                fe=_FE(); fe.type=ev.type
                fe.pos=(mx-self.grid_w, my+self.scroll_offset)
                if hasattr(ev,'button'): fe.button=ev.button
                return fe
        return ev

    # ────────────────────────────────────────────
    #  DRAW GRID
    # ────────────────────────────────────────────
    def DrawGrid(self):
        self.screen.fill(C_BG, (0, 0, self.grid_w, self.grid_h))
        path_set = set(self.path)
        pulse = 0.5 + 0.5*math.sin(self.tick * 0.12)   # 0‥1 pulsing value

        for r in range(self.rows):
            for c in range(self.cols):
                nd  = (r,c)
                rect= self.cellRect(r,c)

                if nd in self.walls:
                    # Walls: dark slate with subtle top-edge highlight
                    pygame.draw.rect(self.screen, C_WALL, rect)
                    pygame.draw.line(self.screen, C_WALL_HL,
                                     (rect.x, rect.y), (rect.right, rect.y))
                    continue

                if nd == self.agent_pos:
                    # Agent cell: pulsing orange glow
                    glow_a = int(60 + 40*pulse)
                    pygame.draw.rect(self.screen, C_EMPTY, rect)
                    gs = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
                    gs.fill((*C_AGENT, glow_a))
                    self.screen.blit(gs, rect.topleft)
                elif nd == self.start:
                    pygame.draw.rect(self.screen, (0, 40, 30), rect)
                    pygame.draw.rect(self.screen, C_START, rect, 2)
                elif nd == self.goal:
                    pygame.draw.rect(self.screen, (40, 0, 20), rect)
                    pygame.draw.rect(self.screen, C_GOAL, rect, 2)
                elif nd in path_set:
                    pygame.draw.rect(self.screen, (0, 35, 45), rect)
                    pygame.draw.rect(self.screen, C_PATH, rect, 1)
                    # Bright dot in center
                    cx = rect.x + rect.w//2; cy = rect.y + rect.h//2
                    pygame.draw.circle(self.screen, C_PATH, (cx,cy), max(2, self.cell//6))
                elif nd in self.trail:
                    pygame.draw.rect(self.screen, C_TRAIL, rect)
                elif nd in self.visited:
                    # Visited: deep blue with subtle brightness based on recency
                    pygame.draw.rect(self.screen, C_VISITED, rect)
                elif nd in self.frontier:
                    # Frontier: neon yellow pulsing
                    t = 0.6 + 0.4*pulse
                    fc = lerpColor(C_VISITED2, C_FRONTIER, t)
                    pygame.draw.rect(self.screen, fc, rect)
                else:
                    pygame.draw.rect(self.screen, C_EMPTY, rect)

        # Grid lines
        if self.cell >= 12:
            for ri in range(self.rows+1):
                pygame.draw.line(self.screen, C_GRID_LINE,
                                 (0, ri*self.cell),(self.grid_w, ri*self.cell))
            for ci in range(self.cols+1):
                pygame.draw.line(self.screen, C_GRID_LINE,
                                 (ci*self.cell,0),(ci*self.cell,self.grid_h))

        # Start / Goal icons
        self.DrawNodeIcon(self.start, C_START, "S")
        self.DrawNodeIcon(self.goal,  C_GOAL,  "G")

        # Agent — pulsing circle with outer ring
        ar,ac = self.agent_pos
        cx = ac*self.cell+self.cell//2; cy = ar*self.cell+self.cell//2
        rad = max(3, self.cell//2 - 2)
        ring_r = rad + 3 + int(3*pulse)
        ring_a = int(180 - 120*pulse)
        # Outer glow ring
        ring_surf = pygame.Surface((ring_r*2+4, ring_r*2+4), pygame.SRCALPHA)
        pygame.draw.circle(ring_surf, (*C_AGENT, ring_a),
                           (ring_r+2, ring_r+2), ring_r, 2)
        self.screen.blit(ring_surf, (cx-ring_r-2, cy-ring_r-2))
        # Agent body
        pygame.draw.circle(self.screen, C_AGENT, (cx,cy), rad)
        pygame.draw.circle(self.screen, WHITE,   (cx,cy), max(2, rad//3))

        # Flash overlay
        if self.flash_frames > 0 and self.flash_color:
            a = int(120 * self.flash_frames / 16)
            fs = pygame.Surface((self.grid_w, self.grid_h), pygame.SRCALPHA)
            fs.fill((*self.flash_color, a))
            self.screen.blit(fs, (0,0))
            self.flash_frames -= 1

        # Scanline overlay (every other row, very subtle)
        for sy in range(0, self.grid_h, 2):
            sl = pygame.Surface((self.grid_w, 1), pygame.SRCALPHA)
            sl.fill((0,0,0,18))
            self.screen.blit(sl, (0,sy))

    def DrawNodeIcon(self, pos, color, letter):
        r,c = pos
        rect = self.cellRect(r,c)
        if self.cell >= 18:
            f = pygame.font.SysFont("Consolas", max(10, self.cell//2), bold=True)
            lbl = f.render(letter, True, color)
            self.screen.blit(lbl, lbl.get_rect(center=rect.center))

    # ────────────────────────────────────────────
    #  DRAW PANEL (offscreen surface → clipped blit)
    # ────────────────────────────────────────────
    def DrawPanel(self):
        win_h  = self.screen.get_height()
        vis_h  = win_h - self.STATUS_BAR_H
        gx     = self.grid_w
        mouse_local = (pygame.mouse.get_pos()[0]-gx,
                       pygame.mouse.get_pos()[1]+self.scroll_offset)

        # ── Offscreen surface ─────────────────────
        off = pygame.Surface((PANEL_W, self.CONTENT_H))

        # Gradient background
        fillGradientVertical(off, (0,0,PANEL_W,self.CONTENT_H), P_BG1, P_BG2)

        # Subtle vertical neon accent line on left edge
        for i in range(3):
            a = [80, 40, 15][i]
            pygame.draw.line(off, (*P_GLOW_C, 255), (i,0),(i,self.CONTENT_H))

        # ── Fonts ─────────────────────────────────
        f_title = pygame.font.SysFont("Consolas", 13, bold=True)
        f_sec   = pygame.font.SysFont("Consolas", 14, bold=True)
        f_small = pygame.font.SysFont("Consolas", 11)
        f_val   = pygame.font.SysFont("Consolas", 14, bold=True)
        f_big   = pygame.font.SysFont("Consolas", 16, bold=True)

        W  = PANEL_W - 20
        OX = 10

        # ── Section header helper ──────────────────
        def secHdr(text, y, color=P_GLOW_C):
            band_h = 22
            band_surf = pygame.Surface((W+8, band_h), pygame.SRCALPHA)
            band_surf.fill((*tuple(c//5 for c in color), 180))
            off.blit(band_surf, (OX-4, y))
            # Inside sec_hdr function in draw_panel
            pygame.draw.line(off, color, (OX-4, y+band_h), (OX+W+4, y+band_h), 1)
            pygame.draw.rect(off, color, (OX-4, y, 3, band_h))
            pygame.draw.rect(off, color, (OX+W+1, y, 3, band_h))
            lbl = f_sec.render(text, True, color)
            off.blit(lbl, lbl.get_rect(centerx=PANEL_W//2, y=y+3))
            gs = pygame.Surface((W+8, 1), pygame.SRCALPHA)
            gs.fill((*color, 60))
            off.blit(gs, (OX-4, y+band_h-1))

        # ── TITLE block ───────────────────────────
        ty = self.Y_TITLE
        # Title card background
        pygame.draw.rect(off, (16,10,36), (OX-2, ty, W+4, 38), border_radius=8)
        drawGlowRect(off, P_GLOW_C, (OX-2, ty, W+4, 38), radius=8, glow_r=6, alpha=60)
        t1 = f_title.render("⬡  PATHFINDING AGENT", True, P_GLOW_C)
        t2 = f_small.render("Dynamic Re-Planning System", True, DIM)
        off.blit(t1, t1.get_rect(centerx=PANEL_W//2, y=ty+4))
        off.blit(t2, t2.get_rect(centerx=PANEL_W//2, y=ty+22))

        # ── Section labels ────────────────────────
        secHdr("ALGORITHM",        self.Y_ALG,  P_GLOW_C)
        secHdr("HEURISTIC",        self.Y_HEU,  P_GLOW_G)
        secHdr("EDIT MODE",        self.Y_EDIT, (180,180,255))
        secHdr("ACTIONS",          self.Y_ACT,  P_GLOW_C)
        secHdr("DYNAMIC OBSTACLE", self.Y_DYN,  P_GLOW_M)
        secHdr("PARAMETERS",       self.Y_SLD,  P_GLOW_C)

        # ── Buttons ───────────────────────────────
        for b in self.buttons.values():
            b.draw_on(off, mouse_local)

        # ── Sliders ───────────────────────────────
        self.sl_spawn.draw(off)
        self.sl_aspeed.draw(off)
        self.sl_sspeed.draw(off)

        # ── METRICS (glassmorphism cards) ─────────
        secHdr("METRICS", self.Y_MET, P_GLOW_C)
        my = self.Y_MET + 18

        metric_data = [
            ("NODES VISITED",  f"{self.m_nodes}",          P_GLOW_C,  "◈"),
            ("PATH COST",      f"{self.m_cost:.2f}",       P_GLOW_G,  "◈"),
            ("SEARCH TIME",    f"{self.m_time:.1f} ms",    (180,180,255),"◈"),
            ("RE-PLANS",       f"{self.m_replans}",        P_GLOW_M,  "⚡"),
            ("TOTAL DIST",     f"{self.m_total:.2f}",      C_FRONTIER, "◈"),
            ("WALLS SPAWNED",  f"{self.m_spawned}",        C_GOAL,    "◈"),
        ]

        card_h = 26; card_gap = 4
        for i, (label, val, color, icon) in enumerate(metric_data):
            cy2 = my + i*(card_h+card_gap)
            # Card background
            pygame.draw.rect(off, P_CARD, (OX, cy2, W, card_h), border_radius=5)
            # Left accent bar
            pygame.draw.rect(off, color,  (OX, cy2, 3, card_h), border_radius=2)
            # Label
            off.blit(f_small.render(f" {icon} {label}", True, DIM),
                     (OX+8, cy2+7))
            # Value (right-aligned)
            vs = f_val.render(val, True, color)
            off.blit(vs, (OX+W-vs.get_width()-4, cy2+6))

        # ── LEGEND ───────────────────────────────
        secHdr("LEGEND", self.Y_LEG, (180,180,255))
        ly = self.Y_LEG + 18
        legend = [
            (C_FRONTIER,"FRONTIER"), (C_VISITED, "VISITED"),
            (C_PATH,    "PATH"),     (C_TRAIL,   "TRAIL"),
            (C_START,   "START"),    (C_GOAL,    "GOAL"),
            (C_AGENT,   "AGENT"),    (C_WALL,    "WALL"),
        ]
        iw = W // 4
        for i,(col,txt) in enumerate(legend):
            ix = OX + (i%4)*iw
            iy = ly + (i//4)*22
            # Colored dot
            pygame.draw.circle(off, col, (ix+6, iy+7), 5)
            pygame.draw.circle(off, tuple(min(255,c+80) for c in col), (ix+6,iy+7), 5, 1)
            off.blit(f_small.render(txt, True, DIM), (ix+14, iy))

        # ── Keyboard hints ────────────────────────
        hint_y = self.Y_LEG + 68
        hints = "[R] Search  [A] Agent  [D] Dynamic  [N] Maze  [C] Clear"
        off.blit(f_small.render(hints, True, (50,50,80)), (OX, hint_y))

        # ── Scroll arrows ─────────────────────────
        if self.scroll_offset > 0:
            _draw_arrow(off, (PANEL_W//2, 16), up=True)
        if self.scroll_offset < self.MaxScroll():
            _draw_arrow(off, (PANEL_W//2, self.CONTENT_H-16), up=False)

        # ── Blit visible slice ────────────────────
        self.screen.blit(off, (gx, 0),
                         pygame.Rect(0, self.scroll_offset, PANEL_W, vis_h))

        # ── Panel left border glow ────────────────
        for i in range(3):
            a = [120,60,20][i]
            pygame.draw.line(self.screen, (*P_GLOW_C, 255),
                             (gx+i, 0),(gx+i, win_h))

        # ── Scrollbar ────────────────────────────
        if self.MaxScroll() > 0:
            sb_x  = gx + PANEL_W - 4
            sb_h  = vis_h
            th    = max(20, int(vis_h*vis_h/self.CONTENT_H))
            ty2   = int(self.scroll_offset/self.MaxScroll()*(sb_h-th))
            pygame.draw.rect(self.screen, (25,25,45), (sb_x,0,4,sb_h))
            pygame.draw.rect(self.screen, P_GLOW_C,  (sb_x, ty2, 4, th), border_radius=2)

        # ── STATUS BAR (pinned) ───────────────────
        bar_y = win_h - self.STATUS_BAR_H
        meta  = self.ST_META.get(self.state, (DIM, self.state, ""))
        sc, st_label, _ = meta

        # Status bar background
        status_surf = pygame.Surface((PANEL_W, self.STATUS_BAR_H))
        fillGradientVertical(status_surf,
                               (0,0,PANEL_W,self.STATUS_BAR_H),
                               (12,8,28), (18,12,40))
        self.screen.blit(status_surf, (gx, bar_y))

        # Top border glow
        for i in range(2):
            pygame.draw.line(self.screen, (*sc, 120-i*60),
                             (gx, bar_y+i),(gx+PANEL_W, bar_y+i))

        # Status dot + label
        pygame.draw.circle(self.screen, sc,    (gx+14, bar_y+self.STATUS_BAR_H//2), 6)
        pygame.draw.circle(self.screen, WHITE, (gx+14, bar_y+self.STATUS_BAR_H//2), 6, 1)
        st_surf = f_big.render(st_label, True, sc)
        self.screen.blit(st_surf, (gx+26, bar_y + self.STATUS_BAR_H//2 - st_surf.get_height()//2))

    # ────────────────────────────────────────────
    #  MASTER DRAW
    # ────────────────────────────────────────────
    def draw(self):
        self.screen.fill(C_BG)
        self.DrawGrid()
        self.DrawPanel()
        pygame.display.flip()

    # ────────────────────────────────────────────
    #  SEARCH CONTROL
    # ────────────────────────────────────────────
    def runAnimated(self):
        self._clear_state(); self.agent_pos=self.start; self.trail=set()
        self.search_gen=SearchAnimated(self.start,self.goal,self.rows,
                                        self.cols,self.walls,self.algorithm,self.heuristic)
        self.state=self.ST_SEARCHING

    def runInstantAnimate(self):
        self._clear_state(); self.agent_pos=self.start; self.trail=set()
        res=searchInstantly(self.start,self.goal,self.rows,self.cols,
                           self.walls,self.algorithm,self.heuristic)
        self.visited=res["visited"]; self.frontier=res["frontier"]
        self.m_nodes=res["nodes"]; self.m_time=res["time_ms"]
        if res["found"]:
            self.path=res["path"]; self.m_cost=res["cost"]
            self.m_total=res["cost"]; self.path_index=0; self.agent_timer=0
            self.state=self.ST_MOVING
        else:
            self.state=self.ST_NO_PATH

    def replan(self, pos):
        res=searchInstantly(pos,self.goal,self.rows,self.cols,
                           self.walls,self.algorithm,self.heuristic)
        self.visited=res["visited"]; self.frontier=res["frontier"]
        self.m_nodes+=res["nodes"]; self.m_time+=res["time_ms"]
        if res["found"]:
            self.path=res["path"]; self.m_cost=res["cost"]
            self.m_total+=res["cost"]; self.path_index=0
            self.agent_timer=0; self.m_replans+=1
            self.flash(C_REPLAN); self.state=self.ST_MOVING
        else:
            self.path=[]; self.state=self.ST_NO_PATH

    def stepAnimated(self, n=1):
        if not self.search_gen: return
        for _ in range(n):
            try:
                st=next(self.search_gen)
                self.visited=st["visited"]; self.frontier=st["frontier"]
                self.m_nodes=st["nodes"]
                if st["done"]:
                    self.path=st["path"]; self.m_cost=st["cost"]
                    self.m_time=st["time_ms"]; self.search_gen=None
                    self.state=self.ST_NO_PATH if not self.path else self.ST_IDLE
                    break
            except StopIteration:
                self.search_gen=None; self.state=self.ST_IDLE; break

    # ────────────────────────────────────────────
    #  AGENT TICK
    # ────────────────────────────────────────────
    def tickAgent(self):
        if self.state!=self.ST_MOVING: return
        self.agent_timer+=1; self.UpdAgentInterval()
        if self.agent_timer<self.agent_interval: return
        self.agent_timer=0

        if self.dynamic_mode: self.maybeSpawn()

        if self.path_index+1>=len(self.path):
            self.agent_pos=self.goal; self.trail.add(self.agent_pos)
            self.state=self.ST_ARRIVED; self.flash(C_ARRIVE); return

        self.path_index+=1
        nxt=self.path[self.path_index]
        if nxt in self.walls:
            self.replan(self.agent_pos); return

        self.trail.add(self.agent_pos); self.agent_pos=nxt
        if self.agent_pos==self.goal:
            self.trail.add(self.agent_pos)
            self.state=self.ST_ARRIVED; self.flash(C_ARRIVE)

    def maybeSpawn(self):
        if random.random()>self.sl_spawn.value: return
        free=[(r,c) for r in range(self.rows) for c in range(self.cols)
              if (r,c) not in self.walls
              and (r,c) not in (self.agent_pos,self.start,self.goal)]
        if not free: return
        w=random.choice(free); self.walls.add(w); self.m_spawned+=1
        if w in self.path[self.path_index:]:
            self.state=self.ST_REPLAN; self.replan(self.agent_pos)

    def flash(self, col):
        self.flash_color=col; self.flash_frames=16

    # ────────────────────────────────────────────
    #  CLEAR / RESET
    # ────────────────────────────────────────────
    def _clear_state(self):
        self.visited=set(); self.frontier=set(); self.path=[]
        self.search_gen=None; self.state=self.ST_IDLE
        self.m_nodes=0; self.m_cost=0.0; self.m_time=0.0
        self.m_replans=0; self.m_total=0.0; self.m_spawned=0
        self.flash_frames=0; self.trail=set()
        self.path_index=0; self.agent_timer=0

    def clearMaze(self):    self._clear_state(); self.agent_pos=self.start
    def newMaze(self): self.clearMaze(); self.walls=generateMaze(self.rows,self.cols,self.density,self.start,self.goal)
    def resetMaze(self):    self.clearMaze(); self.walls=set()

    # ────────────────────────────────────────────
    #  EVENTS
    # ────────────────────────────────────────────
    def handleEvents(self):
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT: pygame.quit(); sys.exit()

            if ev.type==pygame.MOUSEWHEEL:
                mx,_=pygame.mouse.get_pos()
                if mx>=self.grid_w:
                    self.scroll_offset-=ev.y*30; self.ClampScroll()

            sl_ev=self.TranslateEvent(ev)
            self.sl_spawn.handle(sl_ev)
            self.sl_aspeed.handle(sl_ev)
            self.sl_sspeed.handle(sl_ev)

            if ev.type==pygame.KEYDOWN:
                k=ev.key
                if   k==pygame.K_r: self._run_animated()
                elif k==pygame.K_a: self.runInstantAnimate()
                elif k==pygame.K_c: self.clearMaze()
                elif k==pygame.K_n: self.newMaze()
                elif k==pygame.K_d: self.toggleDynamic()

            if ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1: self.dragging=True
            if ev.type==pygame.MOUSEBUTTONUP   and ev.button==1: self.dragging=False

            if ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
                mx,my=ev.pos
                if mx<self.grid_w:
                    nd=self.cell_at(mx,my)
                    if nd: self.gridClick(nd)
                else:
                    self.panelClick(ev)

        if self.dragging and self.edit_mode=="wall":
            mx,my=pygame.mouse.get_pos()
            if mx<self.grid_w:
                nd=self.cell_at(mx,my)
                if nd and nd not in (self.start,self.goal):
                    if   pygame.mouse.get_pressed()[0]: self.walls.add(nd)
                    elif pygame.mouse.get_pressed()[2]: self.walls.discard(nd)

    def gridClick(self, nd):
        if self.state in (self.ST_MOVING, self.ST_SEARCHING): return
        if   self.edit_mode=="wall":
            if nd not in (self.start,self.goal): self.walls^={nd}
        elif self.edit_mode=="start":
            if nd not in self.walls and nd!=self.goal:
                self.start=nd; self.agent_pos=nd
        elif self.edit_mode=="goal":
            if nd not in self.walls and nd!=self.start: self.goal=nd

    def panelClick(self, ev):
        lx=ev.pos[0]-self.grid_w; ly=ev.pos[1]+self.scroll_offset
        class _FE:
            type=pygame.MOUSEBUTTONDOWN; button=1
            def __init__(self,x,y): self.pos=(x,y)
        lev=_FE(lx,ly); B=self.buttons

        def alg(a):
            self.algorithm=a
            B["alg_astar"].active=(a=="A*"); B["alg_gbfs"].active=(a=="GBFS")
        def heur(h):
            self.heuristic=h
            B["h_man"].active=(h=="Manhattan"); B["h_euc"].active=(h=="Euclidean")
        def mode(m):
            self.edit_mode=m
            for k in ("mode_wall","mode_start","mode_goal"): B[k].active=False
            B["mode_"+m].active=True

        if B["alg_astar"].clicked(lev): alg("A*")
        if B["alg_gbfs"].clicked(lev):  alg("GBFS")
        if B["h_man"].clicked(lev):     heur("Manhattan")
        if B["h_euc"].clicked(lev):     heur("Euclidean")
        if B["mode_wall"].clicked(lev):  mode("wall")
        if B["mode_start"].clicked(lev): mode("start")
        if B["mode_goal"].clicked(lev):  mode("goal")

        if B["run"].clicked(lev):     self._run_animated()
        if B["animate"].clicked(lev): self.runInstantAnimate()
        if B["step_s"].clicked(lev):
            if self.state==self.ST_IDLE and not self.search_gen: self._run_animated()
            self.stepAnimated(1)
        if B["clear"].clicked(lev):   self.clearMaze()
        if B["maze"].clicked(lev):    self.newMaze()
        if B["reset"].clicked(lev):   self.resetMaze()
        if B["config"].clicked(lev):  self.reconfigure()
        if B["dynmode"].clicked(lev): self.toggleDynamic()

    def toggleDynamic(self):
        self.dynamic_mode=not self.dynamic_mode
        b=self.buttons["dynmode"]; b.active=self.dynamic_mode
        b.text=f"DYNAMIC MODE : {'ON ' if self.dynamic_mode else 'OFF'}"

    def reconfigure(self):
        root=Tk(); root.withdraw()
        r=simpledialog.askinteger("Rows","Rows (5-60):",minvalue=5,maxvalue=60,initialvalue=self.rows)
        c=simpledialog.askinteger("Cols","Cols (5-60):",minvalue=5,maxvalue=60,initialvalue=self.cols)
        d=simpledialog.askfloat("Density","Obstacle density (0.0–0.6):",minvalue=0.0,maxvalue=0.6,initialvalue=self.density)
        root.destroy()
        if r and c and d is not None:
            self.rows=r; self.cols=c; self.density=d
            self.cell=max(MIN_CELL,min(MAX_CELL,650//max(r,c)))
            self.grid_w=c*self.cell; self.grid_h=r*self.cell
            win_h=max(self.grid_h,500)
            self.screen=pygame.display.set_mode((self.grid_w+PANEL_W,win_h))
            self.start=(0,0); self.goal=(r-1,c-1)
            self.newMaze(); self.Build_UI()

    # ────────────────────────────────────────────
    #  MAIN LOOP
    # ────────────────────────────────────────────
    def run(self):
        while True:
            self.tick+=1
            self.handleEvents()
            if self.state==self.ST_SEARCHING and self.search_gen:
                self.stepAnimated(self.sl_sspeed.int_val)
            self.tickAgent()
            self.draw()
            self.clock.tick(FPS)


# ══════════════════════════════════════════════════════════════
#  MODULE-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════
def _draw_arrow(surf, center, up=True):
    cx, cy = center; d = 7
    pts = [(cx,cy-d),(cx-d,cy+d),(cx+d,cy+d)] if up else \
          [(cx,cy+d),(cx-d,cy-d),(cx+d,cy-d)]
    pygame.draw.polygon(surf, (80,80,110), pts)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root=Tk(); root.withdraw()
    rows    = simpledialog.askinteger("Grid Setup","Rows (5–60):",   minvalue=5, maxvalue=60, initialvalue=20)
    cols    = simpledialog.askinteger("Grid Setup","Cols (5–60):",   minvalue=5, maxvalue=60, initialvalue=20)
    density = simpledialog.askfloat  ("Grid Setup","Obstacle density (0.0–0.6):",
                                      minvalue=0.0,maxvalue=0.6,initialvalue=0.28)
    root.destroy()
    PathfindingApp(rows=rows or 20,
                   cols=cols or 20,
                   density=density if density is not None else 0.28).run()