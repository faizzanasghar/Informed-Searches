"""
Dynamic Pathfinding Agent — Phase 1
====================================
Features:
  - Dynamic grid sizing (user-defined rows x cols)
  - A* and Greedy Best-First Search (GBFS)
  - Heuristics: Manhattan & Euclidean
  - 8-directional movement
  - Interactive map editor (click to toggle walls)
  - Random maze generation with configurable obstacle density
  - Visualization: frontier (yellow), visited (blue), path (green)
  - Real-time metrics: nodes visited, path cost, execution time
"""

import pygame
import sys
import math
import heapq
import random
import time
from tkinter import Tk, simpledialog

# ─────────────────────────────────────────────
#  CONSTANTS & COLORS
# ─────────────────────────────────────────────
WHITE      = (255, 255, 255)
BLACK      = (20,  20,  20)
GRAY       = (40,  40,  40)
LIGHT_GRAY = (180, 180, 180)
DARK_GRAY  = (60,  60,  60)

COLOR_EMPTY    = (30,  30,  30)   # empty cell
COLOR_WALL     = (80,  80,  80)   # obstacle
COLOR_START    = (0,   200, 80)   # green
COLOR_GOAL     = (220, 50,  50)   # red
COLOR_FRONTIER = (230, 200, 0)    # yellow  — in priority queue
COLOR_VISITED  = (50,  120, 220)  # blue    — expanded
COLOR_PATH     = (0,   220, 150)  # teal-green — final path
COLOR_AGENT    = (255, 140, 0)    # orange  — agent dot

PANEL_W   = 320          # right panel width
FPS       = 120
MIN_CELL  = 10
MAX_CELL  = 80

# ─────────────────────────────────────────────
#  HEURISTICS
# ─────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

HEURISTICS = {"Manhattan": manhattan, "Euclidean": euclidean}

# ─────────────────────────────────────────────
#  8-DIRECTIONAL NEIGHBORS
# ─────────────────────────────────────────────
DIRECTIONS = [(-1,0),(1,0),(0,-1),(0,1),
              (-1,-1),(-1,1),(1,-1),(1,1)]

def get_neighbors(node, rows, cols, walls):
    r, c = node
    neighbors = []
    for dr, dc in DIRECTIONS:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in walls:
            # Diagonal cost = sqrt(2), cardinal = 1
            cost = math.sqrt(2) if dr != 0 and dc != 0 else 1.0
            neighbors.append(((nr, nc), cost))
    return neighbors

# ─────────────────────────────────────────────
#  SEARCH ALGORITHMS  (generator — yields states for animation)
# ─────────────────────────────────────────────
def astar(start, goal, rows, cols, walls, heuristic):
    h = HEURISTICS[heuristic]
    open_set = []
    heapq.heappush(open_set, (h(start, goal), 0, start))
    came_from = {start: None}
    g_score   = {start: 0}
    frontier_set = {start}
    visited  = set()

    t0 = time.perf_counter()

    while open_set:
        _, g, current = heapq.heappop(open_set)
        frontier_set.discard(current)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = reconstruct_path(came_from, goal)
            elapsed = (time.perf_counter() - t0) * 1000
            yield {"done": True, "path": path, "visited": visited,
                   "frontier": frontier_set, "nodes": len(visited),
                   "cost": g_score[goal], "time_ms": elapsed}
            return

        for neighbor, cost in get_neighbors(current, rows, cols, walls):
            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + h(neighbor, goal)
                heapq.heappush(open_set, (f, tentative_g, neighbor))
                came_from[neighbor] = current
                frontier_set.add(neighbor)

        yield {"done": False, "visited": set(visited),
               "frontier": set(frontier_set), "nodes": len(visited),
               "cost": g_score.get(goal, 0), "time_ms": 0}

    elapsed = (time.perf_counter() - t0) * 1000
    yield {"done": True, "path": [], "visited": visited,
           "frontier": frontier_set, "nodes": len(visited),
           "cost": 0, "time_ms": elapsed}


def gbfs(start, goal, rows, cols, walls, heuristic):
    h = HEURISTICS[heuristic]
    open_set = []
    heapq.heappush(open_set, (h(start, goal), start))
    came_from = {start: None}
    frontier_set = {start}
    visited = set()

    t0 = time.perf_counter()

    while open_set:
        _, current = heapq.heappop(open_set)
        frontier_set.discard(current)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = reconstruct_path(came_from, goal)
            elapsed = (time.perf_counter() - t0) * 1000
            # compute path cost for GBFS
            cost = sum(
                math.sqrt(2) if abs(path[i][0]-path[i-1][0])==1 and abs(path[i][1]-path[i-1][1])==1 else 1
                for i in range(1, len(path))
            )
            yield {"done": True, "path": path, "visited": visited,
                   "frontier": frontier_set, "nodes": len(visited),
                   "cost": cost, "time_ms": elapsed}
            return

        for neighbor, _ in get_neighbors(current, rows, cols, walls):
            if neighbor not in visited and neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(open_set, (h(neighbor, goal), neighbor))
                frontier_set.add(neighbor)

        yield {"done": False, "visited": set(visited),
               "frontier": set(frontier_set), "nodes": len(visited),
               "cost": 0, "time_ms": 0}

    elapsed = (time.perf_counter() - t0) * 1000
    yield {"done": True, "path": [], "visited": visited,
           "frontier": frontier_set, "nodes": len(visited),
           "cost": 0, "time_ms": elapsed}


def reconstruct_path(came_from, goal):
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path

# ─────────────────────────────────────────────
#  MAZE GENERATOR  (random obstacle density)
# ─────────────────────────────────────────────
def generate_maze(rows, cols, density, start, goal):
    walls = set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) == start or (r, c) == goal:
                continue
            if random.random() < density:
                walls.add((r, c))
    return walls

# ─────────────────────────────────────────────
#  UI BUTTON HELPER
# ─────────────────────────────────────────────
class Button:
    def __init__(self, rect, text, color=(70,70,90), hover=(100,100,130),
                 font_size=16, text_color=WHITE, toggle=False):
        self.rect       = pygame.Rect(rect)
        self.text       = text
        self.color      = color
        self.hover      = hover
        self.font_size  = font_size
        self.text_color = text_color
        self.toggle     = toggle
        self.active     = False          # only used for toggle buttons
        self._font      = None

    def font(self):
        if self._font is None:
            self._font = pygame.font.SysFont("Segoe UI", self.font_size, bold=True)
        return self._font

    def draw(self, surface):
        mouse = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse)
        if self.toggle and self.active:
            bg = (40, 160, 100)
        elif hovered:
            bg = self.hover
        else:
            bg = self.color
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, LIGHT_GRAY, self.rect, 1, border_radius=6)
        label = self.font().render(self.text, True, self.text_color)
        surface.blit(label, label.get_rect(center=self.rect.center))

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False

# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
class PathfindingApp:
    def __init__(self, rows=20, cols=20, density=0.28):
        pygame.init()
        self.rows    = rows
        self.cols    = cols
        self.density = density

        # Compute cell size to fit an 800-tall grid
        self.cell = max(MIN_CELL, min(MAX_CELL, 800 // max(rows, cols)))
        grid_w = self.cols * self.cell
        grid_h = self.rows * self.cell
        win_w  = grid_w + PANEL_W
        win_h  = max(grid_h, 600)
        self.screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Dynamic Pathfinding Agent — Phase 1")
        self.clock = pygame.time.Clock()

        self.grid_w = grid_w
        self.grid_h = grid_h

        # State
        self.start = (0, 0)
        self.goal  = (rows-1, cols-1)
        self.walls = generate_maze(rows, cols, density, self.start, self.goal)

        self.visited  = set()
        self.frontier = set()
        self.path     = []
        self.search_gen = None
        self.running_search = False
        self.no_path   = False

        # Metrics
        self.metric_nodes = 0
        self.metric_cost  = 0.0
        self.metric_time  = 0.0

        # Edit mode: "wall" | "start" | "goal"
        self.edit_mode = "wall"
        self.dragging  = False

        # Algorithm & heuristic selection
        self.algorithm = "A*"           # "A*" or "GBFS"
        self.heuristic = "Manhattan"    # "Manhattan" or "Euclidean"

        # Animation speed (steps per frame)
        self.speed = 5

        self._build_ui()

    # ── UI Layout ────────────────────────────
    def _build_ui(self):
        px = self.grid_w + 10
        self.buttons = {}

        def btn(key, y, text, **kw):
            self.buttons[key] = Button((px, y, PANEL_W-20, 34), text, **kw)

        # Algorithm toggles
        self.buttons["alg_astar"] = Button((px, 10, 125, 34), "A*",     toggle=True)
        self.buttons["alg_gbfs"]  = Button((px+135, 10, 125, 34), "GBFS",  toggle=True)
        self.buttons["alg_astar"].active = True

        # Heuristic toggles
        self.buttons["h_manhattan"] = Button((px, 54, 125, 34), "Manhattan", toggle=True)
        self.buttons["h_euclidean"] = Button((px+135, 54, 125, 34), "Euclidean", toggle=True)
        self.buttons["h_manhattan"].active = True

        # Edit mode
        btn("mode_wall",  100, "✏  Draw Walls",   toggle=True)
        btn("mode_start", 144, "🚀 Move Start",    toggle=True)
        btn("mode_goal",  188, "🏁 Move Goal",     toggle=True)
        self.buttons["mode_wall"].active = True

        # Actions
        btn("run",      240, "▶  Run Search",    color=(30,120,60), hover=(50,160,80))
        btn("step",     284, "⏭  Step",          color=(60,80,120), hover=(80,110,160))
        btn("clear",    328, "🗑  Clear Path",    color=(100,60,30), hover=(140,80,40))
        btn("maze",     372, "🌀 New Maze",       color=(60,60,110), hover=(90,90,150))
        btn("reset",    416, "↺  Reset All",      color=(80,30,30),  hover=(120,50,50))
        btn("config",   460, "⚙  Reconfigure",   color=(50,50,80),  hover=(80,80,120))

        # Speed slider label (drawn manually)
        self.speed_rect = pygame.Rect(px, 510, PANEL_W-20, 20)

    # ── Grid ↔ Pixel conversions ─────────────
    def cell_at(self, px, py):
        c = px // self.cell
        r = py // self.cell
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return None

    def cell_rect(self, r, c):
        return pygame.Rect(c*self.cell, r*self.cell, self.cell-1, self.cell-1)

    # ── Drawing ──────────────────────────────
    def draw_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                node = (r, c)
                rect = self.cell_rect(r, c)
                if node in self.walls:
                    color = COLOR_WALL
                elif node == self.start:
                    color = COLOR_START
                elif node == self.goal:
                    color = COLOR_GOAL
                elif node in self.path:
                    color = COLOR_PATH
                elif node in self.visited:
                    color = COLOR_VISITED
                elif node in self.frontier:
                    color = COLOR_FRONTIER
                else:
                    color = COLOR_EMPTY
                pygame.draw.rect(self.screen, color, rect)

        # Grid lines (only if cells are large enough)
        if self.cell >= 14:
            for r in range(self.rows+1):
                pygame.draw.line(self.screen, DARK_GRAY,
                                 (0, r*self.cell), (self.grid_w, r*self.cell))
            for c in range(self.cols+1):
                pygame.draw.line(self.screen, DARK_GRAY,
                                 (c*self.cell, 0), (c*self.cell, self.grid_h))

    def draw_panel(self):
        panel_rect = pygame.Rect(self.grid_w, 0,
                                 self.screen.get_width()-self.grid_w,
                                 self.screen.get_height())
        pygame.draw.rect(self.screen, (25,25,35), panel_rect)

        for btn in self.buttons.values():
            btn.draw(self.screen)

        font_sm = pygame.font.SysFont("Segoe UI", 14)
        font_md = pygame.font.SysFont("Segoe UI", 15, bold=True)
        font_lg = pygame.font.SysFont("Segoe UI", 18, bold=True)

        px = self.grid_w + 10

        # Section labels
        def label(text, y, color=LIGHT_GRAY, f=font_sm):
            surf = f.render(text, True, color)
            self.screen.blit(surf, (px, y))

        label("ALGORITHM",     -2,  f=font_sm)  # above buttons at y=10 → label at 0 looks off
        # shift labels above button rows
        self.screen.blit(font_sm.render("ALGORITHM", True, LIGHT_GRAY), (px, 0))
        self.screen.blit(font_sm.render("HEURISTIC", True, LIGHT_GRAY), (px, 44))
        self.screen.blit(font_sm.render("EDIT MODE", True, LIGHT_GRAY), (px, 90))

        # Metrics box
        my = 540
        pygame.draw.rect(self.screen, (35,35,50),
                         (px-2, my, PANEL_W-16, 130), border_radius=6)
        pygame.draw.rect(self.screen, DARK_GRAY,
                         (px-2, my, PANEL_W-16, 130), 1, border_radius=6)
        self.screen.blit(font_md.render("── METRICS ──", True, (150,150,200)), (px+10, my+6))

        no_path_txt = "  NO PATH FOUND" if self.no_path else ""
        metrics = [
            ("Nodes Visited", f"{self.metric_nodes}"),
            ("Path Cost",     f"{self.metric_cost:.2f}"),
            ("Time (ms)",     f"{self.metric_time:.2f}"),
        ]
        for i, (k, v) in enumerate(metrics):
            self.screen.blit(font_sm.render(k+":", True, LIGHT_GRAY), (px+6, my+30+i*26))
            self.screen.blit(font_lg.render(v, True, COLOR_PATH),     (px+130, my+28+i*26))

        if self.no_path:
            self.screen.blit(font_md.render("⚠ NO PATH FOUND", True, (220,80,80)),
                             (px+4, my+110))

        # Speed label
        self.screen.blit(font_sm.render(f"Animation Speed: {self.speed} steps/frame",
                                        True, LIGHT_GRAY), (px, 502))

        # Legend
        ly = self.screen.get_height() - 90
        legend = [
            (COLOR_FRONTIER, "Frontier"),
            (COLOR_VISITED,  "Visited"),
            (COLOR_PATH,     "Path"),
            (COLOR_START,    "Start"),
            (COLOR_GOAL,     "Goal"),
            (COLOR_WALL,     "Wall"),
        ]
        self.screen.blit(font_sm.render("LEGEND", True, LIGHT_GRAY), (px, ly-16))
        for i, (col, txt) in enumerate(legend):
            x = px + (i % 3) * 88
            y = ly + (i // 3) * 22
            pygame.draw.rect(self.screen, col, (x, y+2, 12, 12), border_radius=2)
            self.screen.blit(font_sm.render(txt, True, LIGHT_GRAY), (x+16, y))

    def draw(self):
        self.screen.fill(BLACK)
        self.draw_grid()
        self.draw_panel()
        pygame.display.flip()

    # ── Search Control ───────────────────────
    def start_search(self):
        self.visited  = set()
        self.frontier = set()
        self.path     = []
        self.no_path  = False
        self.metric_nodes = 0
        self.metric_cost  = 0.0
        self.metric_time  = 0.0

        algo = self.algorithm
        h    = self.heuristic
        if algo == "A*":
            self.search_gen = astar(self.start, self.goal,
                                    self.rows, self.cols, self.walls, h)
        else:
            self.search_gen = gbfs(self.start, self.goal,
                                   self.rows, self.cols, self.walls, h)
        self.running_search = True

    def step_search(self, steps=1):
        if not self.search_gen:
            return
        for _ in range(steps):
            try:
                state = next(self.search_gen)
                self.visited  = state["visited"]
                self.frontier = state["frontier"]
                self.metric_nodes = state["nodes"]
                if state["done"]:
                    self.path         = state["path"]
                    self.metric_cost  = state["cost"]
                    self.metric_time  = state["time_ms"]
                    self.running_search = False
                    if not self.path:
                        self.no_path = True
                    self.search_gen = None
                    break
            except StopIteration:
                self.running_search = False
                self.search_gen = None
                break

    # ── Event Handling ───────────────────────
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            # ── Keyboard shortcuts ──
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.start_search()
                elif event.key == pygame.K_c:
                    self._clear_path()
                elif event.key == pygame.K_n:
                    self._new_maze()
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.speed = min(50, self.speed + 1)
                elif event.key == pygame.K_MINUS:
                    self.speed = max(1, self.speed - 1)
                elif event.key == pygame.K_SPACE:
                    if self.running_search:
                        self.step_search(1)

            # ── Mouse drag for wall drawing ──
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.dragging = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos

                # Clicked inside grid
                if mx < self.grid_w:
                    node = self.cell_at(mx, my)
                    if node:
                        self._handle_grid_click(node)
                else:
                    # Clicked panel buttons
                    self._handle_panel_click(event)

            # Scroll wheel → change speed
            if event.type == pygame.MOUSEWHEEL:
                self.speed = max(1, min(50, self.speed + event.y))

        # Dragging walls
        if self.dragging and self.edit_mode == "wall":
            mx, my = pygame.mouse.get_pos()
            if mx < self.grid_w:
                node = self.cell_at(mx, my)
                if node and node != self.start and node != self.goal:
                    if pygame.mouse.get_pressed()[0]:
                        self.walls.add(node)
                    elif pygame.mouse.get_pressed()[2]:
                        self.walls.discard(node)

    def _handle_grid_click(self, node):
        if self.edit_mode == "wall":
            if node != self.start and node != self.goal:
                if node in self.walls:
                    self.walls.discard(node)
                else:
                    self.walls.add(node)
        elif self.edit_mode == "start":
            if node not in self.walls and node != self.goal:
                self.start = node
        elif self.edit_mode == "goal":
            if node not in self.walls and node != self.start:
                self.goal = node

    def _handle_panel_click(self, event):
        B = self.buttons

        # Algorithm selection
        if B["alg_astar"].is_clicked(event):
            self.algorithm = "A*";   B["alg_astar"].active=True;  B["alg_gbfs"].active=False
        if B["alg_gbfs"].is_clicked(event):
            self.algorithm = "GBFS"; B["alg_gbfs"].active=True;   B["alg_astar"].active=False

        # Heuristic selection
        if B["h_manhattan"].is_clicked(event):
            self.heuristic = "Manhattan"; B["h_manhattan"].active=True; B["h_euclidean"].active=False
        if B["h_euclidean"].is_clicked(event):
            self.heuristic = "Euclidean"; B["h_euclidean"].active=True; B["h_manhattan"].active=False

        # Edit mode
        if B["mode_wall"].is_clicked(event):
            self.edit_mode="wall";  B["mode_wall"].active=True; B["mode_start"].active=False; B["mode_goal"].active=False
        if B["mode_start"].is_clicked(event):
            self.edit_mode="start"; B["mode_start"].active=True; B["mode_wall"].active=False; B["mode_goal"].active=False
        if B["mode_goal"].is_clicked(event):
            self.edit_mode="goal";  B["mode_goal"].active=True; B["mode_wall"].active=False; B["mode_start"].active=False

        # Actions
        if B["run"].is_clicked(event):
            self.start_search()
        if B["step"].is_clicked(event):
            if not self.running_search:
                self.start_search()
            self.step_search(1)
        if B["clear"].is_clicked(event):
            self._clear_path()
        if B["maze"].is_clicked(event):
            self._new_maze()
        if B["reset"].is_clicked(event):
            self._reset_all()
        if B["config"].is_clicked(event):
            self._reconfigure()

    def _clear_path(self):
        self.visited=set(); self.frontier=set(); self.path=[]
        self.running_search=False; self.search_gen=None; self.no_path=False
        self.metric_nodes=0; self.metric_cost=0.0; self.metric_time=0.0

    def _new_maze(self):
        self._clear_path()
        self.walls = generate_maze(self.rows, self.cols, self.density,
                                   self.start, self.goal)

    def _reset_all(self):
        self._clear_path()
        self.walls = set()

    def _reconfigure(self):
        """Open Tkinter dialogs to reconfigure grid."""
        root = Tk(); root.withdraw()
        r = simpledialog.askinteger("Rows", "Enter number of rows (5-60):",
                                    minvalue=5, maxvalue=60, initialvalue=self.rows)
        c = simpledialog.askinteger("Cols", "Enter number of cols (5-60):",
                                    minvalue=5, maxvalue=60, initialvalue=self.cols)
        d = simpledialog.askfloat("Density", "Obstacle density (0.0–0.6):",
                                   minvalue=0.0, maxvalue=0.6, initialvalue=self.density)
        root.destroy()
        if r and c and d is not None:
            self.rows    = r
            self.cols    = c
            self.density = d
            self.cell = max(MIN_CELL, min(MAX_CELL, 800 // max(r, c)))
            self.grid_w  = self.cols * self.cell
            self.grid_h  = self.rows * self.cell
            win_w = self.grid_w + PANEL_W
            win_h = max(self.grid_h, 600)
            self.screen  = pygame.display.set_mode((win_w, win_h))
            self.start   = (0, 0)
            self.goal    = (r-1, c-1)
            self._new_maze()
            self._build_ui()

    # ── Main Loop ────────────────────────────
    def run(self):
        while True:
            self.handle_events()
            if self.running_search:
                self.step_search(self.speed)
            self.draw()
            self.clock.tick(FPS)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Ask grid size on startup
    root = Tk(); root.withdraw()
    rows = simpledialog.askinteger("Grid Setup", "Number of rows (5-60):",
                                   minvalue=5, maxvalue=60, initialvalue=20)
    cols = simpledialog.askinteger("Grid Setup", "Number of columns (5-60):",
                                   minvalue=5, maxvalue=60, initialvalue=20)
    density = simpledialog.askfloat("Grid Setup", "Obstacle density (0.0 – 0.6):",
                                    minvalue=0.0, maxvalue=0.6, initialvalue=0.28)
    root.destroy()

    rows    = rows    or 20
    cols    = cols    or 20
    density = density if density is not None else 0.28

    app = PathfindingApp(rows=rows, cols=cols, density=density)
    app.run()