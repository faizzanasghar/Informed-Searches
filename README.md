# Dynamic Pathfinding Agent

This project implements an interactive pathfinding visualizer using Python and Pygame. It allows you to explore and compare two popular search algorithms in a grid-based environment: **A\* Search** and **Greedy Best-First Search (GBFS)**.

The project includes a premium "Cyber Edition" GUI (in `main2.py`) with advanced visual effects, animations, and dynamic obstacles, alongside the Phase 1 base implementation (`main.py`).

## Features

- **Search Algorithms:** 
  - A* Search (Pathfinding algorithm that finds the shortest path)
  - Greedy Best-First Search (Faster but does not guarantee the shortest path)
- **Heuristics:** Manhattan Distance and Euclidean Distance.
- **Interactive Grid Editor:** Draw or erase walls, and easily reposition the Start and Goal nodes.
- **Random Maze Generation:** Automatically generate obstacles with a configurable density.
- **Dynamic Mode:** Obstacles can randomly spawn during the agent's movement, forcing the algorithm to re-plan in real-time.
- **Real-Time Metrics:**
  - Nodes Visited
  - Path Cost
  - Search Execution Time
  - Total Distance Traveled
  - Re-planning Count (in Dynamic Mode)
- **Animation Controls:** Adjust both the Search Animation Speed and Agent Movement Speed via sliders.
- **Cyber GUI (Phase 2):** Features a sleek dark theme with glowing neon grid cells, glassmorphism UI metric cards, pulsing animations, and a rich control panel.

## Prerequisites

- **Python 3.x**
- **Pygame** (Used for rendering the GUI)
- **Tkinter** (Standard Python library for configuration dialogs)

To install the required dependencies, run:

```bash
pip install pygame
```

## How to Run

You can run the full, feature-rich version (Phase 2) by executing `main2.py`:

```bash
python main2.py
```

Upon launching, a configuration dialog will appear asking for:
- Number of Rows (5-60)
- Number of Columns (5-60)
- Obstacle Density (0.0 to 0.6)

After confirming the grid setup, the main visualizer will open.

*(If you wish to view the base Phase 1 implementation without the advanced animations and dynamic replanning GUI, you can run `python main.py`)*

## Keyboard Controls

- `[R]` - Run Search instantly
- `[A]` - Animate Agent movement
- `[D]` - Toggle Dynamic Obstacles Mode
- `[N]` - Generate new random maze
- `[C]` - Clear path and visited nodes
- `[Space]` - Step through the search animation
- `[-]` / `[+]` or `[Mouse Scroll]` - Adjust speed

## Project Structure

- `main2.py`: The Phase 2 Cyber Edition script containing the complete implementation of the visualizer, dynamic routing, GUI logic, and search algorithms (`searchInstantly` and `SearchAnimated`).
- `main.py`: The Phase 1 base implementation script.
