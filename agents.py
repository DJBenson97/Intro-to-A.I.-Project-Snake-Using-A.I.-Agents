import random
import heapq

from collections import deque


class RandomAgent:
    def get_action(self, state, valid_moves):
        return random.choice(valid_moves)
    
class GreedyAgent:
    def get_action(self, state, valid_moves):
        head_x, head_y = state["snake"][0]
        food_x, food_y = state["food"]

        best_move = None
        best_distance = float("inf")

        for move in valid_moves:
            dx, dy = {
                "UP": (0, -1),
                "DOWN": (0, 1),
                "LEFT": (-1, 0),
                "RIGHT": (1, 0)
            }[move]

            new_x = head_x + dx
            new_y = head_y + dy

            distance = abs(new_x - food_x) + abs(new_y - food_y)

            if distance < best_distance:
                best_distance = distance
                best_move = move

        return best_move

class BFSAgent:
    def get_action(self, state, valid_moves):
        start = state["snake"][0]
        food = state["food"]
        snake_body = set(state["snake"])

        queue = deque()
        queue.append((start, []))  # (position, path)

        visited = set()
        visited.add(start)

        while queue:
            current, path = queue.popleft()

            if current == food:
                return path[0] if path else valid_moves[0]

            for move, (dx, dy) in {
                "UP": (0, -1),
                "DOWN": (0, 1),
                "LEFT": (-1, 0),
                "RIGHT": (1, 0)
            }.items():
                new_pos = (current[0] + dx, current[1] + dy)

                if new_pos in visited:
                    continue

                x, y = new_pos
                if x < 0 or x >= state["width"] or y < 0 or y >= state["height"]:
                    continue

                if new_pos in snake_body:
                    continue

                visited.add(new_pos)
                queue.append((new_pos, path + [move]))

        # fallback if no path found
        return valid_moves[0]

class AStarAgent:
    def get_action(self, state, valid_moves):
        start = state["snake"][0]
        goal = state["food"]
        snake_body = set(state["snake"])

        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        pq = []
        heapq.heappush(pq, (0 + heuristic(start), 0, start, []))
        visited = set()

        while pq:
            f, g, current, path = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return path[0] if path else valid_moves[0]

            for move, (dx, dy) in {
                "UP": (0, -1),
                "DOWN": (0, 1),
                "LEFT": (-1, 0),
                "RIGHT": (1, 0)
            }.items():
                new_pos = (current[0] + dx, current[1] + dy)

                x, y = new_pos
                if x < 0 or x >= state["width"] or y < 0 or y >= state["height"]:
                    continue
                if new_pos in snake_body:
                    continue
                if new_pos in visited:
                    continue

                new_g = g + 1
                new_f = new_g + heuristic(new_pos)

                heapq.heappush(pq, (new_f, new_g, new_pos, path + [move]))

        return valid_moves[0]

class AStarBayesAgent:
    def get_action(self, state, valid_moves):
        start = state["snake"][0]
        goal = state["food"]
        snake_body = set(state["snake"])

        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # estimate "probability of death" using reachable space
        def risk(pos):
            visited = set()
            queue = deque([pos])
            count = 0

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                count += 1

                for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                    nx, ny = current[0] + dx, current[1] + dy

                    if (nx, ny) in visited:
                        continue
                    if nx < 0 or nx >= state["width"] or ny < 0 or ny >= state["height"]:
                        continue
                    if (nx, ny) in snake_body:
                        continue

                    queue.append((nx, ny))

            # smaller space = higher risk
            return 1 / (count + 1)

        pq = []
        heapq.heappush(pq, (heuristic(start), 0, start, []))
        visited = set()

        while pq:
            f, g, current, path = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return path[0] if path else valid_moves[0]

            for move, (dx, dy) in {
                "UP": (0, -1),
                "DOWN": (0, 1),
                "LEFT": (-1, 0),
                "RIGHT": (1, 0)
            }.items():
                new_pos = (current[0] + dx, current[1] + dy)

                x, y = new_pos
                if x < 0 or x >= state["width"] or y < 0 or y >= state["height"]:
                    continue
                if new_pos in snake_body:
                    continue
                if new_pos in visited:
                    continue

                new_g = g + 1
                r = risk(new_pos)

                # λ controls how much risk matters (we may want to tune this better)
                lam = 50
                new_f = new_g + heuristic(new_pos) + lam * r

                heapq.heappush(pq, (new_f, new_g, new_pos, path + [move]))

        return valid_moves[0]

class AStarBayesLocalAgent:
    def get_action(self, state, valid_moves):
        start = state["snake"][0]
        goal = state["food"]
        snake_body = set(state["snake"])

        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # standard A* to get best path (no risk inside search)
        def astar():
            pq = []
            heapq.heappush(pq, (heuristic(start), 0, start, []))
            visited = set()

            while pq:
                f, g, current, path = heapq.heappop(pq)

                if current in visited:
                    continue
                visited.add(current)

                if current == goal:
                    return path

                for move, (dx, dy) in {
                    "UP": (0, -1),
                    "DOWN": (0, 1),
                    "LEFT": (-1, 0),
                    "RIGHT": (1, 0)
                }.items():
                    new_pos = (current[0] + dx, current[1] + dy)

                    x, y = new_pos
                    if x < 0 or x >= state["width"] or y < 0 or y >= state["height"]:
                        continue
                    if new_pos in snake_body:
                        continue
                    if new_pos in visited:
                        continue

                    new_g = g + 1
                    new_f = new_g + heuristic(new_pos)

                    heapq.heappush(pq, (new_f, new_g, new_pos, path + [move]))

            return None

        # risk = inverse reachable space (limited BFS for speed)
        def risk(pos):
            visited = set()
            queue = deque([pos])
            count = 0
            max_cells = 100  # limit for speed

            while queue and count < max_cells:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                count += 1

                for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                    nx, ny = current[0] + dx, current[1] + dy

                    if (nx, ny) in visited:
                        continue
                    if nx < 0 or nx >= state["width"] or ny < 0 or ny >= state["height"]:
                        continue
                    if (nx, ny) in snake_body:
                        continue

                    queue.append((nx, ny))

            return 1 / (count + 1)

        # get best A* path
        path = astar()

        # if no path, fallback
        if not path:
            return valid_moves[0]

        # evaluate ONLY first move using risk
        best_move = None
        best_score = float("inf")

        lam = 20  # smaller since we only apply locally

        head_x, head_y = start

        for move in valid_moves:
            dx, dy = {
                "UP": (0, -1),
                "DOWN": (0, 1),
                "LEFT": (-1, 0),
                "RIGHT": (1, 0)
            }[move]

            new_pos = (head_x + dx, head_y + dy)

            h = heuristic(new_pos)
            r = risk(new_pos)

            score = h + lam * r

            if score < best_score:
                best_score = score
                best_move = move

        return best_move

class AStarTailSafeAgent:
    def get_action(self, state, valid_moves):
        start = state["snake"][0]
        goal = state["food"]
        snake = state["snake"]

        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # A* to food
        def astar_to_food():
            pq = []
            heapq.heappush(pq, (heuristic(start), 0, start, []))
            visited = set()
            snake_body = set(snake)

            while pq:
                f, g, current, path = heapq.heappop(pq)

                if current in visited:
                    continue
                visited.add(current)

                if current == goal:
                    return path

                for move, (dx, dy) in {
                    "UP": (0, -1),
                    "DOWN": (0, 1),
                    "LEFT": (-1, 0),
                    "RIGHT": (1, 0)
                }.items():
                    new_pos = (current[0] + dx, current[1] + dy)

                    x, y = new_pos
                    if x < 0 or x >= state["width"] or y < 0 or y >= state["height"]:
                        continue
                    if new_pos in snake_body:
                        continue
                    if new_pos in visited:
                        continue

                    new_g = g + 1
                    new_f = new_g + heuristic(new_pos)

                    heapq.heappush(pq, (new_f, new_g, new_pos, path + [move]))

            return None

        # BFS to check reachability
        def can_reach_tail(sim_snake):
            head = sim_snake[0]
            tail = sim_snake[-1]
            body = set(sim_snake[:-1])  # tail is allowed to move

            queue = deque([head])
            visited = set()

            while queue:
                current = queue.popleft()
                if current == tail:
                    return True

                for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                    nx, ny = current[0] + dx, current[1] + dy

                    if (nx, ny) in visited:
                        continue
                    if nx < 0 or nx >= state["width"] or ny < 0 or ny >= state["height"]:
                        continue
                    if (nx, ny) in body:
                        continue

                    visited.add((nx, ny))
                    queue.append((nx, ny))

            return False

        # simulate snake after path
        def simulate_path(path):
            sim_snake = snake.copy()

            for move in path:
                dx, dy = {
                    "UP": (0, -1),
                    "DOWN": (0, 1),
                    "LEFT": (-1, 0),
                    "RIGHT": (1, 0)
                }[move]

                new_head = (sim_snake[0][0] + dx, sim_snake[0][1] + dy)
                sim_snake.insert(0, new_head)

                if new_head == goal:
                    break
                else:
                    sim_snake.pop()

            return sim_snake

        # main logic
        path = astar_to_food()

        if path:
            sim_snake = simulate_path(path)

            # only take path if tail still reachable
            if can_reach_tail(sim_snake):
                return path[0]

        # fallback: move that maximizes space (avoid dying)
        best_move = valid_moves[0]
        best_space = -1

        for move in valid_moves:
            dx, dy = {
                "UP": (0, -1),
                "DOWN": (0, 1),
                "LEFT": (-1, 0),
                "RIGHT": (1, 0)
            }[move]

            new_pos = (start[0] + dx, start[1] + dy)

            # measure open space
            visited = set()
            queue = deque([new_pos])
            count = 0

            while queue and count < 200:
                cur = queue.popleft()
                if cur in visited:
                    continue
                visited.add(cur)
                count += 1

                for dx2, dy2 in [(0,-1),(0,1),(-1,0),(1,0)]:
                    nx, ny = cur[0] + dx2, cur[1] + dy2

                    if (nx, ny) in visited:
                        continue
                    if nx < 0 or nx >= state["width"] or ny < 0 or ny >= state["height"]:
                        continue
                    if (nx, ny) in snake:
                        continue

                    queue.append((nx, ny))

            if count > best_space:
                best_space = count
                best_move = move

        return best_move