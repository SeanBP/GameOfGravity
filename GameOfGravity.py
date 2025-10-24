import pygame
import random
import copy
import threading
from typing import Tuple, Optional
import copy
import math
import sys

# ---------- Game Logic ----------
EMPTY = None

HOLE = "O"

TILE_SIZE = 80
BUTTON_HEIGHT = 40
FOOTER_HEIGHT = 40
WHITE = "W"
BLACK = "B"

DIRECTIONS = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "W": (0, -1),
    #"NE": (-1, 1),
    #"NW": (-1, -1),
    #"SE": (1, 1),
    #"SW": (1, -1),
}

class Game:
    def __init__(self, n=7):
        if n % 2 == 0:
            raise ValueError("Board size must be odd")
        self.n = n
        self.board = [[EMPTY for _ in range(n)] for _ in range(n)]
        self.hole = (n // 2, n // 2)
        self.board[self.hole[0]][self.hole[1]] = HOLE

        # --- Place Black pieces (top row) ---
        for c in range(n):
            self.board[0][c] = BLACK

        # --- Place White pieces (bottom row) ---
        for c in range(n):
            self.board[n-1][c] = WHITE


        # Game state
        self.current_player = WHITE
        self.winner: Optional[str] = None
        self.turn_count: int = 0
        self.last_skipped: bool = False

        # Pull tracking
        self.last_pulled_from: Optional[Tuple[int, int]] = None
        self.last_pulled_player: Optional[str] = None
        self.last_pulled_ship: Optional[Tuple[int, int]] = None

        self.previous_states = set()
        self.save_state()

    def serialize_board(self):
        return tuple(tuple(row) for row in self.board)

    def save_state(self):
        self.previous_states.add(self.serialize_board())

    def in_bounds(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n

    def compute_los(self, r, c):
        """Compute line-of-sight targets in all 8 directions."""
        enemy = BLACK if self.current_player == WHITE else WHITE
        visible = []
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            while self.in_bounds(nr, nc):
                cell = self.board[nr][nc]
                if cell == self.current_player:
                    break
                elif cell == enemy:
                    # Ensure at least one space between piece and target
                    if (dr == 0 or dc == 0) and abs(nr - r) + abs(nc - c) > 1:
                        # Cardinal directions
                        visible.append((nr, nc))
                    elif dr != 0 and dc != 0 and abs(nr - r) > 1:
                        # Diagonal directions: must be at least 1 diagonal step away
                        visible.append((nr, nc))
                    break
                nr += dr
                nc += dc
        return visible

    def list_valid_moves(self):
        moves = []
        for r in range(self.n):
            for c in range(self.n):
                if self.board[r][c] != self.current_player:
                    continue
                for dr, dc in DIRECTIONS.values():
                    nr, nc = r + dr, c + dc
                    if not self.in_bounds(nr, nc) or self.board[nr][nc] != EMPTY:
                        continue
                    # Prevent undoing last pull
                    #if (
                    #    self.last_pulled_ship == (r, c) and
                    #    (nr, nc) == self.last_pulled_from and
                    #    self.current_player == self.last_pulled_player
                    #):
                    #    continue

                    # --- Chebyshev distance check ---  # <-- NEW
                    old_dist = max(abs(r - self.hole[0]), abs(c - self.hole[1]))  # <-- NEW
                    new_dist = max(abs(nr - self.hole[0]), abs(nc - self.hole[1]))  # <-- NEW
                    if new_dist > old_dist:  # move would increase distance  # <-- NEW
                        continue  # <-- NEW

                    # Temporarily apply the move
                    self.board[r][c] = EMPTY
                    self.board[nr][nc] = self.current_player

                    targets = self.compute_los(nr, nc)
                    move_dir = (nr - r, nc - c)

                    for target in targets:
                        tr, tc = target
                        target_player = self.board[tr][tc]

                        # --- Cannot pull from the direction you just came ---
                        pull_dir = (tr - nr, tc - nc)
                        pull_dir = (
                            0 if pull_dir[0] == 0 else pull_dir[0] // abs(pull_dir[0]),
                            0 if pull_dir[1] == 0 else pull_dir[1] // abs(pull_dir[1])
                        )
                        if pull_dir == (-move_dir[0], -move_dir[1]):
                            continue

                        # Simulate pull
                        dr_pull = 0 if tr == nr else (-1 if tr > nr else 1)
                        dc_pull = 0 if tc == nc else (-1 if tc > nc else 1)
                        new_r, new_c = tr + dr_pull, tc + dc_pull

                        orig_target_cell = self.board[new_r][new_c]
                        self.board[tr][tc] = EMPTY
                        if self.board[new_r][new_c] != HOLE:
                            self.board[new_r][new_c] = target_player

                        final_state = self.serialize_board()
                        if final_state not in self.previous_states:
                            moves.append({'from': (r, c), 'to': (nr, nc), 'targets': [target]})

                        # Undo pull
                        if self.board[new_r][new_c] != HOLE:
                            self.board[new_r][new_c] = orig_target_cell
                        self.board[tr][tc] = target_player

                    # Undo move
                    self.board[nr][nc] = EMPTY
                    self.board[r][c] = self.current_player

        if not moves:
            moves.append({'from': None, 'to': None, 'targets': None})
        #if not moves:
        # Instead of None, use an empty list for targets
        #    moves.append({'from': None, 'to': None, 'targets': []})

        return moves

    def apply_move(self, move_from, move_to, target):
        # Skip turn
        if move_from is None:
            self.current_player = WHITE if self.current_player == BLACK else BLACK
            self.turn_count += 1
            self.last_skipped = True
            self.last_pulled_from = None
            self.last_pulled_player = None
            self.last_pulled_ship = None
            self.save_state()
            return

        fr_r, fr_c = move_from
        to_r, to_c = move_to
        moving_player = self.board[fr_r][fr_c]
        self.board[fr_r][fr_c] = EMPTY
        self.board[to_r][to_c] = moving_player

        # Pull target
        tr, tc = target
        target_player = self.board[tr][tc]
        dr = 0 if tr == to_r else (-1 if tr > to_r else 1)
        dc = 0 if tc == to_c else (-1 if tc > to_c else 1)
        new_r, new_c = tr + dr, tc + dc

        self.board[tr][tc] = EMPTY
        if self.board[new_r][new_c] == HOLE:
            self.last_pulled_ship = None
            self.winner = BLACK if target_player == WHITE else WHITE
        else:
            self.board[new_r][new_c] = target_player
            self.last_pulled_ship = (new_r, new_c)

        self.last_pulled_from = (tr, tc)
        self.last_pulled_player = target_player

        if not self.winner:
            self.turn_count += 1
            self.current_player = WHITE if self.current_player == BLACK else BLACK
            self.last_skipped = False

        self.save_state()




# ---------- Pygame GUI ----------
TILE_SIZE = 80
FPS = 30
BUTTON_HEIGHT = 50
FOOTER_HEIGHT = 60

LIGHT_COLOR = (240, 217, 181)
DARK_COLOR = (181, 136, 99)
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
HOLE_COLOR = (128, 0, 128)  # purple

SELECTED_COLOR = (0, 0, 255)
MOVE_COLOR = (0, 255, 0)
TARGET_COLOR = (255, 0, 0)

BUTTON_COLOR = (50, 150, 50)
BUTTON_HOVER = (80, 200, 80)
BUTTON_TEXT = (255, 255, 255)
BUTTON_WIDTH = 250
BUTTON_HEIGHT_MENU = 55

# --- draw functions ---
def draw_board(screen, game: Game, selected=None, valid_moves=None, valid_targets=None):
    n = game.n
    for r in range(n):
        for c in range(n):
            rect = pygame.Rect(c*TILE_SIZE, BUTTON_HEIGHT + r*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            color = LIGHT_COLOR if (r+c)%2==0 else DARK_COLOR
            pygame.draw.rect(screen, color, rect)
            piece = game.board[r][c]
            if piece == WHITE:
                pygame.draw.circle(screen, WHITE_COLOR, rect.center, TILE_SIZE//3)
                pygame.draw.circle(screen, BLACK_COLOR, rect.center, TILE_SIZE//3, 2)
            elif piece == BLACK:
                pygame.draw.circle(screen, BLACK_COLOR, rect.center, TILE_SIZE//3)
                pygame.draw.circle(screen, WHITE_COLOR, rect.center, TILE_SIZE//3, 2)
            elif piece == HOLE:
                pygame.draw.circle(screen, HOLE_COLOR, rect.center, TILE_SIZE//2)
            if selected == (r, c):
                pygame.draw.rect(screen, SELECTED_COLOR, rect, 3)
            if valid_moves and (r, c) in valid_moves:
                pygame.draw.rect(screen, MOVE_COLOR, rect, 3)
            if valid_targets and (r, c) in valid_targets:
                pygame.draw.rect(screen, TARGET_COLOR, rect, 3)

def draw_button(screen, rect, label, font):
    mouse_pos = pygame.mouse.get_pos()
    color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, color, rect)
    pygame.draw.rect(screen, BLACK_COLOR, rect, 2)
    text = font.render(label, True, BUTTON_TEXT)
    screen.blit(text, text.get_rect(center=rect.center))

def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        test_line = line + word + " "
        if font.size(test_line)[0] <= max_width - 80:
            line = test_line
        else:
            lines.append(line.strip())
            line = word + " "
    if line:
        lines.append(line.strip())
    return lines

# ---------- AI ----------
def ai_advanced_move(game: Game, depth=0, cancel_flag=lambda: False, apply_move=True):
    """
    Advanced AI move for Game.
    If depth=0, chooses a random valid move.
    If depth>0, performs a depth-limited search using a simple heuristic.
    """

    ai_color = game.current_player
    WIN_SCORE = sys.maxsize
    LOSS_SCORE = -sys.maxsize

    # --- Depth 0: random move ---
    if depth == 0:
        moves = game.list_valid_moves()
        move = random.choice(moves)
        target = random.choice(move["targets"]) if move["targets"] else None
        if apply_move:
            game.apply_move(move["from"], move["to"], target)
        else:
            return move
        return

    # --- Heuristic for a given board state ---
    def heuristic_score(state: Game, ai_color: str) -> int:
        hole_r, hole_c = state.hole
        opponent_color = WHITE if ai_color == BLACK else BLACK
        ai_score, opp_score = 0, 0

        for r in range(state.n):
            for c in range(state.n):
                cell = state.board[r][c]
                if cell in (EMPTY, HOLE):
                    continue

                # Chebyshev distance treats diagonal same as orthogonal
                dist = max(abs(r - hole_r), abs(c - hole_c))
                dist_sq = dist ** 2

                if cell == ai_color:
                    ai_score += 1 * dist_sq
                #elif cell == opponent_color:
                #    opp_score += 1 * dist_sq 

        return ai_score - opp_score


    # --- Recursive search ---
    def search(state: Game, current_depth: int, is_ai_turn: bool) -> Optional[tuple[int, int]]:
        if cancel_flag():
            return None
        if current_depth == depth:
            return heuristic_score(state, ai_color), current_depth

        moves = state.list_valid_moves()
        child_results = []

        for move in moves:
            # Skip turn
            if move["from"] is None:
                new_state = copy.deepcopy(state)
                new_state.apply_move(None, None, None)
                result = search(new_state, current_depth + 1, not is_ai_turn)
                if result is not None:
                    child_results.append(result)
                continue

            # Normal moves
            for target in move["targets"]:
                new_state = copy.deepcopy(state)
                new_state.apply_move(move["from"], move["to"], target)

                # Immediate win/loss pruning
                if new_state.winner is not None:
                    score = WIN_SCORE if new_state.winner == ai_color else LOSS_SCORE
                    child_results.append((score, current_depth))
                    continue

                result = search(new_state, current_depth + 1, not is_ai_turn)
                if result is not None:
                    child_results.append(result)

        if not child_results:
            return None

        # Max for AI, Min for opponent
        if is_ai_turn:
            return max(child_results, key=lambda r: (r[0], -r[1]))
        else:
            return min(child_results, key=lambda r: (r[0], -r[1]))

    # --- Root-level: pick the best move ---
    best_move = None
    best_score = (LOSS_SCORE, sys.maxsize)  # (score, depth)

    for move in game.list_valid_moves():
        # Skip turn
        if move["from"] is None and move["to"] is None:
            new_state = copy.deepcopy(game)
            new_state.apply_move(None, None, None)
            result = search(new_state, 1, is_ai_turn=False)
            if result is None:
                continue
            score, depth_reached = result
            if (score, -depth_reached) > (best_score[0], -best_score[1]):
                best_score = (score, depth_reached)
                best_move = (None, None, None)
            continue

        # Normal moves
        for target in move["targets"]:
            new_state = copy.deepcopy(game)
            new_state.apply_move(move["from"], move["to"], target)

            # Immediate win/loss pruning
            if new_state.winner is not None:
                score = WIN_SCORE if new_state.winner == ai_color else LOSS_SCORE
                if (score, 0) > (best_score[0], -best_score[1]):
                    best_score = (score, 0)
                    best_move = (move["from"], move["to"], target)
                continue

            result = search(new_state, 1, is_ai_turn=False)
            if result is None:
                continue
            score, depth_reached = result
            if (score, -depth_reached) > (best_score[0], -best_score[1]):
                best_score = (score, depth_reached)
                best_move = (move["from"], move["to"], target)

    # --- Apply or return ---
    if best_move and not cancel_flag():
        if apply_move:
            move_from, move_to, target = best_move
            game.apply_move(move_from, move_to, target)
        else:
            return best_move


# ---------- Menu ----------
def mode_selection(screen, clock, width, height, font):
    selecting_mode = True
    mode = None
    player_is_white = True
    grid_size = 7  # default
    ai_depth = 2   # default depth

    # --- Maximum AI depths per board size ---
    MAX_AI_DEPTH = {
        5: 8,
        7: 6,
        9: 5
    }

    # --- Buttons ---
    # Board size
    grid_5_btn = pygame.Rect(width//2 - 180, height//2 - 180, 100, BUTTON_HEIGHT_MENU)
    grid_7_btn = pygame.Rect(width//2 - 50, height//2 - 180, 100, BUTTON_HEIGHT_MENU)
    grid_9_btn = pygame.Rect(width//2 + 80, height//2 - 180, 100, BUTTON_HEIGHT_MENU)

    # Mode buttons
    pvp_btn = pygame.Rect(width//2 - BUTTON_WIDTH//2, height//2 - 90, BUTTON_WIDTH, BUTTON_HEIGHT_MENU)
    ai_white_btn = pygame.Rect(width//2 - BUTTON_WIDTH//2, height//2 - 20, BUTTON_WIDTH, BUTTON_HEIGHT_MENU)
    ai_black_btn = pygame.Rect(width//2 - BUTTON_WIDTH//2, height//2 + 50, BUTTON_WIDTH, BUTTON_HEIGHT_MENU)

    # AI depth slider
    slider_rect = pygame.Rect(width//2 - 150, height//2 + 120, 300, 20)
    knob_width = 20

    while selecting_mode:
        screen.fill((30,30,30))
        # --- Title ---
        title = font.render("Game of Gravity", True, (255,255,255))
        screen.blit(title, (width//2 - title.get_width()//2, height//2 - 250))

        # --- Board size buttons ---
        grid_buttons = [
            {"rect": grid_5_btn, "size": 5},
            {"rect": grid_7_btn, "size": 7},
            {"rect": grid_9_btn, "size": 9},
        ]
        for btn in grid_buttons:
            rect = btn["rect"]
            size = btn["size"]
            color = (200,200,50) if size == grid_size else BUTTON_COLOR
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK_COLOR, rect, 2)
            text = font.render(f"{size} x {size}", True, (0,0,0) if size == grid_size else BUTTON_TEXT)
            screen.blit(text, text.get_rect(center=rect.center))

        # --- Mode buttons ---
        draw_button(screen, pvp_btn, "Player vs Player", font)
        draw_button(screen, ai_white_btn, "Player vs AI (White)", font)
        draw_button(screen, ai_black_btn, "Player vs AI (Black)", font)

        # --- AI depth slider ---
        pygame.draw.rect(screen, (200,200,200), slider_rect)
        max_depth_for_grid = MAX_AI_DEPTH[grid_size]
        knob_x = slider_rect.x + (ai_depth / max_depth_for_grid) * (slider_rect.width - knob_width)
        knob_rect = pygame.Rect(knob_x, slider_rect.y - 5, knob_width, slider_rect.height + 10)
        pygame.draw.rect(screen, (255,255,0), knob_rect)
        depth_text = font.render(f"AI search depth: {ai_depth}", True, (255,255,255))
        screen.blit(depth_text, (width//2 - depth_text.get_width()//2, slider_rect.y + 30))

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None, None, None
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx,my = event.pos

                # Board size selection
                for btn in grid_buttons:
                    if btn["rect"].collidepoint((mx,my)):
                        grid_size = btn["size"]
                        # Adjust AI depth if exceeding new max
                        ai_depth = min(ai_depth, MAX_AI_DEPTH[grid_size])

                # Mode selection
                if pvp_btn.collidepoint((mx,my)):
                    mode = "pvp"
                    selecting_mode = False
                elif ai_white_btn.collidepoint((mx,my)):
                    mode = "ai"
                    player_is_white = False
                    selecting_mode = False
                elif ai_black_btn.collidepoint((mx,my)):
                    mode = "ai"
                    player_is_white = True
                    selecting_mode = False

                # Slider dragging
                if knob_rect.collidepoint((mx,my)) or slider_rect.collidepoint((mx,my)):
                    dragging = True
                    while dragging:
                        for e in pygame.event.get():
                            if e.type == pygame.MOUSEBUTTONUP:
                                dragging = False
                            elif e.type == pygame.MOUSEMOTION:
                                new_x = max(slider_rect.x, min(slider_rect.x + slider_rect.width - knob_width, e.pos[0] - knob_width//2))
                                ai_depth = round(max_depth_for_grid * (new_x - slider_rect.x)/(slider_rect.width - knob_width))
                                ai_depth = max(0, min(max_depth_for_grid, ai_depth))

                        # Redraw slider while dragging
                        screen.fill((30,30,30))
                        screen.blit(title, (width//2 - title.get_width()//2, height//2 - 250))
                        for btn in grid_buttons:
                            rect = btn["rect"]
                            size = btn["size"]
                            color = (200,200,50) if size == grid_size else BUTTON_COLOR
                            pygame.draw.rect(screen, color, rect)
                            pygame.draw.rect(screen, BLACK_COLOR, rect, 2)
                            text = font.render(f"{size} x {size}", True, (0,0,0) if size == grid_size else BUTTON_TEXT)
                            screen.blit(text, text.get_rect(center=rect.center))
                        draw_button(screen, pvp_btn, "Player vs Player", font)
                        draw_button(screen, ai_white_btn, "Player vs AI (White)", font)
                        draw_button(screen, ai_black_btn, "Player vs AI (Black)", font)
                        pygame.draw.rect(screen, (200,200,200), slider_rect)
                        knob_x = slider_rect.x + (ai_depth / max_depth_for_grid) * (slider_rect.width - knob_width)
                        knob_rect = pygame.Rect(knob_x, slider_rect.y - 5, knob_width, slider_rect.height + 10)
                        pygame.draw.rect(screen, (255,255,0), knob_rect)
                        depth_text = font.render(f"AI search depth: {ai_depth}", True, (255,255,255))
                        screen.blit(depth_text, (width//2 - depth_text.get_width()//2, slider_rect.y + 30))
                        pygame.display.flip()
                        clock.tick(FPS)

        pygame.display.flip()
        clock.tick(FPS)

    return mode, player_is_white, grid_size, ai_depth




def run_game():
    pygame.init()
    width_default, height_default = 7*TILE_SIZE, BUTTON_HEIGHT + 7*TILE_SIZE + FOOTER_HEIGHT
    screen = pygame.display.set_mode((width_default, height_default))
    pygame.display.set_caption("Game of Gravity")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)
    fontR = pygame.font.SysFont(None, 35)

    rules_text = [
        "Game of Gravity Rules:",
        "",
        "- The board is a square grid with a black hole at the center.",
        "- Each player starts with a row of ships along opposite edges.",
        "- White moves first.",
        "- On your turn, move one ship one square up, down, left, or right.",
        "- A move is valid only if:",
        "  • The destination is empty and on the board.",
        "  • It is not farther from the black hole than the starting square.",
        "  • It has line of sight to at least one enemy ship.",
        "  • Line of sight: same row or column, not adjacent, and unblocked.",
        "- After moving, pull one visible enemy ship one square closer.",
        "  • Cannot pull from the direction you just moved from.",
        #"  • Cannot undo last move's pull.",
        "- Win by pulling an enemy ship into the black hole."
    ]

    rules_width = 900
    line_height = fontR.get_height() + 12
    padding_top, padding_bottom, horizontal_margin = 30, 50, 20
    wrapped_lines = []
    for line in rules_text:
        wrapped_lines.extend(wrap_text(line, fontR, rules_width - 2*horizontal_margin))
    rules_height = padding_top + len(wrapped_lines)*line_height + padding_bottom

    while True:
        result = mode_selection(screen, clock, width_default, height_default, font)
        mode, player_is_white, grid_size, *rest = result
        if mode is None or grid_size is None:
            return
        ai_depth = rest[0] if rest else 0

        n = grid_size
        width, height = n*TILE_SIZE, BUTTON_HEIGHT + n*TILE_SIZE + FOOTER_HEIGHT
        screen = pygame.display.set_mode((width, height))

        # --- Initialize game state ---
        game = Game(n)
        game.current_player = WHITE  # Always start with WHITE
        game.winner = None

        selected = None
        valid_moves = []
        selecting_target = False
        valid_targets = []
        move_from = None
        move_to = None
        hint_move = None
        ai_thread = None
        ai_cancel = False
        hint_ai_thread = None
        hint_ai_cancel = False
        skipped_turns = 0
        waiting_skip = False
        skip_display_time = 0
        just_restarted = True  # new game or after restart
        showing_rules = False

        button_labels = ["Menu", "Restart", "Hint", "Rules"]
        num_buttons = len(button_labels)
        button_spacing = 15
        button_width = (width - button_spacing*(num_buttons-1)) // num_buttons
        button_height = 35
        footer_y = height - FOOTER_HEIGHT + 10
        buttons = []
        for i, label in enumerate(button_labels):
            btn = pygame.Rect(i*(button_width + button_spacing), footer_y, button_width, button_height)
            buttons.append(btn)
        menu_btn, restart_btn, hint_btn, rules_btn = buttons

        running = True
        while running:
            clock.tick(FPS)

            # Determine if AI's turn
            is_ai_turn = (mode == "ai") and (
                (game.current_player == WHITE and not player_is_white) or
                (game.current_player == BLACK and player_is_white)
            )

            ai_busy = (ai_thread and ai_thread.is_alive()) or (hint_ai_thread and hint_ai_thread.is_alive())

            # Clean up finished threads
            if ai_thread and not ai_thread.is_alive():
                ai_thread = None
            if hint_ai_thread and not hint_ai_thread.is_alive():
                hint_ai_thread = None

            # --- Automatic skip detection ---
            if not (just_restarted and ((game.current_player == WHITE and not player_is_white) or
                                        (game.current_player == BLACK and player_is_white))):
                valid_moves_player = [m for m in game.list_valid_moves() if m['from'] is not None]
                if not valid_moves_player and not game.winner:
                    if not waiting_skip:
                        waiting_skip = True
                        skip_display_time = pygame.time.get_ticks()
                else:
                    waiting_skip = False

                if waiting_skip and pygame.time.get_ticks() - skip_display_time >= 2000:
                    game.apply_move(None, None, None)
                    waiting_skip = False
                    skip_display_time = pygame.time.get_ticks()
                    skipped_turns += 1
                elif valid_moves_player:
                    skipped_turns = 0

                if skipped_turns >= 2:
                    game.winner = "Draw"

            # --- Event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if ai_thread and ai_thread.is_alive():
                        ai_cancel = True
                        ai_thread.join()
                    if hint_ai_thread and hint_ai_thread.is_alive():
                        hint_ai_cancel = True
                        hint_ai_thread.join()
                    return

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos

                    # Rules back button
                    if showing_rules:
                        back_btn = pygame.Rect(10, screen.get_height()-45, 120, 35)
                        if back_btn.collidepoint((mx, my)):
                            showing_rules = False
                            screen = pygame.display.set_mode((width, height))
                        continue

                    # Menu
                    if menu_btn.collidepoint((mx,my)):
                        if ai_thread and ai_thread.is_alive():
                            ai_cancel = True
                            ai_thread.join()
                        if hint_ai_thread and hint_ai_thread.is_alive():
                            hint_ai_cancel = True
                            hint_ai_thread.join()
                        running = False
                        screen = pygame.display.set_mode((width_default, height_default))
                        break

                    # Restart
                    if restart_btn.collidepoint((mx,my)):
                        if ai_busy:
                            continue
                        if ai_thread and ai_thread.is_alive():
                            ai_cancel = True
                            ai_thread.join()
                        if hint_ai_thread and hint_ai_thread.is_alive():
                            hint_ai_cancel = True
                            hint_ai_thread.join()

                        game = Game(n)
                        game.current_player = WHITE  # Always start with white
                        game.winner = None
                        selected = None
                        valid_moves = []
                        selecting_target = False
                        valid_targets = []
                        move_from = None
                        move_to = None
                        hint_move = None
                        ai_thread = None
                        ai_cancel = False
                        skipped_turns = 0
                        waiting_skip = False
                        skip_display_time = 0
                        just_restarted = True
                        continue

                    # Hint
                    if hint_btn.collidepoint((mx,my)) and not game.winner and not is_ai_turn:
                        if ai_busy:
                            continue
                        if hint_move:
                            move_from, move_to, target = hint_move
                            game.apply_move(move_from, move_to, target)
                            hint_move = None
                            selected, valid_moves = None, []
                            selecting_target, valid_targets = False, []
                            move_from, move_to = None, None
                        else:
                            if hint_ai_thread and hint_ai_thread.is_alive():
                                hint_ai_cancel = True
                                hint_ai_thread.join()
                            hint_ai_cancel = False
                            def hint_ai_job():
                                pygame.time.wait(200)
                                best_move = ai_advanced_move(
                                    game, depth=ai_depth,
                                    cancel_flag=lambda: hint_ai_cancel,
                                    apply_move=False
                                )
                                if not hint_ai_cancel:
                                    nonlocal hint_move
                                    hint_move = best_move
                            hint_ai_thread = threading.Thread(target=hint_ai_job)
                            hint_ai_thread.start()
                        continue

                    # Rules
                    if rules_btn.collidepoint((mx,my)):
                        showing_rules = True
                        screen = pygame.display.set_mode((rules_width, rules_height))
                        continue

                    # Board clicks
                    if game.winner or is_ai_turn:
                        continue
                    if BUTTON_HEIGHT <= my < BUTTON_HEIGHT + n*TILE_SIZE:
                        r, c = (my - BUTTON_HEIGHT)//TILE_SIZE, mx//TILE_SIZE
                        if hint_ai_thread and hint_ai_thread.is_alive():
                            hint_ai_cancel = True
                            hint_ai_thread.join()
                        hint_move = None
                        clicked_cell = game.board[r][c]

                        if selected and (r,c) == selected:
                            selected, valid_moves = None, []
                            selecting_target, valid_targets = False, []
                            move_from, move_to = None, None
                        elif clicked_cell == game.current_player:
                            selected = (r,c)
                            all_moves = game.list_valid_moves()
                            valid_moves = [m['to'] for m in all_moves if m['from'] == selected]
                            selecting_target, valid_targets = False, []
                            move_from, move_to = None, None
                        elif selected and (r,c) in valid_moves:
                            move_from = selected
                            move_to = (r,c)
                            temp_val = game.board[move_from[0]][move_from[1]]
                            game.board[move_from[0]][move_from[1]] = EMPTY
                            game.board[move_to[0]][move_to[1]] = game.current_player
                            valid_targets = game.compute_los(move_to[0], move_to[1])
                            game.board[move_to[0]][move_to[1]] = EMPTY
                            game.board[move_from[0]][move_from[1]] = temp_val
                            if valid_targets:
                                selecting_target = True
                            else:
                                game.apply_move(move_from, move_to, None)
                                selected, valid_moves = None, []
                                selecting_target, valid_targets = False, []
                                move_from, move_to = None, None
                        elif selecting_target and (r,c) in valid_targets:
                            game.apply_move(move_from, move_to, (r,c))
                            selected, valid_moves = None, []
                            selecting_target, valid_targets = False, []
                            move_from, move_to = None, None

            # --- AI automatic turn ---
            if is_ai_turn and ai_thread is None and not game.winner:
                # Block AI only if human is first after restart
                if just_restarted and ((game.current_player == WHITE and player_is_white) or
                                       (game.current_player == BLACK and not player_is_white)):
                    just_restarted = False
                else:
                    just_restarted = False
                    ai_cancel = False
                    def ai_job():
                        pygame.time.wait(1000)
                        ai_advanced_move(game, depth=ai_depth, cancel_flag=lambda: ai_cancel)
                    ai_thread = threading.Thread(target=ai_job)
                    ai_thread.start()

            # --- Draw ---
            screen.fill((0,0,0))
            footer_font = font if n != 5 else pygame.font.SysFont(None, 24)

            if showing_rules:
                y = padding_top
                max_y = screen.get_height() - padding_bottom
                for line in rules_text:
                    for wrapped in wrap_text(line, fontR, rules_width - 2*horizontal_margin):
                        if y + line_height > max_y:
                            break
                        text = fontR.render(wrapped, True, (255,255,255))
                        screen.blit(text, (horizontal_margin, y))
                        y += line_height
                back_btn = pygame.Rect(10, screen.get_height()-45, 120, 35)
                draw_button(screen, back_btn, "Back", font)
            else:
                draw_board(screen, game, selected, valid_moves, valid_targets)

                # Chebyshev rings
                cx, cy = game.hole[1], game.hole[0]
                max_ring = game.n // 2
                ring_color = (160, 32, 240)
                ring_thickness = 3
                for r in range(1, max_ring+1):
                    x = (cx - r) * TILE_SIZE
                    y = BUTTON_HEIGHT + (cy - r) * TILE_SIZE
                    size = (2*r + 1) * TILE_SIZE
                    pygame.draw.rect(screen, ring_color, pygame.Rect(x, y, size, size), ring_thickness)

                # Draw hint moves
                if hint_move is not None:
                    mf, mt, target = hint_move
                    if mf is not None:
                        pygame.draw.rect(screen, (0,255,255), pygame.Rect(
                            mf[1]*TILE_SIZE, BUTTON_HEIGHT + mf[0]*TILE_SIZE, TILE_SIZE, TILE_SIZE), 4)
                    if mt is not None:
                        pygame.draw.rect(screen, (255,0,255), pygame.Rect(
                            mt[1]*TILE_SIZE, BUTTON_HEIGHT + mt[0]*TILE_SIZE, TILE_SIZE, TILE_SIZE), 4)
                    if target is not None:
                        pygame.draw.rect(screen, (255,255,0), pygame.Rect(
                            target[1]*TILE_SIZE, BUTTON_HEIGHT + target[0]*TILE_SIZE, TILE_SIZE, TILE_SIZE), 4)

                # Footer buttons
                draw_button(screen, menu_btn, "Menu", font)
                draw_button(screen, restart_btn, "Restart", font)
                draw_button(screen, hint_btn, "Hint", font)
                draw_button(screen, rules_btn, "Rules", font)

                # Status text
                if ai_busy:
                    status_text = footer_font.render("AI is thinking...", True, (255,255,0))
                    screen.blit(status_text, (width//2 - status_text.get_width()//2, 10))
                elif game.winner:
                    msg = f"Winner: {game.winner}" if game.winner != "Draw" else "Draw!"
                    status_text = footer_font.render(msg, True, (255,0,0))
                    screen.blit(status_text, (width//2 - status_text.get_width()//2, 10))
                elif waiting_skip or game.last_skipped:
                    if pygame.time.get_ticks() - skip_display_time < 2000:
                        status_text = footer_font.render(f"{game.current_player} has no valid moves! Turn skipped.", True, (255,0,0))
                        screen.blit(status_text, (width//2 - status_text.get_width()//2, 10))

                # Turn
                turn_text = footer_font.render(f"Turn: {game.current_player}", True, (255,255,255))
                screen.blit(turn_text, (width - turn_text.get_width() - 10, 10))

            pygame.display.flip()




if __name__=="__main__":
    run_game()
