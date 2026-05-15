const GRID_SIZE = 7;
const MAX_MOVES = 25;
const EXIT = { row: GRID_SIZE - 1, col: GRID_SIZE - 1 };
const BLOCKED_CELLS = [{ row: Math.floor(GRID_SIZE / 2), col: Math.floor(GRID_SIZE / 2) }];

const board = document.querySelector("#board");
const moveStrip = document.querySelector("#move-strip");
const message = document.querySelector("#message");
const restartButton = document.querySelector("#restart-button");
const autoButton = document.querySelector("#auto-button");
const cells = [];
const moveBlocks = [];

const gameState = {
  player: { row: 0, col: 0 },
  movesLeft: MAX_MOVES,
  isOver: false,
};

const moves = {
  up: { row: -1, col: 0 },
  down: { row: 1, col: 0 },
  left: { row: 0, col: -1 },
  right: { row: 0, col: 1 },
};

function createBoard() {
  board.innerHTML = "";
  cells.length = 0;

  for (let row = 0; row < GRID_SIZE; row += 1) {
    for (let col = 0; col < GRID_SIZE; col += 1) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.row = row;
      cell.dataset.col = col;
      board.appendChild(cell);
      cells.push(cell);
    }
  }
}

function createMoveStrip() {
  moveStrip.innerHTML = "";
  moveBlocks.length = 0;

  for (let moveIndex = 1; moveIndex <= MAX_MOVES; moveIndex += 1) {
    const block = document.createElement("span");
    block.className = `move-block ${getMovePhase(moveIndex)}`;
    block.setAttribute("aria-hidden", "true");
    moveStrip.appendChild(block);
    moveBlocks.push(block);
  }
}

function getMovePhase(moveIndex) {
  if (moveIndex <= 10) {
    return "green";
  }

  if (moveIndex <= 20) {
    return "yellow";
  }

  return "red";
}

function randomStartPosition() {
  let row = EXIT.row;
  let col = EXIT.col;

  while (isAtExit({ row, col }) || isBlocked({ row, col })) {
    row = Math.floor(Math.random() * GRID_SIZE);
    col = Math.floor(Math.random() * GRID_SIZE);
  }

  return { row, col };
}

function resetGame() {
  gameState.player = randomStartPosition();
  gameState.movesLeft = MAX_MOVES;
  gameState.isOver = false;
  setMessage("", "");
  render();
}

function getCell(row, col) {
  return cells[row * GRID_SIZE + col];
}

function render() {
  cells.forEach((cell) => {
    cell.classList.remove("player", "exit", "blocked");
  });

  BLOCKED_CELLS.forEach((cell) => {
    getCell(cell.row, cell.col).classList.add("blocked");
  });
  getCell(EXIT.row, EXIT.col).classList.add("exit");
  getCell(gameState.player.row, gameState.player.col).classList.add("player");
  renderMoveStrip();
}

function renderMoveStrip() {
  const usedMoves = MAX_MOVES - gameState.movesLeft;

  moveBlocks.forEach((block, index) => {
    block.classList.toggle("used", index < usedMoves);
  });
}

function setMessage(text, statusClass) {
  message.textContent = text;
  message.className = `message ${statusClass}`.trim();
}

function isInsideGrid(position) {
  return (
    position.row >= 0 &&
    position.row < GRID_SIZE &&
    position.col >= 0 &&
    position.col < GRID_SIZE
  );
}

function isAtExit(position) {
  return position.row === EXIT.row && position.col === EXIT.col;
}

function isBlocked(position) {
  return BLOCKED_CELLS.some((cell) => {
    return cell.row === position.row && cell.col === position.col;
  });
}

function canEnter(position) {
  return isInsideGrid(position) && !isBlocked(position);
}

function movePlayer(direction) {
  if (gameState.isOver || !moves[direction]) {
    return;
  }

  const nextPosition = {
    row: gameState.player.row + moves[direction].row,
    col: gameState.player.col + moves[direction].col,
  };

  if (!canEnter(nextPosition)) {
    return;
  }

  gameState.player = nextPosition;
  gameState.movesLeft -= 1;
  render();

  if (isAtExit(gameState.player)) {
    gameState.isOver = true;
    setMessage("Game successfully completed.", "win");
    return;
  }

  if (gameState.movesLeft === 0) {
    gameState.isOver = true;
    setMessage("Game not completed. Restarting...", "lose");
    window.setTimeout(resetGame, 1400);
    return;
  }
}

function handleKeyboard(event) {
  const keyMoves = {
    ArrowUp: "up",
    w: "up",
    W: "up",
    ArrowDown: "down",
    s: "down",
    S: "down",
    ArrowLeft: "left",
    a: "left",
    A: "left",
    ArrowRight: "right",
    d: "right",
    D: "right",
  };

  const direction = keyMoves[event.key];
  if (!direction) {
    return;
  }

  event.preventDefault();
  movePlayer(direction);
}

function getPublicGameState() {
  return {
    gridSize: GRID_SIZE,
    maxMoves: MAX_MOVES,
    movesLeft: gameState.movesLeft,
    player: { ...gameState.player },
    exit: { ...EXIT },
    blocked: BLOCKED_CELLS.map((cell) => ({ ...cell })),
    isOver: gameState.isOver,
  };
}

// Replace this function tomorrow with your own algorithm.
// It should return one of: "up", "down", "left", or "right".
function chooseAutoMove(state) {
  const blockedKeys = new Set(state.blocked.map((cell) => `${cell.row},${cell.col}`));
  const rowGap = state.exit.row - state.player.row;
  const colGap = state.exit.col - state.player.col;
  const preferredMoves = [];

  if (colGap > 0) {
    preferredMoves.push("right");
  }

  if (rowGap > 0) {
    preferredMoves.push("down");
  }

  if (colGap < 0) {
    preferredMoves.push("left");
  }

  if (rowGap < 0) {
    preferredMoves.push("up");
  }

  preferredMoves.push("right", "down", "left", "up");

  for (const direction of preferredMoves) {
    const delta = moves[direction];
    const nextPosition = {
      row: state.player.row + delta.row,
      col: state.player.col + delta.col,
    };

    const insideGrid =
      nextPosition.row >= 0 &&
      nextPosition.row < state.gridSize &&
      nextPosition.col >= 0 &&
      nextPosition.col < state.gridSize;

    if (insideGrid && !blockedKeys.has(`${nextPosition.row},${nextPosition.col}`)) {
      return direction;
    }
  }

  return null;
}

function runAutoMove() {
  const direction = chooseAutoMove(getPublicGameState());
  if (direction) {
    movePlayer(direction);
  }
}

document.querySelectorAll("[data-move]").forEach((button) => {
  button.addEventListener("click", () => movePlayer(button.dataset.move));
});

restartButton.addEventListener("click", resetGame);
autoButton.addEventListener("click", runAutoMove);
window.addEventListener("keydown", handleKeyboard);

window.escapeGridGame = {
  move: movePlayer,
  reset: resetGame,
  getState: getPublicGameState,
  runAutoMove,
  chooseAutoMove,
};

createBoard();
createMoveStrip();
resetGame();
