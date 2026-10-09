() => {
  // 只在工作台的空闲焦点下接管方向键，让输入、下拉与辅助技术继续使用原生键盘行为。
  document.addEventListener("keydown", (event) => {
    const directions = { ArrowUp: "up", ArrowRight: "right", ArrowDown: "down", ArrowLeft: "left" };
    const direction = directions[event.key];
    if (!direction || event.repeat || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
    if (event.target.closest("input, textarea, select, button, [contenteditable], [role='combobox'], [role='slider'], [role='button']")) return;
    const button = document.getElementById(`r2048-${direction}`);
    if (!button || button.disabled || button.getClientRects().length === 0) return;
    event.preventDefault();
    button.click();
  });
}
