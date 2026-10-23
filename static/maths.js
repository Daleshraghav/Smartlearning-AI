(() => {
  const data = window.MATHS_FORMULA_DATA || {};
  const library = document.getElementById("formulaLibrary");
  const searchInput = document.getElementById("searchInput");
  const bookFilter = document.getElementById("bookFilter");
  const resetBtn = document.getElementById("resetBtn");
  const resultCount = document.getElementById("resultCount");
  const emptyState = document.getElementById("emptyState");

  const escapeHTML = (value) => String(value).replace(/[&<>"']/g, ch => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"
  }[ch]));

  function render() {
    const query = searchInput.value.trim().toLowerCase();
    const chosenBook = bookFilter.value;
    let total = 0;
    let html = "";

    Object.entries(data).forEach(([bookName, book]) => {
      if (chosenBook !== "all" && chosenBook !== bookName) return;
      const chapters = [];
      let bookCount = 0;

      book.chapters.forEach(chapter => {
        const formulas = chapter.formulas.filter(([title, formula]) =>
          `${bookName} ${chapter.name} ${title} ${formula}`.toLowerCase().includes(query)
        );
        if (!formulas.length) return;
        bookCount += formulas.length;
        const formulaHTML = formulas.map(([title, formula]) => `
          <article class="formula-item">
            <div class="formula-title">${escapeHTML(title)}</div>
            <div class="formula-text">${escapeHTML(formula)}</div>
            <div class="formula-footer">
              <button class="copy-btn" type="button" data-copy="${escapeHTML(formula)}">Copy formula</button>
            </div>
          </article>`).join("");
        chapters.push(`<section class="chapter"><h3>${escapeHTML(chapter.name)}</h3>${formulaHTML}</section>`);
      });

      if (!bookCount) return;
      total += bookCount;
      html += `<article class="book-card">
        <div class="book-heading">
          <div><h2>${escapeHTML(bookName)}</h2><div class="book-subtitle">${escapeHTML(book.subtitle)}</div></div>
          <span class="count-pill">${bookCount} formulas</span>
        </div>
        ${chapters.join("")}
      </article>`;
    });

    library.innerHTML = html;
    resultCount.textContent = `${total} formula entries shown`;
    emptyState.classList.toggle("hidden", total !== 0);
  }

  library.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-copy]");
    if (!button) return;
    const formula = button.getAttribute("data-copy");
    try {
      await navigator.clipboard.writeText(formula);
      button.textContent = "Copied!";
      setTimeout(() => { button.textContent = "Copy formula"; }, 1200);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = formula;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
      button.textContent = "Copied!";
      setTimeout(() => { button.textContent = "Copy formula"; }, 1200);
    }
  });

  searchInput.addEventListener("input", render);
  bookFilter.addEventListener("change", render);
  resetBtn.addEventListener("click", () => {
    searchInput.value = "";
    bookFilter.value = "all";
    render();
  });
  render();
})();
