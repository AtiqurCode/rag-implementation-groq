const fileList = document.querySelector("#file-list");
const indexCount = document.querySelector("#index-count");
const indexEmpty = document.querySelector("#index-empty");
const indexStatus = document.querySelector("#index-status");
const sourceSelect = document.querySelector("#source");
const form = document.querySelector("#ask-form");
const question = document.querySelector("#question");
const askButton = document.querySelector("#ask");
const result = document.querySelector("#result");
const resultState = document.querySelector("#result-kicker");
const answer = document.querySelector("#answer");
const sources = document.querySelector("#sources");
const sourcesHeading = document.querySelector("#sources-heading");
const citationList = document.querySelector("#citation-list");
const pdfInput = document.querySelector("#pdf-input");
const reindex = document.querySelector("#reindex");

let selectedSource = "";

document.addEventListener("DOMContentLoaded", () => {
  refreshIndex();
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion();
});

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});

pdfInput.addEventListener("change", () => {
  const file = pdfInput.files && pdfInput.files[0];
  pdfInput.value = "";
  if (file) uploadPdf(file);
});

reindex.addEventListener("click", indexFolder);

sourceSelect.addEventListener("change", () => {
  selectedSource = sourceSelect.value;
});

async function refreshIndex() {
  try {
    const stats = await request("/api/stats");
    renderIndex(stats);
  } catch (error) {
    indexCount.textContent = "Index unavailable";
    setStatus(error.message, true);
  }
}

function renderIndex(stats) {
  const previous = selectedSource;
  fileList.replaceChildren();
  const options = sourceSelect.querySelectorAll("option:not(:first-child)");
  options.forEach((option) => option.remove());

  if (!stats.files.length) {
    indexCount.textContent = "Empty";
    indexEmpty.hidden = false;
    return;
  }

  indexEmpty.hidden = true;
  const fileWord = stats.file_count === 1 ? "file" : "files";
  const chunkWord = stats.chunks === 1 ? "chunk" : "chunks";
  indexCount.textContent = `${stats.file_count} ${fileWord}, ${stats.chunks} ${chunkWord}`;

  for (const file of stats.files) {
    const item = document.createElement("li");
    const name = document.createElement("span");
    name.className = "file-name";
    name.textContent = fileName(file.source);
    const meta = document.createElement("span");
    meta.className = "file-meta";
    meta.textContent = `${file.chunks} ${file.chunks === 1 ? "chunk" : "chunks"}`;
    item.append(name, meta);
    fileList.append(item);

    const option = document.createElement("option");
    option.value = file.source;
    option.textContent = fileName(file.source);
    sourceSelect.append(option);
  }

  if ([...sourceSelect.options].some((option) => option.value === previous)) {
    sourceSelect.value = previous;
  } else {
    sourceSelect.value = "";
    selectedSource = "";
  }
}

async function submitQuestion() {
  const text = question.value.trim();
  if (!text) return;
  askButton.disabled = true;
  askButton.textContent = "Reading";
  showResult("Reading the index.", "", false);
  sources.hidden = true;

  const body = {
    question: text,
    source: sourceSelect.value || null,
  };
  const k = document.querySelector("#k").value;
  const cutoff = document.querySelector("#min-relevance").value;
  if (k) body.k = Number(k);
  if (cutoff) body.min_relevance = Number(cutoff);

  try {
    const payload = await request("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderAnswer(payload);
  } catch (error) {
    showResult(error.message, "", true);
  } finally {
    askButton.disabled = false;
    askButton.textContent = "Ask";
  }
}

function renderAnswer(payload) {
  const grounded = payload.answered_from_context;
  showResult(
    grounded ? "From the indexed pages." : "Not answered from the documents.",
    payload.answer,
    !grounded,
  );
  citationList.replaceChildren();
  if (!payload.citations.length) {
    sources.hidden = true;
    return;
  }
  sources.hidden = false;
  sourcesHeading.textContent = grounded ? "Sources" : "Nearest excerpts";
  payload.citations.forEach((citation, index) => {
    const item = document.createElement("li");
    const meta = document.createElement("div");
    meta.className = "cite-meta";
    const where = document.createElement("span");
    where.textContent = `${index + 1}. ${fileName(citation.source)} · p.${citation.page}`;
    const score = document.createElement("span");
    score.className = "cite-score";
    score.textContent = `similarity ${citation.score.toFixed(2)}`;
    meta.append(where, score);
    const snippet = document.createElement("p");
    snippet.className = "cite-text";
    snippet.textContent = clip(citation.text);
    item.append(meta, snippet);
    citationList.append(item);
  });
}

async function uploadPdf(file) {
  setStatus(`Indexing ${file.name}…`, false);
  const body = new FormData();
  body.append("file", file, file.name);
  try {
    const payload = await request("/api/upload", { method: "POST", body });
    setStatus(reportLine(payload.reports), false);
    await refreshIndex();
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function indexFolder() {
  reindex.disabled = true;
  setStatus("Indexing the PDF folder…", false);
  try {
    const payload = await request("/api/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: "data/pdfs" }),
    });
    setStatus(reportLine(payload.reports), false);
    await refreshIndex();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    reindex.disabled = false;
  }
}

function showResult(state, text, warning) {
  result.classList.toggle("is-empty", !text);
  result.classList.remove("is-entering");
  void result.offsetWidth;
  result.classList.add("is-entering");
  resultState.textContent = state;
  resultState.classList.toggle("warning", warning);
  answer.textContent = text;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  result.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
}

function setStatus(message, isError) {
  indexStatus.textContent = message;
  indexStatus.classList.toggle("error", isError);
}

function reportLine(reports) {
  return reports
    .map((report) => `${fileName(report.source)}: ${report.chunks} chunks, ${report.pages} pages`)
    .join(" ");
}

function fileName(source) {
  const parts = source.split(/[\\/]/);
  return parts[parts.length - 1] || source;
}

function clip(text) {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > 280 ? `${flat.slice(0, 277)}…` : flat;
}

async function request(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload.detail;
    const message = typeof detail === "string"
      ? detail
      : "The request failed. Check the API log.";
    throw new Error(message);
  }
  return payload;
}
