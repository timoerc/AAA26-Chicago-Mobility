# Assignment Template: Team Project

This repository is a Quarto-based template for your team project report. It integrates your data science workflow (Python/Jupyter) directly into a professionally formatted LaTeX/PDF report.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following installed:
- **[Quarto](https://quarto.org/docs/get-started/)**: The publishing system used to render the report.
- **[uv](https://github.com/astral-sh/uv)**: A fast Python package manager.
- **LaTeX**: A TeX distribution (like [TinyTeX](https://yihui.org/tinytex/)) to generate the PDF.
  ```bash
  quarto install tinytex
  ```

### 2. Setup the Environment
We use `uv` to manage dependencies. Run the following commands in the root directory:
```bash
uv sync
uv run python -m ipykernel install --user --name team-project-template --display-name "Python (Team Project Template)"
```
This will create a `.venv` directory and register the Python kernel so Quarto can find it. You can change the `--name` if you prefer, but make sure to select the correct kernel in your editor (e.g., VS Code or Jupyter).

### 3. Rendering the Report
The report must be rendered from inside the `report/` directory so Quarto picks up `_quarto.yml` correctly:

```bash
cd report && uv run quarto render report.qmd
```

If you want to render individual formats, use the `--to` argument:

- **PDF (Final Submission)**:
  ```bash
  cd report && uv run quarto render report.qmd --to pdf
  ```
- **HTML (Interactive Review)**:
  ```bash
  cd report && uv run quarto render report.qmd --to html
  ```

The rendered output lands in `docs/` at the repository root.

> **Note for VS Code Users**: If you use the Quarto VS Code extension, ensure you have the `.venv` selected as your Python interpreter (Cmd/Ctrl + Shift + P -> "Python: Select Interpreter").

### 4. Clean Up
If your preview fails, Quarto might leave temporary files behind. You can clean them up by running:
```bash
cd report && rm -f *.quarto_ipynb_*
```
This is also useful for clearing out build artifacts if you want to perform a completely fresh render.

## 📂 Project Structure
- `report/`: All Quarto report files — `report.qmd`, `_quarto.yml`, `sections/`, `partials/`, `assets/`, `references.bib`.
- `notebooks/`: Jupyter notebooks for exploratory analysis.
- `scripts/helpers/`: Reusable Python modules (`datasets.py`, `preprocessing.py`).
- `scripts/`: One-off utility scripts (data download).
- `data/raw/`: Raw input datasets.
- `docs/`: Rendered output (`report.pdf`).
- `pyproject.toml`: Project dependency management (use `uv`).

## 💡 Key Features
- **Strict Formatting**: The PDF output enforces a 2.5cm margin all around and single line spacing to meet submission requirements.
- **Integrated Analysis**: You can write Python code directly in `report.qmd`.
- **Code Folding**: In the HTML version, code blocks are folded by default to keep the focus on your writing.
- **Margin Content**: Use `#| column: margin` to place small plots or code snippets in the right margin.
- **Embedded Results**: You can embed specific cells from your notebooks into your report using the `{{< embed ../notebooks/your-notebook.ipynb#cell-label >}}` shortcode (note the `../` prefix since `report.qmd` lives in `report/`). Cells must have `#| label: ...` comments. This does not re-execute cells — it takes the latest known output.
- **Modular Report**: For long reports, split your document into multiple files (e.g., `sections/01-intro.qmd`) and pull them together using the `{{< include sections/01-intro.qmd >}}` shortcode in `report.qmd`.

## 📝 Submission
Your final submission must include:
1. The rendered `report.pdf`.
2. All source files (`.qmd`, `.ipynb`, `.bib`, etc.) in this git repository.
3. The `pyproject.toml` and `uv.lock` files to ensure reproducibility.

---
*Chair of Information Systems for Sustainable Society (IS3)*  
*University of Cologne*
