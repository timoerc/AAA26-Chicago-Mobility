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
To generate the final report in different formats, use `uv run` to ensure Quarto uses the correct environment:

  ```bash
  uv run quarto render report.qmd
  ```

This will render all formats as specified in `_quarto.yml`. If you want to render individual formats, use the `--to` argument:

- **PDF (Final Submission)**:
  ```bash
  uv run quarto render report.qmd --to pdf
  ```
- **HTML (Interactive Review)**:
  ```bash
  uv run quarto render report.qmd --to html
  ```

> **Note for VS Code Users**: If you use the Quarto VS Code extension, ensure you have the `.venv` selected as your Python interpreter (Cmd/Ctrl + Shift + P -> "Python: Select Interpreter").

### 4. Clean Up
If your preview fails, Quarto might leave temporary files behind. You can clean them up by running:
```bash
rm -f *.quarto_ipynb_*
```
This is also useful for clearing out build artifacts if you want to perform a completely fresh render.

## 📂 Project Structure
- `report.qmd`: The main document where you write your report.
- `sections/`: Sections to include in the main report.
- `notebooks/`: A directory for your exploratory analysis (`.ipynb` files).
- `assets/`: Put your images and logos here.
- `references.bib`: Manage your citations in BibLaTeX format.
- `_quarto.yml`: Project configuration and styling.
- `pyproject.toml`: Project dependency management (use `uv`).
- `partials/`: Custom LaTeX styling (you shouldn't need to touch this).
- `docs/`: The output directory, where your `report.pdf` lives.

## 💡 Key Features
- **Strict Formatting**: The PDF output enforces a 2.5cm margin all around and single line spacing to meet submission requirements.
- **Integrated Analysis**: You can write Python code directly in `report.qmd`.
- **Code Folding**: In the HTML version, code blocks are folded by default to keep the focus on your writing.
- **Margin Content**: Use `#| column: margin` to place small plots or code snippets in the right margin.
- **Embedded Results**: You can embed specific cells from your notebooks into your report using the `{{< embed notebooks/your-notebook.ipynb#cell-label >}}` shortcode. This keeps your main report clean while allowing complex analyses to live in separate files. Note that cells in your notebooks must have `#| label: ...` comments. Note that this does not re-execute cells when you render the report! This takes the latest known output from the source cell. 
- **Modular Report**: For long reports, you can split your document into multiple files (e.g., `sections/01-intro.qmd`) and pull them together using the `{{< include sections/01-intro.qmd >}}` shortcode in your main `report.qmd`.

## 📝 Submission
Your final submission must include:
1. The rendered `report.pdf`.
2. All source files (`.qmd`, `.ipynb`, `.bib`, etc.) in this git repository.
3. The `pyproject.toml` and `uv.lock` files to ensure reproducibility.

---
*Chair of Information Systems for Sustainable Society (IS3)*  
*University of Cologne*
