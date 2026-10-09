# The notebook tools

## Contents

- Call a tool
- Tools that work
- Tools that do not work
- Read cells
- Edit cells
- Run cells
- After the person restarts the kernel

## Call a tool

Call each tool the same way, from any folder:

`"{python}" -m jupylet.claude call {port} {token} {tool} '{json arguments}'`

Instead of the JSON, the last argument can be `@{file}` (read the JSON from a
file) or `-` (read it from stdin). On Windows, always use one of these (see
`windows.md`).

`"{python}" -m jupylet.claude tools {port} {token}` lists the tools.

Some tools need the id of the kernel:

`"{python}" -m jupylet.claude kernel {port} {token} {notebook}`

`attach` works for each notebook, not only for `11-spaceship.ipynb`.

## Tools that work

Tested on macOS, with `jupyter_server_nbmodel` off:

- `use_notebook` (through `attach`), `list_kernels`, `list_notebooks`,
  `list_files`
- `read_notebook`, `read_cell`
- `insert_cell`, `edit_cell_source`, `overwrite_cell_source`, `move_cell`,
  `delete_cell`, `clear_cell_output`
- `execute_code`: runs code in the kernel, but puts nothing in a cell. Give
  it `kernel_id`.
- `restart_notebook`, `notebook_run-all-cells`, `notebook_get-selected-cell`
- `notebook_run-cell`, `notebook_move-cursor-down`, `notebook_move-cursor-up`:
  only when Jupyter started with the allowlist option (step 3 in
  `start-in-app.md`).

Changes show in the page in less than a second.

## Tools that do not work

`execute_cell` and `insert_execute_code_cell` need `jupyter_server_nbmodel`,
and `prepare` turns it off. Without it, they fail at once. With it, they
stopped responding for minutes. Use `notebook_run-all-cells`, run one cell
(below), or `execute_code` for a quick check.

## Read cells

- `read_notebook` needs `notebook_name`, also after `attach`. For example:
  `{"notebook_name": "11-spaceship", "response_format": "detailed", "limit": 0}`
- `read_cell` returns the source and then the outputs as text, also when
  `include_outputs` is false. For example:
  `{"notebook_name": "11-spaceship", "cell_index": 3, "include_outputs": true}`

## Edit cells

- **Do not write the text from `read_cell` back into a cell.** It contains
  the outputs too. To change the source of a cell, read the source from the
  saved `.ipynb` file. Jupyter saves it a few seconds after each change.
- **Check the cell before you write over it.** The indices of the cells
  change when the person adds or deletes cells, also between two of your own
  calls. Before each `overwrite_cell_source`, read the cell and check its
  content. If it is not the cell you expect, stop.
- **Use the cell id when you can.** `overwrite_cell_source`,
  `clear_cell_output` and `move_cell` accept a `cell_id` (the field `id` in
  the saved `.ipynb`). An id stays correct when cells move.
- **Two changes to the same cell must build on each other.** If you write
  twice from the same saved copy, the second change deletes the first.
- **`delete_cell`** takes `cell_ids_to_delete`, a list.
- **`insert_cell`** puts a cell at the index you give (from 0), and does not
  run it.
- **`overwrite_cell_source`** keeps the old output and execution count until
  the cell runs again.
- **To rewrite a section,** first change each source by id. Then go through
  the new order: `insert_cell` for a new cell, and `move_cell` for a cell
  that exists. `move_cell` keeps the output with the cell. Then check the
  saved file.

## Run cells

- **All cells:** `notebook_run-all-cells`, with no arguments.
- **One cell, as the person does:**

  `"{python}" -m jupylet.claude run-cell {port} {token} {cell index} "{start of its source}"`

  It moves the selection of the page to the cell, one cell at a time. It
  checks that the source starts with the text you gave, and only then runs
  the cell, because the wrong cell can restart the game. It prints `True`,
  or why it did not run. It moves the cursor of the person. Then read the
  output with `read_cell`.
- **Code that is not in a cell:** `execute_code`. Stop a monitoring command
  first: your own run also stops it.

## After the person restarts the kernel

`attach` keeps the old kernel. Call `unuse_notebook` with
`{"notebook_name": "{notebook name}"}`, then `attach` again.
