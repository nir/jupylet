# The Jupyter MCP tools

The Jupyter MCP tools read, change and run the cells of the attached notebook.
Call each one with "Call a tool", as `SKILL.md` says.

## Contents

- Tools that work
- Web page commands
- Tools that do not work
- Read cells
- Change cells
- Run cells
- After the person restarts the kernel

## Tools that work

Tested on macOS, with `jupyter_server_nbmodel` off:

- `use_notebook` (through "Attach to the notebook"), `list_kernels`,
  `list_notebooks`, `list_files`
- `read_notebook`, `read_cell`
- `insert_cell`, `edit_cell_source`, `overwrite_cell_source`, `move_cell`,
  `delete_cell`, `clear_cell_output`
- `execute_code`: runs code in the kernel, but puts nothing in a cell. Give
  it `kernel_id`.
- `restart_notebook`, `notebook_run-all-cells`, `notebook_get-selected-cell`,
  `notebook_run-cell`, `notebook_move-cursor-down`, `notebook_move-cursor-up`

A change shows in the web page in less than a second.

## Web page commands

Some tools are JupyterLab commands: `notebook_run-all-cells`,
`notebook_run-cell`, `notebook_get-selected-cell` and the cursor moves. The
Jupyter server sends each one to a web page, and the web page does it, as if the
person used the menu. The Jupyter server keeps one list of web pages for all its
notebooks, and sends the command to one of them. With more than one web page, the
choice is arbitrary (`jupyter_mcp_tools`, `handlers.py`, version 0.1.8). The
command acts on the current notebook of that web page, so it can act on another
notebook. That is why the skill keeps one web page on a Jupyter server.

The other tools, for example `read_notebook`, `insert_cell` and
`execute_code`, work in the Jupyter server, without a web page.

## Tools that do not work

`execute_cell` and `insert_execute_code_cell` need `jupyter_server_nbmodel`,
which "Prepare the environment" turns off. Without it, they fail at once.
With it, they stopped responding for minutes. Use `notebook_run-all-cells`,
"Run one cell", or `execute_code` for a short check.

## Read cells

- `read_notebook` needs `notebook_name`, also after attach. For example:
  `{"notebook_name": "{notebook name}", "response_format": "detailed", "limit": 0}`
- `read_cell` returns the source and then the outputs as text, also when
  `include_outputs` is false. For example:
  `{"notebook_name": "{notebook name}", "cell_index": 3, "include_outputs": true}`

## Change cells

- **Do not write the text from `read_cell` back into a cell.** It contains the
  outputs too. To change the source of a cell, read the source from the saved
  `.ipynb` file. The Jupyter server saves it a few seconds after each change.
- **Check the cell before you write over it.** The indices of the cells
  change when the person adds or deletes cells, also between two of your own
  calls. Before each `overwrite_cell_source`, read the cell and check its
  content. If it is not the cell that you expect, stop.
- **Use the cell id when you can.** `overwrite_cell_source`,
  `clear_cell_output` and `move_cell` accept a `cell_id` (the field `id` in
  the saved `.ipynb`). An id stays correct when cells move.
- **Two changes to the same cell must build on each other.** If you write
  twice from the same saved copy, the second change deletes the first.
- **`delete_cell`** takes `cell_ids_to_delete`, a list.
- **`insert_cell`** puts a cell at the index that you give (from 0), and does
  not run it.
- **`overwrite_cell_source`** keeps the old output and execution count until
  the cell runs again.
- **To rewrite a section:** first change each source by its id. Then go
  through the new order: `insert_cell` for a new cell, and `move_cell` for a
  cell that exists. `move_cell` keeps the output with the cell. Then check the
  saved file.

## Run cells

- **All cells:** "Run all cells".
- **One cell, as the person does:** "Run one cell". It needs the option for
  web page commands that "Start JupyterLab" gives. A Jupyter server that
  the person started does not have it: there, run all cells, or ask the
  person to run the cell.
- **Code that is not in a cell:** `execute_code`. Stop a monitoring command
  first: your own run also stops it.

## After the person restarts the kernel

Attach keeps the old kernel. Call `unuse_notebook` with
`{"notebook_name": "{notebook name}"}`, then do "Attach to the notebook"
again.
