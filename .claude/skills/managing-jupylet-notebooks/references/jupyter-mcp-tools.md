# The Jupyter MCP tools

The Jupyter MCP tools read, change and run the cells of the attached notebook.
Call each one with "Call a tool", as `SKILL.md` says. This file has the parts
that this skill needs. The whole file is `references/jupyter-mcp-tools.md` of
the skill `working-in-notebooks`.

## Contents

- Tools that work
- Web page commands
- After the person restarts the kernel

## Tools that work

Tested on macOS, with `jupyter_server_nbmodel` off:

- `use_notebook` (through "Attach to the notebook"), `list_kernels`,
  `list_notebooks`, `list_files`

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

## After the person restarts the kernel

Attach keeps the old kernel. Call `unuse_notebook` with
`{"notebook_name": "{notebook name}"}`, then do "Attach to the notebook"
again.
