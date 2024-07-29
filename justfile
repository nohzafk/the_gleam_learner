devcontainer:
    p=$(printf "%s" "$PWD" | hexdump -v -e '/1 "%02x"') && \
    code --folder-uri "vscode-remote://dev-container+${p}/workspaces/$(basename $PWD)"

