# AGENTS.md - Project Contributor Guide

Welcome to this project repository. This file contains the main points for new contributors and AI assistants. 

## Repository overview
Project follows standart golang structure:
- `cmd` - executable files
  - `embedding` - embedding service
  - `mcp` - mcp service
- `dist` - builds
- `docs` - documentation
- `include` - header files
- `indexer` - indexing (python implementation)
- `internal` - internal packages
- `lib` - libraries
- `serverless.yml` - deployment config

## Build
```bash
make build
```

## Testing guidelines
```bash
make test
```